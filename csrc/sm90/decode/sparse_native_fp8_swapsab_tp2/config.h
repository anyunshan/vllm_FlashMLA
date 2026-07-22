#pragma once

#include <cuda_fp8.h>              // __nv_cvt_float_to_fp8 / __NV_SATFINITE / __NV_E4M3 (x448 cast)
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

using namespace cute;

namespace sm90::decode::sparse_native_fp8_swapsab_tp2 {

// ===========================================================================
// KernelTemplate: SM90 sparse decode, NATIVE FP8 x SWAP-AB TP=2 (h_q=32).
// ---------------------------------------------------------------------------
// Fusion of two shipped kernels (see
// docs/native_fp8_swapsab_tp2_decode_plan.md):
//   - native FP8 (csrc/sm90/decode/sparse_native_fp8/): end-to-end fp8, x448
//     cast, 576B/token no-scale KV, token-contiguous sV via producer shuffle
//     transpose, STS.8 scatter, SW64 fp8 layouts.
//   - swap-AB TP2 (csrc/sm90/decode/sparse_fp8_swapsab_tp2/): h_q=32 N=32
//     consumer dataflow -- BMM1 (P^T=K@Q^T), cross-warp softmax, sS^T bridge,
//     D_V 2-way split, BMM2 (O^T=V@S^T), epilogue, split-K.
//
// Technical crux: fp8 wgmma is TN-only (both operands K-major). Swap-AB's two
// gemms both land TN:
//   BMM1  P^T = K @ Q^T   contract = d_qk   -> Q/K d-contiguous  (TN)
//   BMM2  O^T = V @ S^T   contract = token  -> V/S^T token-contig (TN)
// The one thing bf16 swap-AB does that fp8 cannot: the zero-copy V re-view of
// sK. fp8 A=V needs token-contiguous, sK is d-contiguous -> a SEPARATE
// token-contiguous sV buffer (D3), written by the producer's shuffle transpose
// (reused from native FP8).
//
// In-scope: h_q=32, V32 (d_qk=576), B_TOPK=64, cluster=1, s_q=1, 576B/token
// fp8 KV (no scales), no attn_sink / extra_kv / topk_length.
// Out-of-scope (run() asserts): any of the above unmet, h_q != 32, MODEL1.
//
// Status: M1 (scaffolding). config.h layouts + TiledMMA + SharedMemoryPlan +
// save_rS_to_sS are complete and are the compile gate (v32_h32_stub.cu).
// devfunc / run / store_o land in M2-M4.
// ===========================================================================
template<int NUM_HEADS>
class KernelTemplate {
public:

// ----- Scope hard constraints -----
static_assert(NUM_HEADS == 32,
              "native_fp8_swapsab_tp2 only supports h_q=32 (TP=2).");

// ----- Compile-time constants -----
static constexpr int CLUSTER_SIZE = 1;
static constexpr int NUM_M_BLOCKS = 1;   // h_q=32 < BLOCK_M=64 -> one head-block per CTA

static constexpr int HEAD_DIM_K    = 576;  // V32
static constexpr int HEAD_DIM_V    = 512;
static constexpr int HEAD_DIM_ROPE = 64;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_K - HEAD_DIM_ROPE;   // 512

static constexpr int NUM_THREADS     = 384;  // 3 warpgroups x 128 (WG0/WG1 consumers + WG2 producer)
static constexpr int BLOCK_M         = 64;   // wgmma atom M
static constexpr int BLOCK_N         = 32;   // wgmma atom N (= h_q)
static constexpr int TOPK_BLOCK_SIZE = 64;   // B_TOPK
static constexpr int D_V_SPLIT       = HEAD_DIM_V / 2;   // 256 cols per consumer WG
static constexpr int NUM_K_BUFS      = 2;    // K SMEM pipeline depth
static constexpr int NUM_V_BUFS      = 2;    // V SMEM pipeline depth (D3: separate from K)

// x448 cast trick (plan section 2.5) + softmax numerical guard.
static constexpr float MAX_INIT_VAL = -1e30f;   // -inf would NaN scale_for_old; -1e30 -> exp2(0)=1
static constexpr float FP8_P_SCALE  = 448.0f;   // e4m3 max; folded back in o_scale = 1/(448*rL)

// KV cache byte layout (FP8 R5 V32): 512 fp8 NoPE + 64 fp8 RoPE, no scales.
static constexpr int NUM_BYTES_PER_TOKEN = HEAD_DIM_K;   // 576
// Output bf16 STSM/TMA swizzle width (matches native FP8: 64 bf16 = 128B).
static constexpr int OBUF_SW = 64;

// =========================================================================
// SmemLayouts -- fp8 SW64 K-major (native FP8 layout family, swap-AB shapes).
// All are the compile gate for the SW64-fp8 divisibility table (plan section 4).
// =========================================================================

// sQ: BMM1 B operand. (h_q=32, d_qk=576) fp8 SW64 K-major (d contiguous).
using SmemLayoutQTile = decltype(tile_to_shape(
    GMMA::Layout_K_SW64_Atom<fp8>{},
    Shape<Int<NUM_HEADS>, _64>{}                              // (32, 64)
));
using SmemLayoutQ = decltype(tile_to_shape(
    SmemLayoutQTile{},
    Shape<Int<NUM_HEADS>, Int<HEAD_DIM_K>>{},                 // (32, 576)
    Step<_1, _2>{}
));

// sK: BMM1 A operand. (B_TOPK=64, d_qk=576) fp8 SW64 K-major (d contiguous).
// Logical (token, dim): dim 0..511 NoPE, 512..575 RoPE. RoPE feeds QK only.
using SmemLayoutKTile = decltype(tile_to_shape(
    GMMA::Layout_K_SW64_Atom<fp8>{},
    Shape<Int<TOPK_BLOCK_SIZE>, _64>{}                        // (64, 64)
));
using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutKTile{},
    Shape<Int<TOPK_BLOCK_SIZE>, Int<HEAD_DIM_K>>{},           // (64, 576)
    Step<_1, _2>{}
));

// sV: BMM2 A operand. (d_v=512, B_TOPK=64) fp8 SW64 K-major (token contiguous).
// SEPARATE buffer (D3) -- cannot re-view sK under fp8 TN. Producer writes it via
// transpose_8x16_via_shuffle (reused from native FP8). PV consumes NoPE only.
using SmemLayoutVTile = decltype(tile_to_shape(
    GMMA::Layout_K_SW64_Atom<fp8>{},
    Shape<_64, Int<TOPK_BLOCK_SIZE>>{}                        // (64, 64)
));
using SmemLayoutV = decltype(tile_to_shape(
    SmemLayoutVTile{},
    Shape<Int<HEAD_DIM_V>, Int<TOPK_BLOCK_SIZE>>{},           // (512, 64)
    Step<_1, _2>{}
));
// Half-V view: WG0 takes d_v[0:256] (base + 0), WG1 takes d_v[256:512]
//   (base + SmemLayoutV{}(_256{}, _0{}) = 4 * VTile_cosize, a tile boundary).
using SmemLayoutHalfV = decltype(tile_to_shape(
    SmemLayoutVTile{},
    Shape<Int<HEAD_DIM_V/2>, Int<TOPK_BLOCK_SIZE>>{},         // (256, 64)
    Step<_1, _2>{}
));

// sS^T: BMM2 B operand + rP->rS scatter target. (h_q=32, B_TOPK=64) fp8 SW64
// K-major (token contiguous). WG0 scatters rS here; WG0+WG1 read via wgmma SS
// descriptor. This is the (h_q=32, 64) SW64-fp8 divisibility gate (D10).
using SmemLayoutS_T = decltype(tile_to_shape(
    GMMA::Layout_K_SW64_Atom<fp8>{},
    Shape<Int<NUM_HEADS>, Int<TOPK_BLOCK_SIZE>>{}             // (32, 64)
));

// Epilogue staging (union with sK/sV; dead after BMM2).
// sOAccumBuf: (h_q=32, d_v=512) fp32, stride 520 (bank-conflict avoidance).
using SmemLayoutOAccumBuf = Layout<
    Shape<Int<NUM_HEADS>, Int<HEAD_DIM_V>>,
    Stride<Int<520>, _1>
>;
// sOBuf: (h_q=32, d_v=512) bf16 row-major (no-split path; DSA needs bf16).
using SmemLayoutOBuf = Layout<
    Shape<Int<NUM_HEADS>, Int<HEAD_DIM_V>>,
    Stride<Int<HEAD_DIM_V>, _1>
>;

// =========================================================================
// TiledMMA -- both BMM1 and BMM2 use the fp8 N=32 TN atom
// (MMA_64x32x32_F32E4M3E4M3_SS_TN exists: mma_sm90_gmma.hpp:13417).
// fp8 K=32 (bf16 was K=16): BMM1 = 576/32 = 18 K-iters, BMM2 = 64/32 = 2.
// The C-fragment (CLayout_64x32) is K-independent -> softmax / scatter /
// epilogue lane mapping is identical to the bf16 swap-AB TP2 kernel (D6).
// =========================================================================

// BMM1: P^T = K @ Q^T. A=K (M=token=64, K=d), B=Q (N=head=32, K=d). rP (64,32).
using TiledMMA_QK = decltype(make_tiled_mma(
    GMMA::MMA_64x32x32_F32E4M3E4M3_SS_TN<>{},
    Layout<Shape<_1, _1, _1>>{}
));

// BMM2: O^T = V @ S^T. A=V (M=d_v, K=token), B=S^T (N=head=32, K=token).
// M=D_V_SPLIT=256 -> 4 M-tiles per consumer WG. rO (256, 32) = 64 fp32/thread.
using TiledMMA_PV = decltype(make_tiled_mma(
    GMMA::MMA_64x32x32_F32E4M3E4M3_SS_TN<>{},
    Layout<Shape<_1, _1, _1>>{}
));

// =========================================================================
// save_rS_to_sS -- scatter the fp8 rS register fragment (CLayout_64x32, from
// the x448 softmax cast) into sS^T SMEM.
//
// Mapping is the bf16 swap-AB TP2 CLayout_64x32 scatter (dtype-independent, D6):
//   token_row = warp_in_wg*16 + lane_in_warp/4 + 8*r        (r in [0,2))
//   head_col  = my_col_base + em0 + em2*8                    (my_col_base = (lane%4)*2)
// with fragment element (em0, r, em2), em0 in [0,2), em2 in [0,4). Per lane 16
// fp8 cells (2 token rows x 8 head cols).
//
// FP8 vs bf16: value is fp8, store is STS.8 (implicit via sS^T(h,t)=fp8), and
// sS^T is SW64-fp8 (cute &sS^T(h,t) computes the swizzled byte address, F5).
// The mapping does NOT hand-roll addresses -- cute operator() owns the swizzle.
//
// NOTE (verification scope): the wgmma-descriptor read of sS^T (BMM2 B operand)
// consuming what this scatter wrote is closed end-to-end by the M3 PV gemm test,
// NOT by the standalone M1 round-trip (which only checks the formula bijection +
// SW64 layout). See plan section 9.
// =========================================================================
// rS is the fp8[2][8] register array produced by scale_softmax_fp8_swapsab
// (index rS[row_pair][i], i = em0 + em2*2), NOT a cute Tensor -- the two
// components must agree on the rS representation. col index i maps to head_col
// via my_col_base + (i&1) + ((i>>1)*8) = my_col_base + em0 + em2*8.
template<typename TensorSS>
static __forceinline__ __device__ void save_rS_to_sS(
    const fp8 rS[2][8],           // softmax output: [row_pair][i], i=em0+em2*2
    TensorSS &sS_T,               // SmemLayoutS_T (h_q=32, B_TOPK=64) fp8 SW64
    int idx_in_warpgroup
) {
    const int warp_in_wg   = idx_in_warpgroup / 32;
    const int lane_in_warp = idx_in_warpgroup % 32;
    const int my_col_base  = (lane_in_warp % 4) * 2;
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
        int token_row = warp_in_wg * 16 + (lane_in_warp / 4) + 8 * r;
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
            int head_col = my_col_base + (i & 1) + ((i >> 1) * 8);
            sS_T(head_col, token_row) = rS[r][i];
        }
    }
}

// =========================================================================
// SharedMemoryPlan (plan section 6). K + V separate buffers (D3); V barriers
// separate from K (D9). Epilogue staging unioned with the dead K/V mainloop
// buffers.
// =========================================================================
struct SharedMemoryPlan {
    // Q: persistent across CTA lifetime.
    array_aligned<fp8, cosize_v<SmemLayoutQ>> q;

    // K/V pipeline buffers, unioned with epilogue staging (K/V dead after BMM2).
    union {
        struct {
            array_aligned<fp8, cosize_v<SmemLayoutK>> k[NUM_K_BUFS];
            array_aligned<fp8, cosize_v<SmemLayoutV>> v[NUM_V_BUFS];
        } mainloop;
        array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> oAccumBuf;   // split path
        array_aligned<bf16,  cosize_v<SmemLayoutOBuf>>      oBuf;        // no-split path
    } u;

    // sS^T: WG0 scatters rS here (post x448 cast); WG0+WG1 read as BMM2 B operand.
    CUTE_ALIGNAS(1024) array_aligned<fp8, cosize_v<SmemLayoutS_T>> s_t;

    // OOB validity flags per K-block (producer marks; softmax masks -inf).
    bool is_kv_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];

    // Per-head softmax state (h_q=32 each).
    float sM[NUM_HEADS];        // final max per head (LSE)
    float sL[NUM_HEADS];        // final sum per head (LSE)
    float sScale[NUM_HEADS];    // scale_for_old broadcast to WG1 (rO rescale)
    float sOScale[NUM_HEADS];   // final 1/(448*rL) for WG1 epilogue

    // Cross-warp softmax reduction scratch within WG0 (4 warps x 32 heads).
    float colwise_max[4 * NUM_HEADS];
    float colwise_sum[4 * NUM_HEADS];

    // mbarriers
    transac_bar_t bar_q;
    transac_bar_t bar_k_local_ready[NUM_K_BUFS];   // producer -> consumers (K early, D9)
    transac_bar_t bar_k_avail[NUM_K_BUFS];         // consumers -> producer
    transac_bar_t bar_v_local_ready[NUM_V_BUFS];   // producer -> consumers (V later, D9)
    transac_bar_t bar_v_avail[NUM_V_BUFS];         // consumers -> producer
};

// ----- TmaParams -----
template<typename Shape_Q, typename TMA_Q>
struct TmaParams {
    Shape_Q shape_Q;
    TMA_Q   tma_Q;
    CUtensorMap tensor_map_o;
};

// ----- NamedBarriers (sWap-AB 5-id scheme; V handoff uses the mbarriers above) -----
enum NamedBarriers : uint32_t {
    softmax_to_wg1_ready    = 0,   // WG0 sScale + sS^T ready -> WG1 (count 256)
    softmax_buf_free        = 1,   // WG1 done with sS^T -> WG0 may overwrite (count 256)
    o_buf_free_and_sL_ready = 2,   // WG0 published sOScale/sL; sK union releasable (count 256)
    epilogue_r2s_ready      = 3,   // both D_V halves scattered before bulk_copy (count 256)
    wg0_internal_sync       = 4,   // WG0 cross-warp softmax reduction (count 128)
};

// ----- Forward declarations (devfunc / run defined in splitkv_mla.cuh; M2-M4) -----
template<typename TMAParams>
static __device__ __forceinline__ void
devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params);

static void run(const SparseAttnDecodeParams &params);

};  // class KernelTemplate

// ===========================================================================
// scale_softmax_fp8_swapsab -- WG0 online softmax (Step 4) for the fused
// native-FP8 x swap-AB TP2 kernel.
//
// Body is the bf16 swap-AB TP2 cross-warp softmax (softmax axis = M = token,
// spread over 4 warps -> colwise SMEM round-trip + wg0_internal_sync barrier;
// N=32 -> 8 head cols/thread). The ONLY change vs bf16 is the cast (plan D7 /
// section 2.5, path B): the bf16(p_exp) quantize becomes
//   rS_fp8 = cvt_e4m3_SATFINITE(p_exp * 448)
// while cur_rP / rL stay UNSCALED (LSE never sees 448; o_scale = 1/(448*rL)).
//
// This is a namespace-level free function taking raw pointers (not the plan
// struct) so it is unit-testable in isolation (test_scale_softmax_fp8_n32).
//
// Layout contract (CLayout_64x32, matches save_rS_to_sS, plan D6):
//   token_row = warp_in_wg*16 + lane/4 + 8*r         (r in [0,2))
//   head_col  = my_col_base + col_offset[i], col_offset[i]=(i&1)+((i>>1)*8),
//               my_col_base=(lane%4)*2, i in [0,8)  -> {0,1,8,9,16,17,24,25}
//   rP/rO fragment element (em0,r,em2): ci = em0 + em2*2 (col index within thread)
//
// Outputs: rM/rL updated in-place; rO rescaled by scale_for_old; rS filled
// (fp8, [2][8], to be scattered by save_rS_to_sS); sScale[head] written for WG1.
// ===========================================================================
template<int NUM_HEADS, typename TensorRP, typename TensorRO>
static __forceinline__ __device__ void scale_softmax_fp8_swapsab(
    TensorRP &rP,                 // CLayout_64x32 fp32 (16 fp32/lane)
    fp8 rS[2][8],                 // out: quantized P~^T per (row_pair, col_in_thread)
    TensorRO &rO,                 // CLayout_64x32 fp32 PV-Lo half (rescaled in place)
    float sm_scale_div_log2,
    float rM[8],                  // in/out: per-head-col running max (log2 domain)
    float rL[8],                  // in/out: per-head-col running sum (unscaled)
    float sScale[],               // out: scale_for_old broadcast to WG1 (NUM_HEADS)
    float colwise_max[],          // scratch: 4 warps * NUM_HEADS
    float colwise_sum[],          // scratch: 4 warps * NUM_HEADS
    bool  is_kv_valid[],          // NUM_K_BUFS[buf] token validity (B_TOPK)
    uint32_t wg0_internal_nb,     // NamedBarrier id (count=128, WG0 cross-warp)
    int idx_in_warpgroup
) {
    constexpr float FP8_P_SCALE = 448.0f;
    const int warp_in_wg   = idx_in_warpgroup / 32;
    const int lane_in_warp = idx_in_warpgroup % 32;
    const int my_col_base  = (lane_in_warp % 4) * 2;

    // 4.1 OOB mask + within-thread max per head col.
    float thread_max[8];
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) thread_max[i] = -INFINITY;
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
        int token_row = warp_in_wg * 16 + (lane_in_warp / 4) + 8 * r;
        bool valid = is_kv_valid[token_row];
        Tensor cur_rP = flatten(rP(make_coord(_, r, _), _, _));   // size = 8
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            if (!valid) cur_rP(i) = -INFINITY;
            thread_max[i] = max(thread_max[i], cur_rP(i));
        }
    }

    // 4.2 warp-internal 8-lane reduction (mask 4/8/16).
    CUTE_UNROLL
    for (int c = 0; c < 8; ++c) {
        thread_max[c] = max(thread_max[c], __shfl_xor_sync(0xffffffff, thread_max[c],  4));
        thread_max[c] = max(thread_max[c], __shfl_xor_sync(0xffffffff, thread_max[c],  8));
        thread_max[c] = max(thread_max[c], __shfl_xor_sync(0xffffffff, thread_max[c], 16));
    }

    // 4.3 cross-warp via SMEM (lanes 0..3 write 8 cols each).
    if (lane_in_warp < 4) {
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
            int col = my_col_base + (i & 1) + ((i >> 1) * 8);
            colwise_max[warp_in_wg * NUM_HEADS + col] = thread_max[i];
        }
    }
    NamedBarrier(128, wg0_internal_nb).arrive_and_wait();
    float global_max[8];
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        int col = my_col_base + (i & 1) + ((i >> 1) * 8);
        global_max[i] = -INFINITY;
        CUTE_UNROLL
        for (int w = 0; w < 4; ++w)
            global_max[i] = max(global_max[i], colwise_max[w * NUM_HEADS + col]);
    }

    // 4.4 rM update + scale_for_old + rO rescale + rL pre-scale + sScale write.
    float scale_for_old[8];
    CUTE_UNROLL
    for (int c = 0; c < 8; ++c) {
        float cur_scaled = global_max[c] * sm_scale_div_log2;
        float old_max    = rM[c];
        rM[c]            = max(old_max, cur_scaled);
        scale_for_old[c] = exp2f(old_max - rM[c]);
        rL[c]           *= scale_for_old[c];
    }
    CUTE_UNROLL
    for (int m = 0; m < 4; ++m)
        CUTE_UNROLL
        for (int r = 0; r < 2; ++r)
            CUTE_UNROLL
            for (int em2 = 0; em2 < 4; ++em2)
                CUTE_UNROLL
                for (int em0 = 0; em0 < 2; ++em0) {
                    int ci = em0 + em2 * 2;
                    rO(make_coord(em0, r, em2), m, _0{}) *= scale_for_old[ci];
                }
    if (warp_in_wg == 0 && lane_in_warp < 4) {
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
            int col = my_col_base + (i & 1) + ((i >> 1) * 8);
            sScale[col] = scale_for_old[i];
        }
    }

    // 4.6 exp + x448 fp8 cast (path B) + within-thread sum. rS[row_pair][col].
    float thread_sum[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
        Tensor cur_rP = flatten(rP(make_coord(_, r, _), _, _));   // size = 8
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            float p_exp = exp2f(cur_rP(i) * sm_scale_div_log2 - rM[i]);
            rS[r][i] = fp8::bitcast(__nv_cvt_float_to_fp8(
                p_exp * FP8_P_SCALE, __NV_SATFINITE, __NV_E4M3));
            thread_sum[i] += p_exp;
        }
    }

    // 4.7 warp-internal sum (mask 4/8/16) + cross-warp via SMEM -> rL += global.
    CUTE_UNROLL
    for (int c = 0; c < 8; ++c) {
        thread_sum[c] += __shfl_xor_sync(0xffffffff, thread_sum[c],  4);
        thread_sum[c] += __shfl_xor_sync(0xffffffff, thread_sum[c],  8);
        thread_sum[c] += __shfl_xor_sync(0xffffffff, thread_sum[c], 16);
    }
    if (lane_in_warp < 4) {
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
            int col = my_col_base + (i & 1) + ((i >> 1) * 8);
            colwise_sum[warp_in_wg * NUM_HEADS + col] = thread_sum[i];
        }
    }
    NamedBarrier(128, wg0_internal_nb).arrive_and_wait();
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        int col = my_col_base + (i & 1) + ((i >> 1) * 8);
        float gsum = 0.0f;
        CUTE_UNROLL
        for (int w = 0; w < 4; ++w) gsum += colwise_sum[w * NUM_HEADS + col];
        rL[i] += gsum;
    }
}

}  // namespace sm90::decode::sparse_native_fp8_swapsab_tp2
