#pragma once

#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

using namespace cute;

namespace sm90::decode::sparse_fp8_swapsab_tp4 {

// ===========================================================================
// KernelTemplate: SM90 sparse_fp8 swapsab variant for TP=4 (h_q=16)
// ---------------------------------------------------------------------------
// Production TP=4 sparse decode kernel for SM90 (see
// docs/sm90_sparse_fp8_swapsab_tp4_plan.md). Sibling to the TP=8 variant
// (csrc/sm90/decode/sparse_fp8_swapsab_tp8/) -- same swap-AB strategy, but
// h_q=16 so BMM1/BMM2 use the wgmma N=16 atom (CLayout_64x16, per-thread
// 4 head cols x 2 token rows = 8 fp32). Lane-mapping derivation in plan
// Appendix C.1; SMEM budget ~165 KB total (under the 228 KB cap).
//
// Integrates with the standard flash_mla.flash_mla_with_kvcache infra:
//   - Launch grid (NUM_M_BLOCKS=1, s_q, num_sm_parts); persistent batching.
//   - Consumes params.tile_scheduler_metadata_ptr + params.num_splits_ptr.
//   - Writes via both no-split (bf16 -> sOBuf -> bulk_copy_s2g -> params.out)
//     and split (fp32 -> sOAccumBuf -> bulk_copy_s2g -> params.o_accum) paths.
//
// In-scope: h_q=16, V32, B_TOPK=64, cluster=1, s_q=1.
// Out-of-scope (asserted in run()): MODEL1, extra_kv, attn_sink, topk_length,
// s_q>1, h_q != 16.
// ===========================================================================
template<ModelType MODEL_TYPE, int NUM_HEADS>
class KernelTemplate {
public:

// ----- Scope hard constraints -----
static_assert(NUM_HEADS == 16, "swapsab_tp4 only supports h_q=16");
static_assert(MODEL_TYPE == ModelType::V32,
              "swapsab_tp4 only supports V32 (MODEL1 not supported in this variant)");

// ----- Compile-time constants -----
static constexpr int CLUSTER_SIZE = 1;
static constexpr int NUM_M_BLOCKS = 1;   // h_q=16 < BLOCK_M=64 -> one head-block per CTA

static constexpr int HEAD_DIM_K    = 576;  // V32
static constexpr int HEAD_DIM_V    = 512;
static constexpr int HEAD_DIM_ROPE = 64;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_K - HEAD_DIM_ROPE;  // 512

// V32 quantization: 4 float scales per token (1 scale per 128 dims; 4 x 128 = 512 NoPE)
static constexpr int QUANT_TILE_SIZE = 128;
static constexpr int NUM_SCALES      = 4;

static constexpr int NUM_THREADS     = 384;  // 3 warpgroups x 128
static constexpr int BLOCK_M         = 64;   // wgmma atom M dim (both BMM1 and BMM2)
static constexpr int BLOCK_N         = 16;   // wgmma atom N dim (= h_q)
static constexpr int TOPK_BLOCK_SIZE = 64;   // B_TOPK
static constexpr int D_V_SPLIT       = HEAD_DIM_V / 2;  // 256 cols per consumer WG
static constexpr int NUM_K_BUFS      = 2;    // NB (K SMEM pipeline depth)

// ----- SMEM layouts -----

// Q: (h_q=16, D_QK=576) bf16, K-major SW128. Loaded once per batch.
// SW128 atom (8, 64) bf16 -> 16/8 = 2 atoms x 9 atoms = 18 atoms = 18432 bytes.
using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{},
    Shape<Int<NUM_HEADS>, Int<HEAD_DIM_K>>{},
    Step<_1, _2>{}
));

// K: (B_TOPK=64, D_QK=576) bf16, K-major INTER. Producer writes per K-block.
// Same layout family as sparse_fp8 sK -- producer code can be reused.
// Identical to TP=8 (h_q-independent).
using SmemLayoutKTile = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<TOPK_BLOCK_SIZE>, _64>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(tile_to_shape(
    SmemLayoutKTile{},
    Shape<Int<TOPK_BLOCK_SIZE>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
));

using SmemLayoutK = SmemLayoutKTiles<HEAD_DIM_K/64>;  // 9 tiles -> 576 D_QK

// V: composition re-view of K SMEM as (D_V, B_TOPK) MN-major.
// V[d_v, token] physically addresses K[token, d_v]. Identical to TP=8 (D2 inherited).
template<int NUM_TILES>
using SmemLayoutKTilesTransposed = decltype(composition(
    SmemLayoutKTiles<NUM_TILES>{},
    Layout<Shape<Int<64*NUM_TILES>, Int<TOPK_BLOCK_SIZE>>,
           Stride<Int<TOPK_BLOCK_SIZE>, _1>>{}
));

using SmemLayoutV     = SmemLayoutKTilesTransposed<HEAD_DIM_V/64>;     // 8 tiles -> 512 D_V
using SmemLayoutHalfV = SmemLayoutKTilesTransposed<HEAD_DIM_V/64/2>;   // 4 tiles -> 256 D_V_split

// S^T: (h_q=16, B_TOPK=64) bf16, K-major INTER (B_TOPK fast).
// Shape is (h_q, B_TOPK), so mode-0 = N (head), mode-1 = K (token).
// Tile: 2 atoms x 8 atoms along (mode-0, mode-1), total 2048 bytes. See plan SecC.2.
using SmemLayoutS_T = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<NUM_HEADS>, Int<TOPK_BLOCK_SIZE>>{},
    Step<_1, _2>{}
));

// Output accumulator staging buffer: (h_q=16, D_V=512) fp32.
// Stride 520 (= 512 + 8 padding) for bank-conflict avoidance, matches sparse_fp8 / TP=8 pattern.
// cosize: (16-1)*520 + 512 = 8312 fp32 = 33248 bytes.
using SmemLayoutOAccumBuf = Layout<
    Shape<Int<NUM_HEADS>, Int<HEAD_DIM_V>>,
    Stride<Int<520>, _1>
>;

// Output bf16 staging buffer (no-split path): (h_q=16, D_V=512) bf16, row-major.
// Used when args.is_no_split == true; bulk_copy_s2g per row writes to params.out.
// cosize: 16 * 512 = 8192 bf16 = 16384 bytes.
using SmemLayoutOBuf = Layout<
    Shape<Int<NUM_HEADS>, Int<HEAD_DIM_V>>,
    Stride<Int<HEAD_DIM_V>, _1>
>;

// ----- TiledMMA atoms -----

// BMM1: K @ Q^T -> P^T. Atom <M=64 token, N=16 head, K=16>, 36 K-iters over D_QK=576.
// Single wgmma per K-iter. Per-thread rP = CLayout_64x16 element shape (2,2,2)
// = 8 fp32 (4 head cols x 2 token rows). See plan Appendix C.1.
using TiledMMA_QK = decltype(make_tiled_mma(
    GMMA::MMA_64x16x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<_1, _1, _1>>{}
));

// BMM2: V @ S^T -> O^T. Atom <M=64 D_V, N=16 head, K=16>, 4 K-iters over B_TOPK=64.
// D_V_split=256 -> 4 M-tiles per consumer WG. Per-thread rO_WG = 4 atoms x 8 fp32 = 32 fp32.
using TiledMMA_PV = decltype(make_tiled_mma(
    GMMA::MMA_64x16x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::K>{},
    Layout<Shape<_1, _1, _1>>{}
));

// ----- SharedMemoryPlan -----
struct SharedMemoryPlan {
    // Q: persistent across CTA lifetime.
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;

    // K pipeline buffers (also re-viewed as sV).
    // Epilogue reuses the same physical SMEM as fp32 sOAccumBuf (split path)
    // or bf16 sOBuf (no-split path) via union (sK is no longer needed after
    // main K-block loop).
    union {
        array_aligned<bf16, cosize_v<SmemLayoutK>> k[NUM_K_BUFS];
        array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> oAccumBuf;
        array_aligned<bf16,  cosize_v<SmemLayoutOBuf>>      oBuf;
    } u;

    // S^T scratch: written by WG0 after softmax, read by WG0+WG1 for BMM2.
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutS_T>> s_t;

    // OOB validity flags per K-block (producer marks; softmax masks invalid tokens to -inf).
    bool is_kv_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];

    // Per-head softmax state (16 values each for h_q=16).
    float sM[NUM_HEADS];        // final max per head (for LSE write)
    float sL[NUM_HEADS];        // final sum per head (for LSE write)
    float sScale[NUM_HEADS];    // scale_for_old (= exp(old_max - new_max)) for WG1 rO rescale
    float sOScale[NUM_HEADS];   // final 1/li for WG1 epilogue rescale

    // Cross-warp softmax reduction scratch within WG0 (4 warps x 16 heads = 64 entries).
    float colwise_max[4 * NUM_HEADS];
    float colwise_sum[4 * NUM_HEADS];

    // mbarriers
    transac_bar_t bar_q;
    transac_bar_t bar_k_local_ready[NUM_K_BUFS];  // producer -> consumers
    transac_bar_t bar_k_avail[NUM_K_BUFS];        // consumers -> producer
};

// ----- TmaParams -----
template<typename Shape_Q, typename TMA_Q>
struct TmaParams {
    Shape_Q shape_Q;
    TMA_Q   tma_Q;
};

// ----- NamedBarriers -----
enum NamedBarriers : uint32_t {
    softmax_to_wg1_ready    = 0,
    softmax_buf_free        = 1,
    o_buf_free_and_sL_ready = 2,
    epilogue_r2s_ready      = 3,
    wg0_internal_sync       = 4,   // WG0 internal cross-warp softmax sync (count=128)
};

// ----- Constants -----
// Persistent rM sentinel: -inf would NaN out scale_for_old=exp(old-new)
// when both maxes are still sentinel (-inf - -inf = NaN). Use -1e30 instead;
// (-1e30 - -1e30) = 0 -> exp2(0) = 1 -> li unchanged on first K-block, correct.
// Matches sparse_fp8 / TP=8 convention.
static constexpr float MAX_INIT_VAL = -1e30f;

// ----- Forward declarations -----
template<typename TMAParams>
static __device__ __forceinline__ void
devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params);

static void run(const SparseAttnDecodeParams &params);

};  // class KernelTemplate

}  // namespace sm90::decode::sparse_fp8_swapsab_tp4
