#pragma once

#include "splitkv_mla.h"

#include <cuda_fp8.h>
#include <math_constants.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "utils.h"
#include "config.h"

// Reused verbatim from the native FP8 kernel (fp8 primitives + producer helpers):
//   gemm<>, launch_tma_copy, load_128b_from_gmem, fp8x16, L1/L2 hints,
//   transpose_8x16_via_shuffle (producer sV write).
#include "../sparse_native_fp8/components/fp8_io.h"
#include "../sparse_native_fp8/components/helpers.h"

using namespace cute;

namespace sm90::decode::sparse_native_fp8_swapsab_tp2 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

// Bring reused fp8 primitives / gemm into this namespace's unqualified scope.
using sm90::decode::sparse_native_fp8::gemm;
using sm90::decode::sparse_native_fp8::launch_tma_copy;
using sm90::decode::sparse_native_fp8::load_128b_from_gmem;
using sm90::decode::sparse_native_fp8::fp8x16;
using sm90::decode::sparse_native_fp8::L1CacheHint;
using sm90::decode::sparse_native_fp8::L2PrefetchHint;

// x448 constant (also in config.h helper; producer does not use it).
static constexpr float MAX_INIT_VAL = -1e30f;

// ===========================================================================
// transpose_8x16_via_shuffle -- copied verbatim from the native FP8 kernel
// (splitkv_mla.cuh); it lives in that .cuh (not a header), so it cannot be
// included without pulling the whole native devfunc. The producer (WG2) is a
// planned copy from native anyway (plan section 1 reuse table), so this helper
// travels with it. Verified byte-exact by native tests/test_producer_sv_shuffle.
//
// In-warp byte transpose of an 8-token x 16-dim block so each STS.64 targets 8
// contiguous token slots of a fixed dim row (sV is token-contiguous).
// ===========================================================================
static __forceinline__ __device__ void transpose_8x16_via_shuffle(uint8_t v[16]) {
    const unsigned full = 0xffffffffu;
    const int lane = threadIdx.x % 32;
    uint32_t* V = reinterpret_cast<uint32_t*>(v);
    // p=0: partner lane^1
    {
        const bool lb = (lane >> 0) & 1;
        uint32_t W[4];
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) W[i] = __shfl_xor_sync(full, V[i], 1, 8);
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i)
            V[i] = lb ? __byte_perm(V[i], W[i], 0x3276) : __byte_perm(V[i], W[i], 0x5410);
    }
    // p=1: partner r^4 = neighbor word
    {
        const bool lb = (lane >> 1) & 1;
        uint32_t W[4];
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) W[i] = __shfl_xor_sync(full, V[i], 2, 8);
        uint32_t n0 = lb ? W[1] : V[0]; uint32_t n1 = lb ? V[1] : W[0];
        uint32_t n2 = lb ? W[3] : V[2]; uint32_t n3 = lb ? V[3] : W[2];
        V[0]=n0; V[1]=n1; V[2]=n2; V[3]=n3;
    }
    // p=2: partner r^8 = +2 words
    {
        const bool lb = (lane >> 2) & 1;
        uint32_t W[4];
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) W[i] = __shfl_xor_sync(full, V[i], 4, 8);
        uint32_t n0 = lb ? W[2] : V[0]; uint32_t n1 = lb ? W[3] : V[1];
        uint32_t n2 = lb ? V[2] : W[0]; uint32_t n3 = lb ? V[3] : W[1];
        V[0]=n0; V[1]=n1; V[2]=n2; V[3]=n3;
    }
    // permute: even r -> low 8, odd r -> high 8
    uint32_t o0 = __byte_perm(V[0], V[1], 0x6420);
    uint32_t o1 = __byte_perm(V[2], V[3], 0x6420);
    uint32_t o2 = __byte_perm(V[0], V[1], 0x7531);
    uint32_t o3 = __byte_perm(V[2], V[3], 0x7531);
    V[0]=o0; V[1]=o1; V[2]=o2; V[3]=o3;
}

// ===========================================================================
// devfunc -- fused native-FP8 x swap-AB TP2 device function.
//
// Step-by-step build (plan section 9 M3 fusion): this commit lands the
// prologue + barrier init + Q TMA + MainloopArgs + the FULL producer (WG2,
// copied from native FP8) + EMPTY WG0/WG1 skeletons. The WG0/WG1 main loops
// (BMM1 + softmax + scatter + BMM2 + epilogue) land in the next step. This
// intermediate state COMPILES (compile gate) but does not compute a result.
// ===========================================================================
template<int NUM_HEADS>
template<typename TMAParams>
__device__ void KernelTemplate<NUM_HEADS>::devfunc(
    const SparseAttnDecodeParams &params, const TMAParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int head_block_idx   = NUM_M_BLOCKS == 1 ? 0 : (int)blockIdx.x;
    const int s_q_idx          = (int)blockIdx.y;
    const int partition_idx    = (int)blockIdx.z;
    const int warpgroup_idx    = cutlass::canonical_warp_group_idx();   // 0/1/2
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx         = cutlass::canonical_warp_idx_sync();
    (void)head_block_idx;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});

    // ----- Step 1: prologue -- prefetch + init mbarriers -----
    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
        plan.bar_q.init(1);
        CUTE_UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
            plan.bar_k_local_ready[i].init(128);   // producer WG2 all-128 arrive
            plan.bar_k_avail[i].init(128);          // only WG0 consumes K after split -> 128
        }
        CUTE_UNROLL
        for (int i = 0; i < NUM_V_BUFS; ++i) {
            plan.bar_v_local_ready[i].init(128);   // producer WG2 all-128 arrive
            plan.bar_v_avail[i].init(256);          // WG0 + WG1 both consume V -> 256
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    int bar_phase_k = 0;   // single phase var: K/V buf_idx advance together (A5)

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    // First Q TMA for the begin batch. h_q=32 < BLOCK_M so there is no
    // head-block split (unlike native h_q=64's flat_divide); take the whole
    // (h_q=32, d_qk) box directly (swap-AB TP2 convention).
    if (warp_idx == 0 && elect_one_sync()) {
        Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx);
        launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
        plan.bar_q.arrive_and_expect_tx(NUM_HEADS * HEAD_DIM_K * (int)sizeof(fp8));
    }

    struct MainloopArgs { int start_block_idx, end_block_idx; bool is_no_split; };
    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
        MainloopArgs args;
        int total_topk_padded = params.topk;
        args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        args.end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx
                                                                 : total_topk_padded / TOPK_BLOCK_SIZE;
        args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted
                         : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
        return args;
    };

    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<192>();

        TiledMMA_QK tiled_mma_QK;
        TiledMMA_PV tiled_mma_PV;
        auto thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
        auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

        Tensor rP = partition_fragment_C(tiled_mma_QK, Shape<Int<TOPK_BLOCK_SIZE>, Int<NUM_HEADS>>{});
        Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<D_V_SPLIT>, Int<NUM_HEADS>>{});
        float rM[8], rL[8];

        const int warp_in_wg   = idx_in_warpgroup / 32;
        const int lane_in_warp = idx_in_warpgroup % 32;
        const int my_col_base  = (lane_in_warp % 4) * 2;

        Tensor sS_T = make_tensor(make_smem_ptr(plan.s_t.data()), SmemLayoutS_T{});

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            CUTE_UNROLL
            for (int c = 0; c < 8; ++c) { rL[c] = 0.0f; rM[c] = MAX_INIT_VAL; }
            clear(rO);
            plan.bar_q.wait((sched_meta.begin_req_idx - batch_idx) & 1);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;

                // Wait sK ready; BMM1 P^T = K @ Q^T (A=K, B=Q, fp8 TN N=32).
                plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
                Tensor sK_buf = make_tensor(make_smem_ptr(plan.u.mainloop.k[buf_idx].data()), SmemLayoutK{});
                gemm</*zero_init=*/true, /*wg_wait=*/-1>(
                    tiled_mma_QK,
                    thr_mma_QK.partition_fragment_A(sK_buf),
                    thr_mma_QK.partition_fragment_B(sQ),
                    rP);
                cute::warpgroup_wait<0>();

                // sK done after BMM1 (separate sV -> release now, D9). count=128 (WG0 only).
                plan.bar_k_avail[buf_idx].arrive();

                // Wait sS^T slot free (after first iter) before overwriting.
                if (block_idx != args.start_block_idx)
                    NamedBarrier(256, NamedBarriers::softmax_buf_free).arrive_and_wait();

                // Step 4: online softmax (x448 cast) -> rS fp8; rescales rO; writes sScale.
                fp8 rS[2][8];
                scale_softmax_fp8_swapsab<NUM_HEADS>(
                    rP, rS, rO, params.sm_scale_div_log2, rM, rL,
                    plan.sScale, plan.colwise_max, plan.colwise_sum,
                    plan.is_kv_valid[buf_idx], NamedBarriers::wg0_internal_sync, idx_in_warpgroup);

                // Step 6: scatter rS -> sS^T (STS.8), fence, signal WG1.
                save_rS_to_sS(rS, sS_T, idx_in_warpgroup);
                fence_view_async_shared();
                NamedBarrier(256, NamedBarriers::softmax_to_wg1_ready).arrive();

                // Step 7+8: wait sV, BMM2-lo O^T = V[0:256] @ S^T.
                plan.bar_v_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
                Tensor sV_lo = make_tensor(make_smem_ptr(plan.u.mainloop.v[buf_idx].data()), SmemLayoutHalfV{});
                gemm</*zero_init=*/false, /*wg_wait=*/-1>(
                    tiled_mma_PV,
                    thr_mma_PV.partition_fragment_A(sV_lo),
                    thr_mma_PV.partition_fragment_B(sS_T),
                    rO);
                cute::warpgroup_wait<0>();
                plan.bar_v_avail[buf_idx].arrive();   // count=256 (WG0 lo + WG1 hi)

                bar_phase_k ^= 1 << buf_idx;
            }

            // Next batch's Q TMA (overlap epilogue).
            if (warp_in_wg == 0 && elect_one_sync()) {
                if (batch_idx != sched_meta.end_req_idx) {
                    Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx + 1);
                    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
                    plan.bar_q.arrive_and_expect_tx(NUM_HEADS * HEAD_DIM_K * (int)sizeof(fp8));
                } else {
                    cudaTriggerProgrammaticLaunchCompletion();
                }
            }

            // ===== Step 9 epilogue (WG0): D_V[0:256] + LSE =====
            // o_scale = 1/(448*rL) (x448 reverse folded in, plan D7).
            float o_scale[8];
            CUTE_UNROLL
            for (int c = 0; c < 8; ++c)
                o_scale[c] = (rL[c] == 0.0f) ? 0.0f : __fdividef(1.0f / FP8_P_SCALE, rL[c]);

            // sOScale broadcast + LSE write (warp 0 lanes 0..3 cover all 32 heads).
            if (warp_in_wg == 0 && lane_in_warp < 4) {
                CUTE_UNROLL
                for (int i = 0; i < 8; ++i) {
                    int col = my_col_base + (i & 1) + ((i >> 1) * 8);
                    plan.sOScale[col] = o_scale[i];
                }
                if (args.is_no_split) {
                    float* lse_base = (float*)params.lse + batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q;
                    CUTE_UNROLL
                    for (int i = 0; i < 8; ++i) {
                        int col = my_col_base + (i & 1) + ((i >> 1) * 8);
                        lse_base[col] = (rL[i] == 0.0f) ? +INFINITY : logf(rL[i]) + rM[i] / (float)M_LOG2E;
                    }
                } else {
                    int n_split_idx = (batch_idx == sched_meta.begin_req_idx) ? sched_meta.begin_split_idx : 0;
                    int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
                    float* lse_base = params.lse_accum + split_idx * params.stride_lse_accum_split + s_q_idx * params.stride_lse_accum_s_q;
                    CUTE_UNROLL
                    for (int i = 0; i < 8; ++i) {
                        int col = my_col_base + (i & 1) + ((i >> 1) * 8);
                        lse_base[col] = (rL[i] == 0.0f) ? -INFINITY : log2f(rL[i]) + rM[i];
                    }
                }
            }

            // Sync: sOScale visible; sK/sV union releasable as sOBuf/sOAccumBuf.
            NamedBarrier(256, NamedBarriers::o_buf_free_and_sL_ready).arrive_and_wait();

            // Scatter rO * o_scale into staging buffer D_V[0:256] half.
            if (args.is_no_split) {
                Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
                CUTE_UNROLL
                for (int m = 0; m < 4; ++m) {
                    int d_v_atom_base = m * 64 + warp_in_wg * 16 + (lane_in_warp / 4);
                    CUTE_UNROLL
                    for (int em1 = 0; em1 < 2; ++em1) {
                        int d_v_local = d_v_atom_base + 8 * em1;   // [0,256)
                        CUTE_UNROLL
                        for (int em2 = 0; em2 < 4; ++em2)
                            CUTE_UNROLL
                            for (int em0 = 0; em0 < 2; ++em0) {
                                int ci = em0 + em2 * 2;
                                int head = my_col_base + em0 + em2 * 8;
                                float val = rO(make_coord(em0, em1, em2), m, _0{}) * o_scale[ci];
                                sOBuf(head, d_v_local) = bf16(val);
                            }
                    }
                }
            } else {
                Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
                CUTE_UNROLL
                for (int m = 0; m < 4; ++m) {
                    int d_v_atom_base = m * 64 + warp_in_wg * 16 + (lane_in_warp / 4);
                    CUTE_UNROLL
                    for (int em1 = 0; em1 < 2; ++em1) {
                        int d_v_local = d_v_atom_base + 8 * em1;
                        CUTE_UNROLL
                        for (int em2 = 0; em2 < 4; ++em2)
                            CUTE_UNROLL
                            for (int em0 = 0; em0 < 2; ++em0) {
                                int ci = em0 + em2 * 2;
                                int head = my_col_base + em0 + em2 * 8;
                                float val = rO(make_coord(em0, em1, em2), m, _0{}) * o_scale[ci];
                                sOAccumBuf(head, d_v_local) = val;
                            }
                    }
                }
            }
            fence_view_async_shared();
            NamedBarrier(256, NamedBarriers::epilogue_r2s_ready).arrive_and_wait();

            // bulk_copy_s2g: WG0 covers rows 0..15 (4 rows/warp).
            if (cute::elect_one_sync()) {
                CUTE_UNROLL
                for (int rj = 0; rj < 4; ++rj) {
                    int row = warp_in_wg * 4 + rj;   // 0..15
                    if (row < (int)NUM_HEADS) {
                        if (args.is_no_split) {
                            bf16* gO_row = (bf16*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q + row * params.stride_o_h_q;
                            Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
                            SM90_BULK_COPY_S2G::copy(&sOBuf(row, _0{}), gO_row, HEAD_DIM_V * (int)sizeof(bf16));
                        } else {
                            int n_split_idx = (batch_idx == sched_meta.begin_req_idx) ? sched_meta.begin_split_idx : 0;
                            int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
                            float* gO_row = params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q + row * params.stride_o_accum_h_q;
                            Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
                            SM90_BULK_COPY_S2G::copy(&sOAccumBuf(row, _0{}), gO_row, HEAD_DIM_V * (int)sizeof(float));
                        }
                    }
                }
                cute::tma_store_arrive();
            }
            cute::tma_store_wait<0>();
            __syncthreads();
        }
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<160>();

        TiledMMA_PV tiled_mma_PV;
        auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
        Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<D_V_SPLIT>, Int<NUM_HEADS>>{});

        const int warp_in_wg   = idx_in_warpgroup / 32;
        const int lane_in_warp = idx_in_warpgroup % 32;
        const int my_col_base  = (lane_in_warp % 4) * 2;

        Tensor sS_T = make_tensor(make_smem_ptr(plan.s_t.data()), SmemLayoutS_T{});

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            clear(rO);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                const int buf_idx = (block_idx - args.start_block_idx) % NUM_V_BUFS;

                // Wait WG0's sScale + sS^T ready (arrive_and_wait: 128 here + 128 WG0's arrive).
                NamedBarrier(256, NamedBarriers::softmax_to_wg1_ready).arrive_and_wait();

                // rO *= scale_for_old (this thread's 8 head cols).
                float scale_for_old[8];
                CUTE_UNROLL
                for (int i = 0; i < 8; ++i) {
                    int col = my_col_base + (i & 1) + ((i >> 1) * 8);
                    scale_for_old[i] = plan.sScale[col];
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

                // Wait sV, BMM2-hi O^T = V[256:512] @ S^T.
                plan.bar_v_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
                fp8* sV_hi_base = plan.u.mainloop.v[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{});
                Tensor sV_hi = make_tensor(make_smem_ptr(sV_hi_base), SmemLayoutHalfV{});
                gemm</*zero_init=*/false, /*wg_wait=*/-1>(
                    tiled_mma_PV,
                    thr_mma_PV.partition_fragment_A(sV_hi),
                    thr_mma_PV.partition_fragment_B(sS_T),
                    rO);
                cute::warpgroup_wait<0>();
                plan.bar_v_avail[buf_idx].arrive();   // count=256 (WG0 lo + WG1 hi)

                // sS^T consumed -> WG0 may overwrite next iter (skip on last block).
                if (block_idx != args.end_block_idx - 1)
                    NamedBarrier(256, NamedBarriers::softmax_buf_free).arrive();

                bar_phase_k ^= 1 << buf_idx;
            }

            // ===== Step 9 epilogue (WG1): D_V[256:512], no LSE (WG0 owns it) =====
            NamedBarrier(256, NamedBarriers::o_buf_free_and_sL_ready).arrive_and_wait();

            float o_scale[8];
            CUTE_UNROLL
            for (int i = 0; i < 8; ++i) {
                int col = my_col_base + (i & 1) + ((i >> 1) * 8);
                o_scale[i] = plan.sOScale[col];
            }

            if (args.is_no_split) {
                Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
                CUTE_UNROLL
                for (int m = 0; m < 4; ++m) {
                    int d_v_atom_base = m * 64 + warp_in_wg * 16 + (lane_in_warp / 4);
                    CUTE_UNROLL
                    for (int em1 = 0; em1 < 2; ++em1) {
                        int d_v_global = D_V_SPLIT + d_v_atom_base + 8 * em1;   // [256,512)
                        CUTE_UNROLL
                        for (int em2 = 0; em2 < 4; ++em2)
                            CUTE_UNROLL
                            for (int em0 = 0; em0 < 2; ++em0) {
                                int ci = em0 + em2 * 2;
                                int head = my_col_base + em0 + em2 * 8;
                                float val = rO(make_coord(em0, em1, em2), m, _0{}) * o_scale[ci];
                                sOBuf(head, d_v_global) = bf16(val);
                            }
                    }
                }
            } else {
                Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
                CUTE_UNROLL
                for (int m = 0; m < 4; ++m) {
                    int d_v_atom_base = m * 64 + warp_in_wg * 16 + (lane_in_warp / 4);
                    CUTE_UNROLL
                    for (int em1 = 0; em1 < 2; ++em1) {
                        int d_v_global = D_V_SPLIT + d_v_atom_base + 8 * em1;
                        CUTE_UNROLL
                        for (int em2 = 0; em2 < 4; ++em2)
                            CUTE_UNROLL
                            for (int em0 = 0; em0 < 2; ++em0) {
                                int ci = em0 + em2 * 2;
                                int head = my_col_base + em0 + em2 * 8;
                                float val = rO(make_coord(em0, em1, em2), m, _0{}) * o_scale[ci];
                                sOAccumBuf(head, d_v_global) = val;
                            }
                    }
                }
            }
            fence_view_async_shared();
            NamedBarrier(256, NamedBarriers::epilogue_r2s_ready).arrive_and_wait();

            // bulk_copy_s2g: WG1 covers rows 16..31 (4 rows/warp).
            if (cute::elect_one_sync()) {
                CUTE_UNROLL
                for (int rj = 0; rj < 4; ++rj) {
                    int row = 16 + warp_in_wg * 4 + rj;   // 16..31
                    if (row < (int)NUM_HEADS) {
                        if (args.is_no_split) {
                            bf16* gO_row = (bf16*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q + row * params.stride_o_h_q;
                            Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
                            SM90_BULK_COPY_S2G::copy(&sOBuf(row, _0{}), gO_row, HEAD_DIM_V * (int)sizeof(bf16));
                        } else {
                            int n_split_idx = (batch_idx == sched_meta.begin_req_idx) ? sched_meta.begin_split_idx : 0;
                            int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
                            float* gO_row = params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q + row * params.stride_o_accum_h_q;
                            Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
                            SM90_BULK_COPY_S2G::copy(&sOAccumBuf(row, _0{}), gO_row, HEAD_DIM_V * (int)sizeof(float));
                        }
                    }
                }
                cute::tma_store_arrive();
            }
            cute::tma_store_wait<0>();
            __syncthreads();
        }
    } else {
        // ----- Step 2: producer (WG2) -- copied from native FP8 producer -----
        // h_q-independent (only reads K cache; does not touch Q). Writes sK
        // (d-contiguous) + sV (token-contiguous via shuffle transpose) + is_kv_valid.
        cutlass::arch::warpgroup_reg_dealloc<152>();

        static_assert(CLUSTER_SIZE == 1, "Phase 1 cluster=1 only.");
        static constexpr int NUM_TOKENS_PER_THREAD = 2;
        static constexpr int NUM_TOKENS_PER_ROUND  = 32;
        int prod_warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);
        int lane_idx      = idx_in_warpgroup % 32;
        int my_token_idx_base = prod_warp_idx * 8 + lane_idx % 8;

        CUTE_NO_UNROLL
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            int* gIndices = params.indices + batch_idx * params.stride_indices_b
                                           + s_q_idx * params.stride_indices_s_q;

            int nxt_token_indexs[NUM_TOKENS_PER_THREAD];
            CUTE_UNROLL
            for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round)
                nxt_token_indexs[round] = __ldg(gIndices + args.start_block_idx * TOPK_BLOCK_SIZE
                                                + round * NUM_TOKENS_PER_ROUND + my_token_idx_base);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
                int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;

                plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
                plan.bar_v_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);

                Tensor sK_t = make_tensor(make_smem_ptr(plan.u.mainloop.k[buf_idx].data()), SmemLayoutK{});
                Tensor sV_t = make_tensor(make_smem_ptr(plan.u.mainloop.v[buf_idx].data()), SmemLayoutV{});

                CUTE_UNROLL
                for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                    int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;
                    int token_index = nxt_token_indexs[round];
                    if (block_idx + 1 != args.end_block_idx)
                        nxt_token_indexs[round] = __ldg(gIndices + (block_idx+1) * TOPK_BLOCK_SIZE + my_token_idx);

                    int block_index      = (token_index == -1) ? 0 : (int)((uint32_t)token_index / (uint32_t)params.page_block_size);
                    int rel_idx_in_block = (uint32_t)token_index % (uint32_t)params.page_block_size;
                    fp8* gK_base = (fp8*)params.kv + block_index * params.stride_kv_block + rel_idx_in_block * params.stride_kv_row;

                    // NoPE: 8 ldg.128/lane; per ldg: 1 STS.128 -> sK, shuffle transpose + 2 STS.64 -> sV.
                    fp8* gK_nope = gK_base + (lane_idx / 8) * 16;
                    int tok0 = my_token_idx - (lane_idx % 8);
                    CUTE_UNROLL
                    for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE / 64; ++dim_idx) {
                        int dim_offset = dim_idx * 64 + (lane_idx / 8) * 16;
                        fp8x16 cur_data = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(
                            gK_nope + dim_idx * 64);
                        if (token_index == -1) *(uint128_t*)&cur_data = uint128_t();

                        fp8* sK_dst = &sK_t(my_token_idx, dim_offset);
                        *(uint128_t*)sK_dst = *(uint128_t*)&cur_data;

                        uint8_t bytes[16];
                        CUTE_UNROLL
                        for (int j = 0; j < 16; ++j) bytes[j] = reinterpret_cast<const uint8_t*>(&cur_data)[j];
                        transpose_8x16_via_shuffle(bytes);
                        int s = lane_idx % 8;
                        int dim_a = dim_offset + 2 * s;
                        int dim_b = dim_a + 1;
                        *reinterpret_cast<uint64_t*>(&sV_t(dim_a, tok0)) = *reinterpret_cast<const uint64_t*>(&bytes[0]);
                        *reinterpret_cast<uint64_t*>(&sV_t(dim_b, tok0)) = *reinterpret_cast<const uint64_t*>(&bytes[8]);
                    }

                    // RoPE: 4-lane subgroup writes sK only (sV has no RoPE).
                    {
                        fp8* gK_rope = gK_base + HEAD_DIM_NOPE + (lane_idx / 8) * 16;
                        int rope_dim_offset = HEAD_DIM_NOPE + (lane_idx / 8) * 16;
                        fp8x16 cur_rope = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope);
                        if (token_index == -1) *(uint128_t*)&cur_rope = uint128_t();
                        fp8* sK_rope_dst = &sK_t(my_token_idx, rope_dim_offset);
                        *(uint128_t*)sK_rope_dst = *(uint128_t*)&cur_rope;
                    }
                }

                fence_view_async_shared();

                if (idx_in_warpgroup < 32) {
                    int2 indices = __ldg((int2*)(gIndices + block_idx * TOPK_BLOCK_SIZE + lane_idx * 2));
                    *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx * 2]) = {
                        (char)(indices.x != -1), (char)(indices.y != -1)
                    };
                }

                plan.bar_k_local_ready[buf_idx].arrive();
                plan.bar_v_local_ready[buf_idx].arrive();
                bar_phase_k ^= (1 << buf_idx);
            }
            __syncthreads();
        }
    }
#else
    if (cute::thread0()) CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
#endif
}

template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
flash_fwd_splitkv_mla_native_fp8_swapsab_tp2_kernel(
    __grid_constant__ const SparseAttnDecodeParams params,
    __grid_constant__ const TMAParams tma_params) {
    Kernel::devfunc(params, tma_params);
}

// ===========================================================================
// run() -- host launcher. Q TMA fp8 (from native); 5D O TMA bf16 (from native);
// grid (NUM_M_BLOCKS, s_q, num_sm_parts) (from swap-AB TP2).
// ===========================================================================
template<int NUM_HEADS>
void KernelTemplate<NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    using Kernel = KernelTemplate<NUM_HEADS>;

    KU_ASSERT(params.h_q == NUM_HEADS, "native_fp8_swapsab_tp2 only supports h_q=32");
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.s_q == 1, "swapsab_tp2 only supports s_q=1");
    KU_ASSERT(params.model_type == ModelType::V32, "native_fp8 supports V32 only");
    KU_ASSERT(params.extra_kv == nullptr, "no extra_kv");
    KU_ASSERT(params.attn_sink == nullptr, "no attn_sink (Phase 1)");
    KU_ASSERT(params.topk_length == nullptr, "no dynamic topk_length");
    KU_ASSERT(params.stride_kv_row == NUM_BYTES_PER_TOKEN,
              "FP8 R5 KV cache = 576 bytes/token (512 fp8 NoPE + 64 fp8 RoPE, no scales)");

    // Q TMA (fp8): shape (h_q, d_qk, s_q, b).
    auto shape_Q = make_shape((int)params.h_q, (int)HEAD_DIM_K, (int)params.s_q, (int)params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((fp8*)params.q),
                    make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{},
                                                     params.stride_q_s_q, params.stride_q_b))),
        SmemLayoutQ{});

    // 5D O TMA (bf16 output for DSA downstream), same form as native FP8.
    CUtensorMap tensor_map_o;
    {
        uint64_t size[5]      = {OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V/OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
        uint64_t stride[4]    = {params.stride_o_h_q*sizeof(bf16), OBUF_SW*sizeof(bf16), params.stride_o_s_q*sizeof(bf16), params.stride_o_b*sizeof(bf16)};
        uint32_t box_size[5]  = {OBUF_SW, BLOCK_M, HEAD_DIM_V/OBUF_SW, 1, 1};
        uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
        constexpr int swizzle_bytes = OBUF_SW * sizeof(bf16);   // 64*2 = 128B
        CUtensorMapSwizzle swizzle =
            swizzle_bytes == 128 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
            swizzle_bytes ==  64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B  :
            swizzle_bytes ==  32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B  :
                                   CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_o, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5,
            params.out, size, stride, box_size, elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    typename Kernel::template TmaParams<decltype(shape_Q), decltype(tma_Q)> tma_params = {
        shape_Q, tma_Q, tensor_map_o
    };
    auto kernel = &flash_fwd_splitkv_mla_native_fp8_swapsab_tp2_kernel<Kernel, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size, params.stream
    };
    cutlass::launch_kernel_on_cluster(launch_params, (void*)kernel, params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

template<int NUM_HEADS>
void run_flash_splitkv_mla_native_fp8_swapsab_tp2_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<NUM_HEADS>::run(params);
}

}  // namespace sm90::decode::sparse_native_fp8_swapsab_tp2
