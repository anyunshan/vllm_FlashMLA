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
#include "../sparse_fp8/components/dequant.h"
#include "../sparse_fp8/components/helpers.h"
#include "config.h"

using namespace cute;

namespace sm90::decode::sparse_fp8_swapsab_tp4 {

// Bring in shared utilities (launch_tma_copy, gemm<>, fp8x8 / cvt_fp8x8_bf16x8
// dequant helpers, etc.) from the sparse_fp8 variant. The swap-AB path does
// not use get_AorC_row_idx (it computes m_base inline since the fragment row
// corresponds to token, not head -- see plan Sec 3.1).
using namespace sm90::decode::sparse_fp8;

// ===========================================================================
// devfunc -- main device function: production multi-batch persistent driver.
// All 9 Steps (Q TMA, K production, BMM1, softmax, mid-loop rescale,
// sS^T scatter, V re-view, BMM2, epilogue) are implemented per plan Sec 5.
// ===========================================================================
template<ModelType MODEL_TYPE, int NUM_HEADS>
template<typename TMAParams>
__device__ void
KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(
    const SparseAttnDecodeParams &params,
    const TMAParams              &tma_params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))

    // -------------------------------------------------------------------
    // Grid (NUM_M_BLOCKS=1, s_q, num_sm_parts); cluster=1. Mirrors
    // sparse_fp8 / sparse_fp8_swapsab_tp8.
    // -------------------------------------------------------------------
    const int head_block_idx   = NUM_M_BLOCKS == 1 ? 0 : (int)blockIdx.x;
    const int s_q_idx          = (int)blockIdx.y;
    const int partition_idx    = (int)blockIdx.z;
    (void)head_block_idx;  // unused for h_q < BLOCK_M; kept for symmetry

    const int warpgroup_idx    = cutlass::canonical_warp_group_idx();   // 0 / 1 / 2
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx         = cutlass::canonical_warp_idx_sync();    // 0 .. 11

    // -------------------------------------------------------------------
    // SMEM access
    // -------------------------------------------------------------------
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
    // sS_T / sOAccumBuf / sOBuf views are created later inside the steps that use them.

    // -------------------------------------------------------------------
    // Step 1 -- Prologue: prefetch + init mbarriers
    // -------------------------------------------------------------------
    if (warp_idx == 0 && elect_one_sync()) {
        // Prefetch Q TMA descriptor into TMA cache.
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());

        // Initialize mbarriers.
        plan.bar_q.init(1);                       // single TMA issuer thread
        CUTE_UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
            plan.bar_k_local_ready[i].init(128);  // WG2 (producer) all-128 arrives
            plan.bar_k_avail[i].init(256);        // WG0 + WG1 each-128 arrives
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();   // ensure barriers visible to all threads before use

    // Consumer-side phase tracker for sK[buf]. Persistent across batches.
    int bar_phase_k = 0;

    // -------------------------------------------------------------------
    // Read tile scheduler metadata for this partition. Empty range -> early exit.
    // -------------------------------------------------------------------
    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    // First Q TMA load for the begin batch in this partition's range.
    if (warp_idx == 0 && elect_one_sync()) {
        Tensor gQ_all = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q);
        Tensor gQ     = gQ_all(_, _, s_q_idx, sched_meta.begin_req_idx);   // (h_q, D_QK)
        launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q,
                        TMA::CacheHintSm90::EVICT_FIRST);
        plan.bar_q.arrive_and_expect_tx(NUM_HEADS * HEAD_DIM_K * (int)sizeof(bf16));
    }

    // -------------------------------------------------------------------
    // MainloopArgs: per-batch begin/end K-block range + is_no_split flag.
    // V32 only -> total_topk_padded = params.topk; no MODEL1 logic needed.
    // -------------------------------------------------------------------
    struct MainloopArgs {
        int  start_block_idx;
        int  end_block_idx;
        bool is_no_split;
    };
    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
        MainloopArgs args;
        const int total_topk_padded = params.topk;
        args.start_block_idx = (batch_idx == sched_meta.begin_req_idx)
            ? sched_meta.begin_block_idx : 0;
        args.end_block_idx = (batch_idx == sched_meta.end_req_idx)
            ? sched_meta.end_block_idx : (total_topk_padded / TOPK_BLOCK_SIZE);
        args.is_no_split = (batch_idx == sched_meta.begin_req_idx)
            ? !sched_meta.is_first_req_splitted
            : ((batch_idx == sched_meta.end_req_idx)
                ? !sched_meta.is_last_req_splitted
                : true);
        return args;
    };

    // ===================================================================
    // Warpgroup dispatch.
    //   WG0 -- consumer_A: BMM1 + softmax + BMM2 D_V[0:256] + epilogue first half
    //   WG1 -- consumer_B: BMM2 D_V[256:512] + epilogue second half
    //   WG2 -- producer:   TMA gather4 fp8 K -> dequant -> write sK
    // ===================================================================
    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<192>();

        // ===============================================================
        // Per-CTA RF state setup (persistent across batches in this CTA's
        // [begin_req_idx, end_req_idx] range)
        // ===============================================================
        TiledMMA_QK tiled_mma_QK;
        TiledMMA_PV tiled_mma_PV;
        auto thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
        auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

        // rP: BMM1 output P^T. Shape (B_TOPK=64 token, h_q=16 head).
        // Per thread (CLayout_64x16): 8 fp32 = 4 head cols x 2 token rows
        // (col offsets {0,1,8,9} from my_col_a = (lane%4)*2). Overwritten per K-block.
        Tensor rP = partition_fragment_C(
            tiled_mma_QK,
            Shape<Int<TOPK_BLOCK_SIZE>, Int<NUM_HEADS>>{});

        // rO: BMM2 accumulator O^T. Shape (D_V_split=256, h_q=16).
        // Per thread: 32 fp32 = 4 M-tiles x 4 head cols x 2 D_V rows.
        // Reset to 0 per batch inside the batch loop below.
        Tensor rO = partition_fragment_C(
            tiled_mma_PV,
            Shape<Int<D_V_SPLIT>, Int<NUM_HEADS>>{});

        // Online softmax state per thread. Each thread holds NUM_HEADS / 4 = 4
        // head cols (col offsets {0,1,8,9} from my_col_base). Reset per batch.
        // rM init: MAX_INIT_VAL (-1e30) not -inf -- see config.h.
        float rM[4];
        float rL[4];

        // WG0 thread layout (used by Step 4 cross-lane reduction + Steps 6/9 scatter).
        // Per-thread wgmma C fragment (CLayout_64x16): 2 token rows x 4 head cols
        // = 8 fp32 elements. Token rows: warp_in_wg*16 + lane/4 + 8*row_pair_idx.
        // Head cols: my_col_base + col_offset[i] for i in [0,4) and col_offset[i]
        // = (i&1) + ((i>>1)*8) yielding {0,1,8,9} relative to my_col_base.
        const int warp_in_wg   = idx_in_warpgroup / 32;
        const int lane_in_warp = idx_in_warpgroup % 32;
        const int my_col_base  = (lane_in_warp % 4) * 2;   // first head col of this thread's group

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx;
             batch_idx <= sched_meta.end_req_idx;
             ++batch_idx) {

        MainloopArgs args = get_cur_req_info(batch_idx);

        // Reset per-batch RF state.
        CUTE_UNROLL
        for (int c = 0; c < 4; ++c) { rL[c] = 0.0f; rM[c] = MAX_INIT_VAL; }
        clear(rO);

        // Wait for this batch's Q to land. Phase alternates per batch:
        //   batch_idx == begin_req_idx -> phase 0 (initial load)
        //   each subsequent batch toggles phase.
        plan.bar_q.wait((sched_meta.begin_req_idx - batch_idx) & 1);

            // -----------------------------------------------------------
            // Main K-block loop (per batch)
            // -----------------------------------------------------------
            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;

                // Wait sK[buf] to be filled by WG2 (Step 2).
                plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);

                // -------------------------------------------------------
                // Step 3: BMM1 -- K @ Q^T -> P^T  (rP in RF)
                //
                // Atom <M=64, N=16, K=16>, both K-major SS. K-iters = 36 (576/16).
                // Single wgmma per K-iter. zero_init=true: rP overwritten
                // each K-block (no carry-over). wg_wait=-1: manual
                // warpgroup_wait<0> below since softmax (Step 4) needs rP.
                // -------------------------------------------------------
                Tensor sK_buf = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()),
                                            SmemLayoutK{});
                gemm</*zero_init=*/true, /*wg_wait=*/-1>(
                    tiled_mma_QK,
                    thr_mma_QK.partition_fragment_A(sK_buf),   // A = K
                    thr_mma_QK.partition_fragment_B(sQ),       // B = Q
                    rP
                );
                bar_phase_k ^= 1 << buf_idx;
                cute::warpgroup_wait<0>();
                // rP now holds the BMM1 output for this K-block. sK[buf]
                // still held; released only after BMM2 in Step 8.

                // -------------------------------------------------------
                // Step 4: Online softmax on rP -> rS bf16  (also rescales rO,
                // updates rM/rL, broadcasts scale_for_old to WG1 via sScale[])
                //
                // Per-thread fragment layout (CLayout_64x16, element shape
                // (2,2,2), M_TILE=1, N_TILE=1) -> 8 fp32 elements per thread
                // (2 token rows x 4 head cols). Reduction axis is M (token),
                // per head col. Within-thread max+sum span 2 row_pair entries;
                // warp-internal reduction across 8 lanes (lane%4 fixed) via
                // shfl_xor mask 4/8/16; cross-warp via SMEM (colwise_max /
                // colwise_sum, NamedBarrier).
                //
                // flatten(rP(make_coord(_, local_row_idx, _), _, _)) at fixed
                // em1=local_row_idx yields size = 4 (em0=2 x em2=2). The 4
                // elements correspond to col offsets {0,1,8,9} from
                // my_col_base, indexed by i in [0,4) where (em0, em2) =
                // (i&1, i>>1).
                // -------------------------------------------------------

                // Step 4.1: OOB mask + within-thread max per head col.
                float thread_max[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
                CUTE_UNROLL
                for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
                    int token_row = warp_in_wg * 16 + (lane_in_warp / 4)
                                  + 8 * local_row_idx;
                    bool valid = plan.is_kv_valid[buf_idx][token_row];
                    Tensor cur_rP = flatten(
                        rP(make_coord(_, local_row_idx, _), _, _));   // size = 4
                    CUTE_UNROLL
                    for (int i = 0; i < size(cur_rP); ++i) {
                        if (!valid) cur_rP(i) = -INFINITY;
                        thread_max[i] = max(thread_max[i], cur_rP(i));
                    }
                }

                // Step 4.2: warp-internal 8-lane reduction (mask 4 / 8 / 16).
                CUTE_UNROLL
                for (int c = 0; c < 4; ++c) {
                    thread_max[c] = max(thread_max[c],
                        __shfl_xor_sync(0xffffffff, thread_max[c],  4));
                    thread_max[c] = max(thread_max[c],
                        __shfl_xor_sync(0xffffffff, thread_max[c],  8));
                    thread_max[c] = max(thread_max[c],
                        __shfl_xor_sync(0xffffffff, thread_max[c], 16));
                }

                // Step 4.3: cross-warp via SMEM scratch.
                //   Write phase: lane_in_warp < 4 of each warp writes 4 cols
                //   (col offsets {0,1,8,9} from my_col_base) to colwise_max.
                //   Each (warp, col) cell has a unique writer.
                if (lane_in_warp < 4) {
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        int col_offset = (i & 1) + ((i >> 1) * 8);   // {0,1,8,9}
                        int col = my_col_base + col_offset;
                        plan.colwise_max[warp_in_wg * NUM_HEADS + col] = thread_max[i];
                    }
                }
                cutlass::arch::NamedBarrier(128, NamedBarriers::wg0_internal_sync)
                    .arrive_and_wait();

                //   Read phase: every thread reads 4 warp partials per col.
                float global_max[4];
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    int col_offset = (i & 1) + ((i >> 1) * 8);
                    int col = my_col_base + col_offset;
                    global_max[i] = -INFINITY;
                    CUTE_UNROLL
                    for (int w = 0; w < 4; ++w) {
                        global_max[i] = max(global_max[i],
                                            plan.colwise_max[w * NUM_HEADS + col]);
                    }
                }

                // Step 4.4: update rM, compute scale_for_old, rescale rO + pre-scale rL.
                float scale_for_old[4];
                CUTE_UNROLL
                for (int c = 0; c < 4; ++c) {
                    float cur_max_scaled = global_max[c] * params.sm_scale_div_log2;
                    float old_max        = rM[c];
                    rM[c]                = max(old_max, cur_max_scaled);
                    scale_for_old[c]     = exp2f(old_max - rM[c]);
                    rL[c]               *= scale_for_old[c];
                }

                // Rescale rO (32 fp32 = 4 M-tiles x 2 row_pair x 4 col).
                // Per-col index c = (em0, em2) -> ci = em0 + em2*2 in flatten.
                CUTE_UNROLL
                for (int m = 0; m < 4; ++m) {
                    CUTE_UNROLL
                    for (int r = 0; r < 2; ++r) {
                        CUTE_UNROLL
                        for (int em2 = 0; em2 < 2; ++em2) {
                            CUTE_UNROLL
                            for (int em0 = 0; em0 < 2; ++em0) {
                                int ci = em0 + em2 * 2;
                                rO(make_coord(em0, r, em2), m, _0{}) *= scale_for_old[ci];
                            }
                        }
                    }
                }

                // Step 4.5: write sScale[head] for WG1 (broadcast scale_for_old).
                // Only warp_in_wg == 0 lanes 0..3 write -- they collectively
                // cover all 16 head cols (4 per lane).
                if (warp_in_wg == 0 && lane_in_warp < 4) {
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        int col_offset = (i & 1) + ((i >> 1) * 8);
                        plan.sScale[my_col_base + col_offset] = scale_for_old[i];
                    }
                }

                // Step 4.6: exp + bf16 quantize + within-thread sum per col.
                // bf16 cast is direct (no FP8_P_SCALE) since p_exp in [0, 1].
                // rS layout [row_pair=2][col_in_thread=4] matches the cur_rP
                // flatten ordering used in scatter Step 6.
                bf16  rS[2][4];
                float thread_sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                CUTE_UNROLL
                for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
                    Tensor cur_rP = flatten(
                        rP(make_coord(_, local_row_idx, _), _, _));   // size = 4
                    CUTE_UNROLL
                    for (int i = 0; i < size(cur_rP); ++i) {
                        float p_exp = exp2f(cur_rP(i) * params.sm_scale_div_log2
                                            - rM[i]);
                        rS[local_row_idx][i] = bf16(p_exp);
                        thread_sum[i] += p_exp;
                    }
                }

                // Step 4.7: warp-internal sum reduction (mask 4 / 8 / 16) +
                // cross-warp via SMEM (same pattern as max).
                CUTE_UNROLL
                for (int c = 0; c < 4; ++c) {
                    thread_sum[c] += __shfl_xor_sync(0xffffffff, thread_sum[c],  4);
                    thread_sum[c] += __shfl_xor_sync(0xffffffff, thread_sum[c],  8);
                    thread_sum[c] += __shfl_xor_sync(0xffffffff, thread_sum[c], 16);
                }
                if (lane_in_warp < 4) {
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        int col_offset = (i & 1) + ((i >> 1) * 8);
                        int col = my_col_base + col_offset;
                        plan.colwise_sum[warp_in_wg * NUM_HEADS + col] = thread_sum[i];
                    }
                }
                cutlass::arch::NamedBarrier(128, NamedBarriers::wg0_internal_sync)
                    .arrive_and_wait();

                // Step 4.8: read 4 warp partials per col, update rL.
                float global_sum[4];
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    int col_offset = (i & 1) + ((i >> 1) * 8);
                    int col = my_col_base + col_offset;
                    global_sum[i] = 0.0f;
                    CUTE_UNROLL
                    for (int w = 0; w < 4; ++w) {
                        global_sum[i] += plan.colwise_sum[w * NUM_HEADS + col];
                    }
                    rL[i] += global_sum[i];   // rL was pre-scaled in Step 4.4
                }

                // -------------------------------------------------------
                // Step 6: scatter rS bf16 to sS_T SMEM, signal WG1.
                //
                // sS_T physical layout: (h_q=16, B_TOPK=64) K-major INTER.
                // Each thread writes 8 bf16 to (head_col, token_row) for
                // token_row in {m_base, m_base+8} and head_col offsets
                // {0,1,8,9} from my_col_base. 8 STS.16 per thread (no
                // vectorize: the 8 positions span non-contiguous addresses
                // in this INTER atom layout).
                // -------------------------------------------------------

                // Wait for sS_T slot to be free (only after first iter --
                // first iter has no producer/consumer cycle yet on this
                // NamedBarrier).
                if (block_idx != args.start_block_idx) {
                    cutlass::arch::NamedBarrier(
                        256, NamedBarriers::softmax_buf_free).arrive_and_wait();
                }

                Tensor sS_T = make_tensor(make_smem_ptr(plan.s_t.data()),
                                          SmemLayoutS_T{});
                CUTE_UNROLL
                for (int r = 0; r < 2; ++r) {
                    int token_row = warp_in_wg * 16 + (lane_in_warp / 4) + 8 * r;
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        int col_offset = (i & 1) + ((i >> 1) * 8);
                        int head_col = my_col_base + col_offset;
                        sS_T(head_col, token_row) = rS[r][i];
                    }
                }

                // Make generic SMEM stores visible to wgmma async proxy
                // (BMM2 in Step 8 reads sS_T via wgmma B operand SMEM descriptor).
                fence_view_async_shared();

                // Signal WG1: sScale + sS_T are ready. arrive() is non-blocking
                // (count=256 = 128 from WG0 here + 128 from WG1's arrive_and_wait).
                cutlass::arch::NamedBarrier(
                    256, NamedBarriers::softmax_to_wg1_ready).arrive();

                // -------------------------------------------------------
                // Step 7 + 8: V re-view (no data movement) + BMM2 V @ S^T -> rO
                //
                // Atom MMA_64x16x16 SS<MN, K>: A=V (MN-major, D_V fast),
                // B=S^T (K-major, B_TOPK fast). D_V_SPLIT=256 -> 4 M-tiles.
                // B_TOPK=64 -> 4 K-iters = 16 wgmma instructions per WG.
                // accumulate (no zero_init); rO already pre-rescaled by
                // scale_for_old in Step 4.4 to new max basis.
                //
                // WG0 covers D_V[0:256] half (sV_view base = sK[buf] start).
                // -------------------------------------------------------
                Tensor sV_view = make_tensor(
                    make_smem_ptr(plan.u.k[buf_idx].data()),
                    SmemLayoutHalfV{});

                gemm</*zero_init=*/false, /*wg_wait=*/-1>(
                    tiled_mma_PV,
                    thr_mma_PV.partition_fragment_A(sV_view),   // A = V (MN-major)
                    thr_mma_PV.partition_fragment_B(sS_T),      // B = S^T (K-major)
                    rO);
                cute::warpgroup_wait<0>();

                // Release sK[buf_idx] slot -- producer can overwrite (count=256
                // contribution: 128 from WG0 here + 128 from WG1's arrive after
                // its own BMM2).
                plan.bar_k_avail[buf_idx].arrive();
            }

        // ===============================================================
        // Issue next batch's Q TMA load (overlaps with this batch's epilogue).
        // ===============================================================
        if (warp_in_wg == 0 && elect_one_sync()) {
            if (batch_idx != sched_meta.end_req_idx) {
                Tensor gQ_all = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q);
                Tensor gQ     = gQ_all(_, _, s_q_idx, batch_idx + 1);
                launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q,
                                TMA::CacheHintSm90::EVICT_FIRST);
                plan.bar_q.arrive_and_expect_tx(NUM_HEADS * HEAD_DIM_K * (int)sizeof(bf16));
            } else {
                // Last batch this CTA handles -- signal PDL for combine kernel.
                cudaTriggerProgrammaticLaunchCompletion();
            }
        }

        // ===============================================================
        // Step 9 epilogue (WG0): D_V[0:256] half + LSE write
        // ===============================================================

        // 9.1: o_scale per head col (4 entries per thread).
        float o_scale[4];
        CUTE_UNROLL
        for (int c = 0; c < 4; ++c) {
            o_scale[c] = (rL[c] == 0.0f) ? 0.0f : __fdividef(1.0f, rL[c]);
        }

        // 9.2 + 9.3: sOScale broadcast + LSE write.
        // warp 0 lanes 0..3 each cover 4 head cols (col offsets {0,1,8,9}).
        if (warp_in_wg == 0 && lane_in_warp < 4) {
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i) {
                int col_offset = (i & 1) + ((i >> 1) * 8);
                plan.sOScale[my_col_base + col_offset] = o_scale[i];
            }

            if (args.is_no_split) {
                // No-split: LSE in natural-log domain to params.lse.
                // lse = +INF when rL == 0 (no valid token attended).
                float* lse_base = (float*)params.lse
                                + batch_idx * params.stride_lse_b
                                + s_q_idx * params.stride_lse_s_q;
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    int col_offset = (i & 1) + ((i >> 1) * 8);
                    float lse_val = (rL[i] == 0.0f)
                                  ? +INFINITY
                                  : logf(rL[i]) + rM[i] / (float)M_LOG2E;
                    lse_base[my_col_base + col_offset] = lse_val;
                }
            } else {
                // Split: LSE in log2 domain to lse_accum at split_idx.
                int n_split_idx = (batch_idx == sched_meta.begin_req_idx)
                                ? sched_meta.begin_split_idx : 0;
                int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
                float* lse_base = params.lse_accum
                                + split_idx * params.stride_lse_accum_split
                                + s_q_idx * params.stride_lse_accum_s_q;
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    int col_offset = (i & 1) + ((i >> 1) * 8);
                    float lse_val = (rL[i] == 0.0f)
                                  ? -INFINITY
                                  : log2f(rL[i]) + rM[i];
                    lse_base[my_col_base + col_offset] = lse_val;
                }
            }
        }

        // 9.4: NamedBarrier sync -- sK union slot now releasable; sOScale visible.
        cutlass::arch::NamedBarrier(
            256, NamedBarriers::o_buf_free_and_sL_ready).arrive_and_wait();

        // 9.5: scatter rO * o_scale into staging buffer at D_V[0:256] half.
        // Per-thread 32 stores = 4 M-tiles x 2 d_v_row x 4 head cols.
        // d_v_row: m * 64 + warp_in_wg * 16 + lane/4 + 8 * em1
        // head_col: my_col_base + col_offset(i)
        // ci = em0 + em2 * 2 (matches Step 4.5 rescale loop).
        if (args.is_no_split) {
            Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()),
                                       SmemLayoutOBuf{});
            CUTE_UNROLL
            for (int m = 0; m < 4; ++m) {
                int d_v_atom_base = m * 64 + warp_in_wg * 16
                                  + (lane_in_warp / 4);
                CUTE_UNROLL
                for (int em1 = 0; em1 < 2; ++em1) {
                    int d_v_local = d_v_atom_base + 8 * em1;  // [0, 256)
                    CUTE_UNROLL
                    for (int em2 = 0; em2 < 2; ++em2) {
                        CUTE_UNROLL
                        for (int em0 = 0; em0 < 2; ++em0) {
                            int ci = em0 + em2 * 2;
                            int col_offset = em0 + em2 * 8;
                            int head = my_col_base + col_offset;
                            float val = rO(make_coord(em0, em1, em2), m, _0{})
                                      * o_scale[ci];
                            sOBuf(head, d_v_local) = bf16(val);
                        }
                    }
                }
            }
        } else {
            Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()),
                                            SmemLayoutOAccumBuf{});
            CUTE_UNROLL
            for (int m = 0; m < 4; ++m) {
                int d_v_atom_base = m * 64 + warp_in_wg * 16
                                  + (lane_in_warp / 4);
                CUTE_UNROLL
                for (int em1 = 0; em1 < 2; ++em1) {
                    int d_v_local = d_v_atom_base + 8 * em1;  // [0, 256)
                    CUTE_UNROLL
                    for (int em2 = 0; em2 < 2; ++em2) {
                        CUTE_UNROLL
                        for (int em0 = 0; em0 < 2; ++em0) {
                            int ci = em0 + em2 * 2;
                            int col_offset = em0 + em2 * 8;
                            int head = my_col_base + col_offset;
                            float val = rO(make_coord(em0, em1, em2), m, _0{})
                                      * o_scale[ci];
                            sOAccumBuf(head, d_v_local) = val;
                        }
                    }
                }
            }
        }

        // 9.6: make generic stores visible to async proxy.
        fence_view_async_shared();

        // 9.7: NamedBarrier -- wait WG1 finishes its D_V[256:512] scatter.
        cutlass::arch::NamedBarrier(
            256, NamedBarriers::epilogue_r2s_ready).arrive_and_wait();

        // 9.8: lane-elected bulk_copy_s2g per row. WG0 covers rows 0..7
        // (warp w writes rows 2w + rj for rj in {0, 1}).
        if (cute::elect_one_sync()) {
            CUTE_UNROLL
            for (int rj = 0; rj < 2; ++rj) {
                int row = warp_in_wg * 2 + rj;   // 0..7
                if (row < (int)NUM_HEADS) {
                    if (args.is_no_split) {
                        bf16* gO_row = (bf16*)params.out
                                     + batch_idx * params.stride_o_b
                                     + s_q_idx * params.stride_o_s_q
                                     + row * params.stride_o_h_q;
                        Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()),
                                                   SmemLayoutOBuf{});
                        SM90_BULK_COPY_S2G::copy(
                            &sOBuf(row, _0{}),
                            gO_row,
                            HEAD_DIM_V * (int)sizeof(bf16));
                    } else {
                        int n_split_idx = (batch_idx == sched_meta.begin_req_idx)
                                        ? sched_meta.begin_split_idx : 0;
                        int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
                        float* gO_row = params.o_accum
                                      + split_idx * params.stride_o_accum_split
                                      + s_q_idx * params.stride_o_accum_s_q
                                      + row * params.stride_o_accum_h_q;
                        Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()),
                                                        SmemLayoutOAccumBuf{});
                        SM90_BULK_COPY_S2G::copy(
                            &sOAccumBuf(row, _0{}),
                            gO_row,
                            HEAD_DIM_V * (int)sizeof(float));
                    }
                }
            }
            cute::tma_store_arrive();
        }
        cute::tma_store_wait<0>();

        // Sync all 3 WGs before starting next batch: ensures sK union slot
        // (consumed by WG2's K writes from next batch's first K-block) and
        // sOBuf/sOAccumBuf (overwritten by next batch's epilogue) are released.
        __syncthreads();
        }
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<160>();

        // ===============================================================
        // Per-CTA RF state setup (persistent across batches)
        // ===============================================================
        TiledMMA_PV tiled_mma_PV;
        auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

        // rO: BMM2 accumulator for D_V[256:512] x h_q=16. Same shape per WG
        // as WG0's rO (32 fp32 = 4 M-tiles x 2 D_V rows x 4 head cols).
        // Reset to 0 per batch inside the batch loop below.
        Tensor rO = partition_fragment_C(
            tiled_mma_PV,
            Shape<Int<D_V_SPLIT>, Int<NUM_HEADS>>{});

        // WG1 thread layout (same lane mapping convention as WG0).
        const int warp_in_wg   = idx_in_warpgroup / 32;
        const int lane_in_warp = idx_in_warpgroup % 32;
        const int my_col_base  = (lane_in_warp % 4) * 2;
        (void)warp_in_wg;  // suppressed; Step 9 uses it

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx;
             batch_idx <= sched_meta.end_req_idx;
             ++batch_idx) {

        MainloopArgs args = get_cur_req_info(batch_idx);

        // Reset per-batch RF state.
        clear(rO);

        // ===============================================================
        // Main K-block loop (per batch)
        // ===============================================================
        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
            const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;

            // -----------------------------------------------------------
            // Step 5: wait for sScale + sS_T from WG0, then rescale rO.
            //
            // arrive_and_wait contributes 128 (WG1) + 128 (WG0 in Step 6
            // of same K-block) = 256 -> barrier flips, both unblock.
            // After this returns, sScale and sS_T are both visible.
            // -----------------------------------------------------------
            cutlass::arch::NamedBarrier(
                256, NamedBarriers::softmax_to_wg1_ready).arrive_and_wait();

            // Read this thread's 4 head cols' scale_for_old (col offsets
            // {0,1,8,9} from my_col_base).
            float scale_for_old[4];
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i) {
                int col_offset = (i & 1) + ((i >> 1) * 8);
                scale_for_old[i] = plan.sScale[my_col_base + col_offset];
            }

            // Rescale rO (32 fp32 = 4 M-tiles x 2 row_pair x 4 cols).
            // Per-col index ci = em0 + em2 * 2 from the wgmma accumulator coord.
            CUTE_UNROLL
            for (int m = 0; m < 4; ++m) {
                CUTE_UNROLL
                for (int r = 0; r < 2; ++r) {
                    CUTE_UNROLL
                    for (int em2 = 0; em2 < 2; ++em2) {
                        CUTE_UNROLL
                        for (int em0 = 0; em0 < 2; ++em0) {
                            int ci = em0 + em2 * 2;
                            rO(make_coord(em0, r, em2), m, _0{}) *= scale_for_old[ci];
                        }
                    }
                }
            }

            // -----------------------------------------------------------
            // Step 7 + 8: V re-view (D_V[256:512] offset) + BMM2 V @ S^T -> rO
            //
            // sV_view base = sK[buf] + offset to skip first 256 D_V cols.
            // Offset is computed via the full SmemLayoutV at logical (256, 0)
            // -- same pattern as sparse_fp8 / TP=8.
            //
            // Atom MMA_64x16x16 SS<MN, K>: 4 M-tiles x 4 K-iters = 16 wgmma.
            // accumulate (no zero_init); rO already pre-rescaled by Step 5.
            // -----------------------------------------------------------
            Tensor sV_view = make_tensor(
                make_smem_ptr(plan.u.k[buf_idx].data()
                              + (SmemLayoutV{})(_256{}, _0{})),
                SmemLayoutHalfV{});
            Tensor sS_T = make_tensor(make_smem_ptr(plan.s_t.data()),
                                      SmemLayoutS_T{});

            gemm</*zero_init=*/false, /*wg_wait=*/-1>(
                tiled_mma_PV,
                thr_mma_PV.partition_fragment_A(sV_view),
                thr_mma_PV.partition_fragment_B(sS_T),
                rO);
            cute::warpgroup_wait<0>();

            // Release sK[buf_idx] slot for producer (128 of 256).
            plan.bar_k_avail[buf_idx].arrive();

            // Signal sS_T can be overwritten by next iter's WG0 Step 6.
            // Done HERE (after BMM2) since BMM2 is the consumer of sS_T.
            // Skip on the LAST K-block of this batch: WG0 also won't wait next round.
            if (block_idx != args.end_block_idx - 1) {
                cutlass::arch::NamedBarrier(
                    256, NamedBarriers::softmax_buf_free).arrive();
            }
        }

        // ===============================================================
        // Step 9 epilogue (WG1): D_V[256:512] half
        //
        // WG1 doesn't compute o_scale itself -- reads sOScale[head] from
        // SMEM (broadcast by WG0). Doesn't write LSE either (WG0 owns).
        // Same split / no_split branching as WG0; differs only in d_v range.
        // ===============================================================

        // 9.4: NamedBarrier sync -- wait WG0 to write sOScale and finalize
        // sK union slot release.  WG1's arrive contributes 128 of count=256.
        cutlass::arch::NamedBarrier(
            256, NamedBarriers::o_buf_free_and_sL_ready).arrive_and_wait();

        // Read this thread's 4 head cols' final scale (1/rL).
        float o_scale[4];
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) {
            int col_offset = (i & 1) + ((i >> 1) * 8);
            o_scale[i] = plan.sOScale[my_col_base + col_offset];
        }

        // 9.5: scatter rO * o_scale into staging buffer at D_V[256:512] half.
        if (args.is_no_split) {
            Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()),
                                       SmemLayoutOBuf{});
            CUTE_UNROLL
            for (int m = 0; m < 4; ++m) {
                int d_v_atom_base = m * 64 + warp_in_wg * 16
                                  + (lane_in_warp / 4);
                CUTE_UNROLL
                for (int em1 = 0; em1 < 2; ++em1) {
                    int d_v_global = D_V_SPLIT + d_v_atom_base + 8 * em1;  // [256, 512)
                    CUTE_UNROLL
                    for (int em2 = 0; em2 < 2; ++em2) {
                        CUTE_UNROLL
                        for (int em0 = 0; em0 < 2; ++em0) {
                            int ci = em0 + em2 * 2;
                            int col_offset = em0 + em2 * 8;
                            int head = my_col_base + col_offset;
                            float val = rO(make_coord(em0, em1, em2), m, _0{})
                                      * o_scale[ci];
                            sOBuf(head, d_v_global) = bf16(val);
                        }
                    }
                }
            }
        } else {
            Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()),
                                            SmemLayoutOAccumBuf{});
            CUTE_UNROLL
            for (int m = 0; m < 4; ++m) {
                int d_v_atom_base = m * 64 + warp_in_wg * 16
                                  + (lane_in_warp / 4);
                CUTE_UNROLL
                for (int em1 = 0; em1 < 2; ++em1) {
                    int d_v_global = D_V_SPLIT + d_v_atom_base + 8 * em1;  // [256, 512)
                    CUTE_UNROLL
                    for (int em2 = 0; em2 < 2; ++em2) {
                        CUTE_UNROLL
                        for (int em0 = 0; em0 < 2; ++em0) {
                            int ci = em0 + em2 * 2;
                            int col_offset = em0 + em2 * 8;
                            int head = my_col_base + col_offset;
                            float val = rO(make_coord(em0, em1, em2), m, _0{})
                                      * o_scale[ci];
                            sOAccumBuf(head, d_v_global) = val;
                        }
                    }
                }
            }
        }

        fence_view_async_shared();

        // 9.7: NamedBarrier -- both halves populated.
        cutlass::arch::NamedBarrier(
            256, NamedBarriers::epilogue_r2s_ready).arrive_and_wait();

        // 9.8: lane-elected bulk_copy_s2g per row. WG1 covers rows 8..15
        // (warp w writes rows 8 + 2w + rj for rj in {0, 1}).
        if (cute::elect_one_sync()) {
            CUTE_UNROLL
            for (int rj = 0; rj < 2; ++rj) {
                int row = 8 + warp_in_wg * 2 + rj;   // 8..15
                if (row < (int)NUM_HEADS) {
                    if (args.is_no_split) {
                        bf16* gO_row = (bf16*)params.out
                                     + batch_idx * params.stride_o_b
                                     + s_q_idx * params.stride_o_s_q
                                     + row * params.stride_o_h_q;
                        Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()),
                                                   SmemLayoutOBuf{});
                        SM90_BULK_COPY_S2G::copy(
                            &sOBuf(row, _0{}),
                            gO_row,
                            HEAD_DIM_V * (int)sizeof(bf16));
                    } else {
                        int n_split_idx = (batch_idx == sched_meta.begin_req_idx)
                                        ? sched_meta.begin_split_idx : 0;
                        int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
                        float* gO_row = params.o_accum
                                      + split_idx * params.stride_o_accum_split
                                      + s_q_idx * params.stride_o_accum_s_q
                                      + row * params.stride_o_accum_h_q;
                        Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()),
                                                        SmemLayoutOAccumBuf{});
                        SM90_BULK_COPY_S2G::copy(
                            &sOAccumBuf(row, _0{}),
                            gO_row,
                            HEAD_DIM_V * (int)sizeof(float));
                    }
                }
            }
            cute::tma_store_arrive();
        }
        cute::tma_store_wait<0>();

        // Sync all 3 WGs before starting next batch.
        __syncthreads();
        }   // end per-batch loop (WG1)
    } else {  // warpgroup_idx == 2 (producer)
        // ===================================================================
        // WG2 (producer): per K-block, fp8 K gather + dequant -> write sK[buf]
        //
        // Step 2 implementation. Identical to TP=8 swapsab producer (the path
        // is h_q-independent: it only reads the K cache, does not touch Q).
        // Adapted from sparse_fp8 with cluster=2 multicast paths and
        // MODEL1/extra_kv branches removed (V32-only). Wrapped in a per-batch
        // persistent loop matching WG0/WG1.
        // ===================================================================
        cutlass::arch::warpgroup_reg_dealloc<152>();

        // Producer thread layout:
        //   - 4 warps x 32 lanes = 128 threads in WG2
        //   - my_token_idx_base in [0, 32) selects 32 distinct token positions
        //     (warp_in_wg in [0,4) x lane%8 in [0,8))
        //   - NUM_TOKENS_PER_THREAD=2 rounds x 32 positions = 64 = B_TOPK
        //   - The 4 threads sharing one (warp_in_wg, lane%8) split each token's
        //     512 fp8 NoPE across lane/8 in [0,4): each does 128 fp8 of one token.
        static constexpr int NUM_TOKENS_PER_THREAD = 2;
        static constexpr int NUM_TOKENS_PER_ROUND  = 32;

        const int warp_in_wg = idx_in_warpgroup / 32;          // [0, 4)
        const int lane_idx   = idx_in_warpgroup % 32;          // [0, 32)
        const int my_token_idx_base = warp_in_wg * 8 + lane_idx % 8;  // [0, 32)

        // V32 KV cache pointers (only path supported).
        const int     page_block_size = params.page_block_size;
        const int64_t k_block_stride  = params.stride_kv_block;
        const int64_t k_row_stride    = params.stride_kv_row;
        fp8* const    k_ptr           = (fp8*)params.kv;

        // ===============================================================
        // Per-batch persistent loop (mirrors WG0 / WG1).
        // ===============================================================
        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx;
             batch_idx <= sched_meta.end_req_idx;
             ++batch_idx) {

        MainloopArgs args = get_cur_req_info(batch_idx);

        // Per-batch indices base. s_q=1 -> no s_q stride term.
        int* gIndices = params.indices + batch_idx * params.stride_indices_b
                                       + s_q_idx * params.stride_indices_s_q;

        // Prefetch token indices for the first K-block this batch covers.
        int nxt_token_indices[NUM_TOKENS_PER_THREAD];
        CUTE_UNROLL
        for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
            nxt_token_indices[round] = __ldg(
                gIndices + args.start_block_idx * TOPK_BLOCK_SIZE
                         + round * NUM_TOKENS_PER_ROUND
                         + my_token_idx_base);
        }

        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
            const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;

            CUTE_UNROLL
            for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                const int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;

                // Wait once per K-block for the buffer to be free.
                // Init phase = 1; first wait phase = (0>>buf&1)^1 = 1 -> passes immediately.
                if (round == 0) {
                    plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
                }

                // SMEM dst base for this thread's (token, dim-slice) write.
                bf16* sK_nope_base = plan.u.k[buf_idx].data()
                                   + my_token_idx * 8
                                   + (lane_idx / 8) * 16 * TOPK_BLOCK_SIZE;

                // Use prefetched token index for this round; prefetch next K-block's
                // (within this batch only -- next batch will re-prefetch).
                int token_index = nxt_token_indices[round];
                if (block_idx + 1 < args.end_block_idx) {
                    nxt_token_indices[round] = __ldg(
                        gIndices + (block_idx + 1) * TOPK_BLOCK_SIZE
                                 + round * NUM_TOKENS_PER_ROUND
                                 + my_token_idx_base);
                }

                // Paged-cache address.
                int block_index = (token_index == -1)
                    ? 0
                    : (int)((uint32_t)token_index / (uint32_t)page_block_size);
                int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;
                fp8* gK_base = k_ptr + block_index * k_block_stride
                                     + rel_idx_in_block * k_row_stride;

                // V32: 4 float scales (one per 128 NoPE dims) follow NoPE in gmem.
                float scales_float[NUM_SCALES];
                *(float4*)(scales_float) = load_128b_from_gmem<
                        float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(
                    (float*)(gK_base + HEAD_DIM_NOPE));
                bf16 scales[NUM_SCALES];
                CUTE_UNROLL
                for (int i = 0; i < NUM_SCALES; ++i) scales[i] = (bf16)scales_float[i];
                if (token_index == -1) {
                    CUTE_UNROLL
                    for (int i = 0; i < NUM_SCALES; ++i) scales[i] = (bf16)0.0f;
                }

                // ===== NoPE: load 512 fp8 / token, dequant, write bf16 to sK =====
                fp8* gK_nope = gK_base + (lane_idx / 8) * 16;
                CUTE_UNROLL
                for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE / 64; ++dim_idx) {
                    fp8x16 cur_fp8x16 = load_128b_from_gmem<
                            fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(
                        gK_nope + dim_idx * 64);

                    if (token_index == -1) {
                        *(uint128_t*)(&cur_fp8x16) = uint128_t();
                    }
                    bf16 scale = scales[dim_idx / 2];   // V32: 1 scale per 2 dim tiles (128 dims)
                    auto dequant_and_save = [&](const fp8x8 &data, int offset) {
                        int smem_offset = (dim_idx * 64 + offset) * TOPK_BLOCK_SIZE;
                        bf16x8 cur_bf16x8 = cvt_fp8x8_bf16x8(
                            data,
                            __bfloat162bfloat162(*(__nv_bfloat16*)(&scale)));
                        *(__int128_t*)(sK_nope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                    };
                    dequant_and_save(cur_fp8x16.lo, 0);
                    dequant_and_save(cur_fp8x16.hi, 8);
                }

                // ===== RoPE: load 64 bf16 / token (already bf16; no dequant) =====
                bf16* gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE + NUM_SCALES * sizeof(float))
                              + (lane_idx / 8) * 8;
                bf16* sK_rope_base = plan.u.k[buf_idx].data()
                                   + my_token_idx * 8
                                   + (lane_idx / 8) * 8 * TOPK_BLOCK_SIZE;
                CUTE_UNROLL
                for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE / 32; ++dim_idx) {
                    bf16x8 cur_bf16x8 = load_128b_from_gmem<
                            bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(
                        gK_rope + dim_idx * 32);
                    if (token_index == -1)
                        *(uint128_t*)(&cur_bf16x8) = uint128_t();
                    int smem_offset = (HEAD_DIM_NOPE + dim_idx * 32) * TOPK_BLOCK_SIZE;
                    *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                }
            }   // for round

            fence_view_async_shared();

            // OOB validity flags: warp 0 (32 lanes) writes 2 entries each = 64 = B_TOPK.
            if (idx_in_warpgroup < 32) {
                int2 indices_pair = __ldg(
                    (int2*)(gIndices + block_idx * TOPK_BLOCK_SIZE + lane_idx * 2));
                *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx * 2]) = {
                    (char)(indices_pair.x != -1),
                    (char)(indices_pair.y != -1)
                };
            }

            // Signal sK[buf_idx] is ready. All 128 threads in WG2 arrive once.
            plan.bar_k_local_ready[buf_idx].arrive();
            bar_phase_k ^= 1 << buf_idx;
        }   // end K-block loop

        // Sync all 3 WGs before starting next batch.
        __syncthreads();
        }   // end per-batch loop (WG2)
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
}

// ===========================================================================
// __global__ kernel wrapper
// ===========================================================================
template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
flash_fwd_splitkv_mla_fp8_swapsab_tp4_kernel(
    __grid_constant__ const SparseAttnDecodeParams params,
    __grid_constant__ const TMAParams              tma_params
) {
    Kernel::devfunc(params, tma_params);
}

// ===========================================================================
// run() -- host entry point
// ===========================================================================
template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    // ----- Production scope asserts (see config.h header comment for rationale) -----
    KU_ASSERT(params.h_q == NUM_HEADS);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.s_q == 1, "swapsab_tp4 only supports s_q=1");
    KU_ASSERT(params.extra_kv == nullptr, "swapsab_tp4 does not support extra_kv");
    KU_ASSERT(params.attn_sink == nullptr, "swapsab_tp4 does not support attn_sink");
    KU_ASSERT(params.topk_length == nullptr,
              "swapsab_tp4 does not support per-batch topk_length");
    KU_ASSERT(params.stride_kv_row == 656,
              "expected V32 KV cache stride_kv_row = 656 bytes per token "
              "(512 fp8 NoPE + 4 float scales + 64 bf16 RoPE)");

    // ----- Build TMA descriptor for Q -----
    // Q gmem shape: (h_q=16, D_QK=576, s_q, b). Persistent batching reads
    // Q[:,:,s_q_idx, batch_idx] per batch.
    auto shape_Q = make_shape((int)params.h_q, (int)HEAD_DIM_K,
                              (int)params.s_q, (int)params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(shape_Q,
                        make_stride(params.stride_q_h_q, _1{},
                                    params.stride_q_s_q, params.stride_q_b))
        ),
        SmemLayoutQ{}
    );

    TmaParams<decltype(shape_Q), decltype(tma_Q)> tma_params = {shape_Q, tma_Q};

    // ----- Launch -----
    auto kernel = &flash_fwd_splitkv_mla_fp8_swapsab_tp4_kernel<
        KernelTemplate<MODEL_TYPE, NUM_HEADS>, decltype(tma_params)
    >;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Grid (NUM_M_BLOCKS=1, s_q, num_sm_parts) matches sparse_fp8 / TP=8.
    cutlass::ClusterLaunchParams launch_params = {
        dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size,
        params.stream
    };
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

// ===========================================================================
// Top-level entry exposed via splitkv_mla.h
// ===========================================================================
template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_swapsab_tp4_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}  // namespace sm90::decode::sparse_fp8_swapsab_tp4
