#pragma once

// Launcher for sm90 native FP8 x swap-AB TP=2 (h_q=32) sparse decode.
//
// Grafted from FlashMLA_FP8_claude (feat/sm90-native-fp8-swapsab-tp2-decode)
// and rewritten against the libtorch-stable ABI (see sparse_decode.h for the
// reference style). Independent entry point -- does not touch the main
// sparse_decode_fwd dispatch.
//
// Scope: h_q=32 / s_q=1 / cluster=1 / V32 native layout (576B/token all-fp8,
// no scales) / no attn_sink / no extra_kv / no topk_length.

#include "common.h"

#include "params.h"

#include <cutlass/numeric_types.h>

#include "sm90/decode/sparse_native_fp8_swapsab_tp2/splitkv_mla.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "smxx/decode/combine/combine.h"

// Returns (out_bf16, lse, tile_scheduler_metadata, num_splits).
//   q:        [b, s_q, h_q=32, d_qk=576] FP8 e4m3
//   kv:       [num_blocks, page_block_size, h_kv=1, 576 bytes/token] FP8 e4m3
//   indices:  [b, s_q, topk] int32, absolute token index, -1 = invalid
//   sm_scale: softmax scale (= 1/sqrt(d_qk))
// sched-meta reuse: pass empty (numel()==0) int32 tensors on the first call to
// trigger generation; carry the returned pair across same-shape invocations.
static std::tuple<Tensor, Tensor, Tensor, Tensor>
sparse_native_fp8_swapsab_tp2_decode_fwd(
    const Tensor &q,
    const Tensor &kv,
    const Tensor &indices,
    double sm_scale,
    Tensor tile_scheduler_metadata,
    Tensor num_splits
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    STD_TORCH_CHECK(arch.is_sm90a(), "native_fp8_swapsab_tp2 requires SM90a");

    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);

    int b = q.size(0);
    int s_q = q.size(1);
    int h_q = q.size(2);
    int d_qk = q.size(3);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int topk = indices.size(2);

    constexpr int TP2_NUM_HEADS = 32;
    constexpr int FP8_R5_BYTES_PER_TOKEN = 576;
    constexpr int D_V = 512;

    STD_TORCH_CHECK(h_q == TP2_NUM_HEADS,
                    "native_fp8_swapsab_tp2 supports h_q=32 only (got h_q=", h_q, ")");
    STD_TORCH_CHECK(h_kv == 1, "MQA only (got h_kv=", h_kv, ")");
    STD_TORCH_CHECK(d_qk == 576, "V32 only: d_qk=576 (got ", d_qk, ")");
    STD_TORCH_CHECK(topk > 0 && topk % 64 == 0, "topk must be a positive multiple of 64");

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DTYPE(q, ScalarType::Float8_e4m3fn);
    STD_TORCH_CHECK(kv.scalar_type() == ScalarType::Float8_e4m3fn
                        || kv.scalar_type() == ScalarType::Char
                        || kv.scalar_type() == ScalarType::Byte,
                    "kv dtype must be fp8_e4m3fn / int8 / uint8");
    KU_CHECK_DTYPE(indices, ScalarType::Int);
    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, FP8_R5_BYTES_PER_TOKEN);
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    STD_TORCH_CHECK(kv.stride(1) == FP8_R5_BYTES_PER_TOKEN,
                    "kv block must be contiguous 576 bytes/token");
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);

    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());

    Tensor out = torch::stable::new_empty(q, {b, s_q, h_q, D_V}, ScalarType::BFloat16);
    Tensor lse = torch::stable::new_empty(q, {b, s_q, h_q}, ScalarType::Float);

    // SparseAttnDecodeParams: pointer fields are bf16* placeholders; the kernel
    // reinterprets q/kv as fp8* at runtime (native convention).
    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, D_V,
        (float)sm_scale, (float)(sm_scale * LOG_2_E),
        num_blocks, page_block_size, topk,
        ModelType::V32,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        nullptr,                    // topk_length
        nullptr,                    // attn_sink
        (float*)lse.data_ptr(),
        (bf16*)out.data_ptr(),

        0, 0, 0,                    // extra_*
        nullptr, nullptr, nullptr,

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),

        0, 0, 0, 0,                 // extra strides

        get_current_cuda_stream(q)
    };

    // Tile scheduling metadata (reuse caller tensors when non-empty).
    constexpr int FIXED_OVERHEAD_NUM_BLOCKS = 5;
    constexpr int BLOCK_SIZE_TOPK = 64;
    int num_sm_parts = std::max(arch.num_sms / s_q / (h_q / 64 > 0 ? h_q / 64 : 1), 1);

    if (tile_scheduler_metadata.numel() == 0) {
        tile_scheduler_metadata = torch::stable::new_empty(
            q, {num_sm_parts, (int)(sizeof(DecodingSchedMeta) / sizeof(int))}, ScalarType::Int);
        num_splits = torch::stable::new_empty(q, {b + 1}, ScalarType::Int);

        GetDecodeSchedMetaParams sched_params = {
            b, s_q, BLOCK_SIZE_TOPK, FIXED_OVERHEAD_NUM_BLOCKS,
            topk, 0, nullptr, nullptr, nullptr,
            (DecodingSchedMeta*)tile_scheduler_metadata.data_ptr(),
            num_splits.mutable_data_ptr<int>(), num_sm_parts,
            get_current_cuda_stream(q)
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(sched_params);
    }
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DTYPE(tile_scheduler_metadata, ScalarType::Int);
    KU_CHECK_DTYPE(num_splits, ScalarType::Int);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_SHAPE(tile_scheduler_metadata, num_sm_parts, sizeof(DecodingSchedMeta) / sizeof(int));
    KU_CHECK_SHAPE(num_splits, b + 1);
    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata.data_ptr();
    params.num_splits_ptr = num_splits.mutable_data_ptr<int>();
    params.num_sm_parts = num_sm_parts;

    // Split-K intermediate buffers.
    const int total_num_splits = b + num_sm_parts;
    Tensor lse_accum = torch::stable::new_empty(q, {total_num_splits, s_q, h_q}, ScalarType::Float);
    Tensor o_accum = torch::stable::new_empty(q, {total_num_splits, s_q, h_q, D_V}, ScalarType::Float);
    KU_CHECK_CONTIGUOUS(lse_accum);
    KU_CHECK_CONTIGUOUS(o_accum);
    params.lse_accum = lse_accum.mutable_data_ptr<float>();
    params.o_accum = o_accum.mutable_data_ptr<float>();
    params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
    params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
    params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
    params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
    params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

    // Launch the fused native-fp8 x swap-AB TP2 kernel.
    sm90::decode::sparse_native_fp8_swapsab_tp2::run_flash_splitkv_mla_native_fp8_swapsab_tp2_kernel<TP2_NUM_HEADS>(params);

    // Combine kernel<bf16>: merge multi-split o_accum -> out (bf16) + lse_accum
    // (log2) -> lse (natural log). is_no_split slots are pass-through.
    CombineParams combine_params = {
        b, s_q, h_q, D_V,

        params.lse,
        params.out,
        params.stride_lse_b, params.stride_lse_s_q,
        params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,

        params.lse_accum,
        params.o_accum,
        params.stride_lse_accum_split, params.stride_lse_accum_s_q,
        params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,

        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.num_sm_parts,

        nullptr,                    // attn_sink
        get_current_cuda_stream(q)
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

    return {out, lse, tile_scheduler_metadata, num_splits};
}
