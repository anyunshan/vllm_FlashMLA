#pragma once

#include "params.h"

namespace sm90::decode::sparse_native_fp8_swapsab_tp2 {

// SM90 sparse decode, native FP8 x swap-AB TP=2 (h_q=32) kernel launcher.
// Fusion of sparse_native_fp8 (fp8 primitives) + sparse_fp8_swapsab_tp2
// (N=32 consumer dataflow). See docs/native_fp8_swapsab_tp2_decode_plan.md.
//
// Scope: h_q=32 / cluster=1 / V32 / 576B token / no attn_sink / no extra_kv /
// no topk_length. NUM_HEADS is templated for symmetry with the sibling kernels
// but only 32 is instantiated (static_assert in config.h).
template<int NUM_HEADS>
void run_flash_splitkv_mla_native_fp8_swapsab_tp2_kernel(const SparseAttnDecodeParams &params);

}  // namespace sm90::decode::sparse_native_fp8_swapsab_tp2
