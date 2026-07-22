#pragma once

#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include "defines.h"

using namespace cute;

namespace sm90::decode::sparse_native_fp8 {

// FP8 R5 KV cache byte layout (V32 only):
//   each token = 576 bytes = 512 NoPE fp8 + 64 RoPE fp8 (no scales segment)
//   vs original V32 sparse_fp8 = 512 NoPE fp8 + 16 fp32 scales + 128 RoPE bf16 = 656
// HEAD_DIM_K/V/NOPE/ROPE are centralized in main sparse_native_fp8/config.h KernelTemplate
// class scope (plan section 4 Part 1), not redeclared here, to avoid shadowing the class
// scope static constexpr definitions.
static constexpr int NUM_BYTES_PER_TOKEN = 576;
static constexpr int PAGE_BLOCK_SIZE     = 64;

}
