#pragma once

#include "params.h"

namespace sm90::decode::sparse_fp8_swapsab_tp8 {

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_swapsab_tp8_kernel(const SparseAttnDecodeParams &params);

}
