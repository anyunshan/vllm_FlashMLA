#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_fp8_swapsab_tp2 {

template void run_flash_splitkv_mla_fp8_swapsab_tp2_kernel<ModelType::V32, 32>(
    const SparseAttnDecodeParams &params);

}
