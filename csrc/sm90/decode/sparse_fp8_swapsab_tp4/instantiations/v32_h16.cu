#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_fp8_swapsab_tp4 {

template void run_flash_splitkv_mla_fp8_swapsab_tp4_kernel<ModelType::V32, 16>(
    const SparseAttnDecodeParams &params);

}
