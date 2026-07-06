#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_fp8_swapsab_tp8 {

template void run_flash_splitkv_mla_fp8_swapsab_tp8_kernel<ModelType::V32, 8>(
    const SparseAttnDecodeParams &params);

}
