// Phase 1 instantiation: NUM_HEADS=32 (TP=2) / cluster=1 / V32.
// This compiles the full devfunc + run() (splitkv_mla.cuh). During the M3
// step-by-step fusion, WG0/WG1 are still skeletons; this file is the compile
// gate that the producer + prologue + launcher build clean.

#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_native_fp8_swapsab_tp2 {

template void run_flash_splitkv_mla_native_fp8_swapsab_tp2_kernel<32>(const SparseAttnDecodeParams &params);

}  // namespace sm90::decode::sparse_native_fp8_swapsab_tp2
