#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "sparse_native_fp8_swapsab_tp2_fwd.h"
#include "dense_decode.h"
#include "dense_fwd.h"

// Namespace renamed from _flashmla_C to _flashmla_C_novita so this wheel can
// coexist in one process with vLLM's own vllm._flashmla_C extension (which
// keeps serving the BF16 prefill path).
STABLE_TORCH_LIBRARY(_flashmla_C_novita, m) {
    m.def("sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, Tensor(a)? tile_scheduler_metadata, Tensor(b)? num_splits, Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, int d_v, float sm_scale, Tensor(c!)? out_) -> (Tensor(c!), Tensor, Tensor(a)?, Tensor(b)?)");
    m.def("dense_decode_fwd(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float softmax_scale, bool is_causal, Tensor(a)? tile_scheduler_metadata, Tensor(b)? num_splits, Tensor(c!)? out_) -> (Tensor(c!), Tensor, Tensor(a)?, Tensor(b)?)");
    m.def("sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? attn_sink, Tensor? topk_length, Tensor(a!)? out_) -> Tensor[]");
    m.def("dense_prefill_fwd(Tensor workspace_buffer, Tensor q, Tensor k, Tensor v, Tensor cumulative_seqlen_q, Tensor cumulative_seqlen_kv, Tensor(a!) o, Tensor(b!) lse, int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) -> ()");
    // Native-FP8 swap-AB TP2 sparse decode (SM90, h_q=32, s_q=1, 576B/token
    // all-fp8 cache, no scales). Empty tile_scheduler_metadata/num_splits on
    // the first call trigger generation; the returned pair is reused across
    // same-shape calls.
    m.def("sparse_native_fp8_swapsab_tp2_decode_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, Tensor tile_scheduler_metadata, Tensor num_splits) -> (Tensor, Tensor, Tensor, Tensor)");
#ifdef FLASH_MLA_ENABLE_DENSE_BWD
    // Dense prefill backward is only registered when its kernel is compiled
    // (standalone setup.py). vLLM's integrated build is inference-only and does
    // not compile fmha_cutlass_bwd_sm100.cu, matching the original 4-op library.
    m.def("dense_prefill_bwd(Tensor(a!) workspace_buffer, Tensor d_o, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, Tensor cumulative_seqlen_q, Tensor cumulative_seqlen_kv, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) -> ()");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_flashmla_C_novita, CUDA, m) {
    m.impl("sparse_decode_fwd", TORCH_BOX(&sparse_attn_decode_interface));
    m.impl("dense_decode_fwd", TORCH_BOX(&dense_attn_decode_interface));
    m.impl("sparse_prefill_fwd", TORCH_BOX(&sparse_attn_prefill_interface));
    m.impl("dense_prefill_fwd", TORCH_BOX(&FMHACutlassSM100FwdRun));
    m.impl("sparse_native_fp8_swapsab_tp2_decode_fwd", TORCH_BOX(&sparse_native_fp8_swapsab_tp2_decode_fwd));
#ifdef FLASH_MLA_ENABLE_DENSE_BWD
    m.impl("dense_prefill_bwd", TORCH_BOX(&FMHACutlassSM100BwdRun));
#endif
}

// To enable importing flash_mla._flashmla_C_novita as a python module
PyMODINIT_FUNC PyInit__flashmla_C_novita() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_flashmla_C_novita", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
