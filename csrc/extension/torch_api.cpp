#include <Python.h>

#include <optional>
#include <vector>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

extern std::vector<Tensor>
fwd_kvcache_mla_fp8(
    Tensor q,                                    // batch_size x seqlen_q x num_heads x head_size
    const Tensor &kcache,                        // num_blocks x page_block_size x num_heads_k x head_size (when is_fp8 is False) or num_blocks x num_heads_k x (page_block_size*656) (when is_fp8 is True)
    const int64_t head_size_v,
    const Tensor &seqlens_k,                     // batch_size
    const Tensor &block_table,                   // batch_size x max_num_blocks_per_seq
    const double softmax_scale,
    bool is_causal,
    const Tensor &tile_scheduler_metadata,       // num_sm_parts x TileSchedulerMetaDataSize
    const Tensor &num_splits,                    // batch_size + 1
    const std::optional<Tensor> &descale_q,      // None or batch_size
    const std::optional<Tensor> &descale_k       // None or batch_size
);

extern std::vector<Tensor>
get_mla_decoding_metadata_dense_fp8(
    const Tensor &seqlens_k,
    const int64_t num_heads_per_head_k,
    const int64_t num_heads_k
);

STABLE_TORCH_LIBRARY(_flashmla_extension_C, m) {
    m.def("fwd_kvcache_mla_fp8(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, Tensor num_splits, Tensor? descale_q, Tensor? descale_k) -> Tensor[]");
    m.def("get_mla_decoding_metadata_dense_fp8(Tensor seqlens_k, int num_heads_per_head_k, int num_heads_k) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(_flashmla_extension_C, CUDA, m) {
    m.impl("fwd_kvcache_mla_fp8", TORCH_BOX(&fwd_kvcache_mla_fp8));
    m.impl("get_mla_decoding_metadata_dense_fp8", TORCH_BOX(&get_mla_decoding_metadata_dense_fp8));
}

PyMODINIT_FUNC PyInit__flashmla_extension_C() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_flashmla_extension_C", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
