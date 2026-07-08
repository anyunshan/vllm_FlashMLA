__version__ = "1.0.0"

# NOTE: flash_mla_with_kvcache_fp8 (dense FP8 decode) is intentionally not
# imported here: it lives in the _flashmla_extension_C module built only by
# vLLM's CMake (csrc/extension/), which the standalone setup.py build does
# not compile. The vendored-into-vLLM path never imports this __init__.py.
from flash_mla_tp.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd
)

__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd"
]
