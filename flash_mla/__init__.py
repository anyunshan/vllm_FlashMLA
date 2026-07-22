__version__ = "1.0.0"

import torch
from pathlib import Path

# Load the compiled extension so its STABLE_TORCH_LIBRARY registrations run.
# Operators are then available as torch.ops._flashmla_C_novita.<op>. The glob
# matches the abi3 name (_flashmla_C_novita.abi3.so) and the cpython-*.so name.
# The namespace is renamed from _flashmla_C so this wheel can coexist in one
# process with vLLM's own vllm._flashmla_C extension.
_so_files = list(Path(__file__).parent.glob("_flashmla_C_novita*.so"))
assert len(_so_files) == 1, (
    f"Expected one _flashmla_C_novita*.so file, found {_so_files}"
)
torch.ops.load_library(_so_files[0])

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd,
    sparse_native_fp8_swapsab_tp2_decode_fwd,
)

__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd",
    "sparse_native_fp8_swapsab_tp2_decode_fwd",
]
