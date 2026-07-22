import os
from pathlib import Path
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME
)


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]

def get_features_args():
    # ABI-stable flags (always on): TORCH_TARGET_VERSION pins the minimum runtime
    # PyTorch version and bans unstable ATen/c10/torch headers at compile time;
    # USE_CUDA exposes aoti_torch_get_current_cuda_stream from the shim.
    # FLASH_MLA_ENABLE_DENSE_BWD registers dense_prefill_bwd (for when we build
    # the extension directly in the fork); vLLM's integrated build will omit it as
    # it is inference-only.
    features_args = [
        "-DTORCH_TARGET_VERSION=0x020a000000000000",  # PyTorch >= 2.10 at runtime
        "-DUSE_CUDA",
        "-DFLASH_MLA_ENABLE_DENSE_BWD",
    ]
    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    return features_args

def get_arch_flags():
    # Check NVCC Version
    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable. For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), '--version'], stderr=subprocess.STDOUT
    ).decode('utf-8')
    nvcc_version_number = nvcc_version.split('release ')[1].split(',')[0].strip()
    major, minor = map(int, nvcc_version_number.split('.'))
    print(f'Compiling using NVCC {major}.{minor}')

    DISABLE_SM100 = is_flag_set("FLASH_MLA_DISABLE_SM100")
    DISABLE_SM90 = is_flag_set("FLASH_MLA_DISABLE_SM90")
    if major < 12 or (major == 12 and minor <= 8):
        assert DISABLE_SM100, "sm100 compilation for Flash MLA requires NVCC 12.9 or higher. Please set FLASH_MLA_DISABLE_SM100=1 to disable sm100 compilation, or update your environment."    # TODO Implement this

    arch_flags = []
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100f,code=sm_100f"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    return arch_flags

def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]

this_dir = os.path.dirname(os.path.abspath(__file__))

# No git invocations at build time (build scripts must stay git-free): require
# the cutlass submodule to be present instead of initializing it here.
assert os.path.exists(os.path.join(this_dir, "csrc", "cutlass", "include")), (
    "csrc/cutlass is missing. Run `git submodule update --init csrc/cutlass` "
    "once before building."
)

def get_extra_include_dirs():
    # CUDA 13 moved CCCL/libcudacxx headers to $CUDA_HOME/include/cccl; g++
    # compiling host sources (csrc/api/api.cpp) fails on <cuda/std/...>
    # without it. No-op on CUDA 12.
    extra_dirs = []
    if CUDA_HOME is not None:
        cccl_dir = os.path.join(CUDA_HOME, "include", "cccl")
        if os.path.isdir(cccl_dir):
            extra_dirs.append(Path(cccl_dir))
    return extra_dirs

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla._flashmla_C_novita",
        sources=[
            # API
            "csrc/api/api.cpp",

            # Misc kernels for decoding
            "csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
            "csrc/smxx/decode/combine/combine.cu",

            # sm90 dense decode
            "csrc/sm90/decode/dense/instantiations/fp16.cu",
            "csrc/sm90/decode/dense/instantiations/bf16.cu",

            # sm90 sparse decode
            "csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu",

            # sm90 native FP8 x swap-AB TP=2 (h_q=32) sparse decode
            "csrc/sm90/decode/sparse_native_fp8_swapsab_tp2/instantiations/v32_h32.cu",

            # sm90 sparse prefill
            "csrc/sm90/prefill/sparse/fwd.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu",

            # sm100 dense prefill & backward
            "csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
            "csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",

            # sm100 sparse prefill
            "csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu",
            "csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu",
            "csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu",
            "csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu",
            "csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu",

            # sm100 sparse decode
            "csrc/sm100/decode/head64/instantiations/v32.cu",
            "csrc/sm100/decode/head64/instantiations/model1.cu",
            "csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": [
                "-O3",
                "-std=c++20",
                "-DNDEBUG",
                "-D_USE_MATH_DEFINES",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use",
                "-lineinfo",
                "--source-in-ptx",
            ] + get_features_args() + get_arch_flags() + get_nvcc_thread_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "kerutils" / "include",   # TODO Remove me
            Path(this_dir) / "csrc" / "sm90",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
        ] + get_extra_include_dirs(),
        # Build against CPython's Limited API (abi3) so one wheel works across
        # multiple CPython versions, which is possible now that pybind11 is gone
        py_limited_api=True,
    )
)

setup(
    name="flash_mla",
    # Fixed version (no git/datetime): non-editable installs run setup.py
    # twice and mismatched dynamic versions make pip reject the wheel.
    version="1.0.0+nfp8tp2",
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
