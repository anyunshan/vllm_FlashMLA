#!/usr/bin/env bash
# =============================================================================
# Stage-1 standalone verification for the SM90 TP small-head decode kernels
# (sparse_fp8_swapsab_tp2/tp4/tp8) grafted onto vllm-project/FlashMLA.
#
# Run this ON THE H200 BOX from the repo root:
#
#   bash tests/build_and_test_tp_decode.sh            # build + all tests
#   bash tests/build_and_test_tp_decode.sh build      # build only
#   bash tests/build_and_test_tp_decode.sh test       # tests only (already built)
#
# What it does:
#   1. pip install -e . (standalone pybind build via csrc/api/api.cpp;
#      SM100 disabled — H200 is SM90, and this avoids the nvcc>=12.9 gate)
#   2. Runs the three TP correctness suites (new functionality)
#   3. Runs the stock sparse decoding test (h_q=64/128 regression: proves the
#      dispatch merge didn't break the original paths)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

STEP="${1:-all}"

build() {
    echo "==> git submodule update --init csrc/cutlass"
    git submodule update --init csrc/cutlass

    echo "==> building flash_mla (SM90 only)"
    FLASH_MLA_DISABLE_SM100=1 python -m pip install --no-build-isolation -v -e .
}

run_tests() {
    cd "${ROOT_DIR}/tests"

    echo "==> [new] TP=2 (h_q=32) correctness"
    python test_sm90_tp2_correctness.py

    echo "==> [new] TP=4 (h_q=16) correctness"
    python test_sm90_tp4_correctness.py

    echo "==> [new] TP=8 (h_q=8) correctness"
    python test_sm90_tp8_correctness.py

    echo "==> [regression] stock sparse decoding (h_q=64/128)"
    python test_flash_mla_sparse_decoding.py

    echo "==> ALL PASSED"
}

case "${STEP}" in
    build) build ;;
    test)  run_tests ;;
    all)   build; run_tests ;;
    *) echo "Usage: $0 [build|test|all]"; exit 1 ;;
esac
