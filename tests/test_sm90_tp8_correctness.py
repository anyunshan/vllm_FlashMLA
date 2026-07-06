"""
Correctness tests for the SM90 TP=8 sparse decode kernel (sparse_fp8_swapsab_tp8).

Routes through `flash_mla.flash_mla_with_kvcache` -> SparseAttnDecodeParams ->
Decode_Sm90_Impl::run_, which dispatches h_q=8 to the swap-AB path
(P^T = K @ Q^T, fills the wgmma M=64 atom along the token axis).

NOTE: unlike the original dev repo, this fork ships only the swap-AB variant
(the padded_tp8 benchmark baseline and its FLASHMLA_SM90_TP8_USE_SWAPSAB env
switch were intentionally not integrated).

Workloads are limited to V32 (d_qk=576), s_q=1, h_q=8, no extra_kv, no
topk_length, no attn_sink (matches the kernel's hard asserts).
"""
import sys
from typing import List, Tuple

import torch
import kernelkit as kk

import flash_mla

import lib
from lib import RawTestParamForDecode as RawTestParam
import ref


def gen_h8_cases() -> List[RawTestParam]:
    cases: List[RawTestParam] = []
    for b in [2, 16, 64]:
        for s_kv in [2048, 8192]:
            for topk in [256, 1024, 2048]:
                if topk > s_kv:
                    continue
                cases.append(RawTestParam(
                    b=b, h_q=8, s_q=1, h_kv=1, s_kv=s_kv,
                    is_varlen=True, topk=topk,
                    enable_attn_sink=False,
                    have_topk_length=False,
                    extra_s_k=None, extra_topk=None,
                    block_size=64, extra_block_size=None,
                    have_extra_topk_length=False,
                    d_qk=576, d_v=512,
                    check_correctness=True, num_runs=0,
                ))
    # All-indices-invalid corner case: exercises the MAX_INIT_VAL=-1e30 nan-guard
    # (rM stays at MAX_INIT_VAL because no valid token contributes), the
    # rL==0 -> o_scales=0 path in store_o, and the lse=+inf "lonely q" branch.
    for b, s_kv, topk in [(2, 2048, 256), (16, 8192, 1024)]:
        cases.append(RawTestParam(
            b=b, h_q=8, s_q=1, h_kv=1, s_kv=s_kv,
            is_varlen=True, topk=topk,
            is_all_indices_invalid=True,
            enable_attn_sink=False,
            have_topk_length=False,
            extra_s_k=None, extra_topk=None,
            block_size=64, extra_block_size=None,
            have_extra_topk_length=False,
            d_qk=576, d_v=512,
            check_correctness=True, num_runs=0,
        ))
    return cases


@torch.inference_mode()
def run_one_case(p, t, sched_meta) -> bool:
    torch.cuda.synchronize()
    out_ans, lse_ans = lib.run_flash_mla_decode(p, t, sched_meta, None)
    torch.cuda.synchronize()

    out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)

    ok_out = kk.check_is_allclose(
        "[swapsab_tp8] out", out_ans, out_ref,
        abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6,
    )
    ok_lse = kk.check_is_allclose(
        "[swapsab_tp8] lse", lse_ans, lse_ref,
        abs_tol=1e-6, rel_tol=8.01 / 65536,
    )
    return ok_out and ok_lse


def main():
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    raw_cases = gen_h8_cases()
    print(f"{kk.colors['CYAN_BG']}{len(raw_cases)} h_q=8 testcases{kk.colors['CLEAR']}")

    fails: List[RawTestParam] = []
    counter = kk.Counter()
    for raw in raw_cases:
        if raw.seed == -1:
            raw.seed = counter.next()
        torch.cuda.empty_cache()
        p = raw.to_test_param()
        print("=" * 60)
        print(f"Case: b={raw.b} s_kv={raw.s_kv} topk={raw.topk}"
              f" all_invalid={raw.is_all_indices_invalid}")
        t = lib.generate_testcase_for_decode(p)
        sched_meta, _ = flash_mla.get_mla_metadata()
        ok = run_one_case(p, t, sched_meta)
        if not ok:
            fails.append(raw)

    print("=" * 60)
    total = len(raw_cases)
    if fails:
        print(f"{kk.colors['RED_BG']}{total - len(fails)}/{total} runs passed{kk.colors['CLEAR']}")
        for raw in fails:
            print(f"  FAIL: b={raw.b} s_kv={raw.s_kv} topk={raw.topk}"
                  f" all_invalid={raw.is_all_indices_invalid}")
        sys.exit(1)
    else:
        print(f"{kk.colors['GREEN_BG']}{total}/{total} runs passed{kk.colors['CLEAR']}")


if __name__ == "__main__":
    main()
