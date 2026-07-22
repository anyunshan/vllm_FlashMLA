"""
test_native_fp8_swapsab_tp2_e2e: end-to-end for the fused native-FP8 x swap-AB
TP=2 (h_q=32) decode kernel + combine.

Mirrors tests/test_main_kernel_e2e.py (the h_q=64 native e2e), changed to h_q=32
and the sparse_native_fp8_swapsab_tp2_decode_fwd entry point. Same fp64 online-
softmax reference + x448 path-B cast-back + bf16 output cast. Tolerance identical.
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

import flash_mla as fc

sys.path.insert(0, os.path.dirname(__file__))
from fp8_truth import fp8_e4m3fn_to_fp32_truth, fp32_to_fp8_e4m3fn_truth

H_Q = 32                # TP=2
TOPK_BLOCK_SIZE = 64
HEAD_DIM_K = 576
HEAD_DIM_V = 512
FP8_P_SCALE = 448.0
LOG2_E = 1.4426950408889634

ATOL_FLOOR = 2.0 ** -23
E_SAFETY_FP8_ACCUM = 0.05
COS_SIM_MIN_LSE = 0.99999
FLOOR_GAP_TOLERANCE = 0.005
MAX_INIT_VAL = -1e30


def _bytes_to_fp64_truth(byte_tensor: torch.Tensor) -> torch.Tensor:
    bytes_np = byte_tensor.cpu().numpy().astype(np.uint8)
    fp32_np = fp8_e4m3fn_to_fp32_truth(bytes_np)
    return torch.from_numpy(fp32_np.astype(np.float64)).cuda()


def _fp32_to_fp8_truth(x: torch.Tensor) -> torch.Tensor:
    fp32_np = x.cpu().numpy().astype(np.float32)
    bytes_np = fp32_to_fp8_e4m3fn_truth(fp32_np)
    return torch.from_numpy(bytes_np).cuda().view(torch.float8_e4m3fn)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.double().flatten(); b64 = b.double().flatten()
    norm = (a64.norm() * b64.norm()).item()
    if norm == 0.0:
        return 1.0 if (a64 - b64).abs().max().item() == 0.0 else 0.0
    return (a64 @ b64 / norm).item()


def _make_inputs(seed: int, topk: int, b: int = 1, invalid_mode: str = "none"):
    """invalid_mode: 'none' | 'tail' (last quarter of each row = -1) |
    'all' (batch 0's entire row = -1; other batches valid)."""
    torch.manual_seed(seed)
    s_q, h_kv = 1, 1
    page_block_size = 64
    num_blocks = max(2, (topk + page_block_size - 1) // page_block_size + 1) * b
    q_f32 = torch.randn(b, s_q, H_Q, HEAD_DIM_K, dtype=torch.float32, device="cuda") * 0.1
    kv_f32 = torch.randn(num_blocks, page_block_size, h_kv, HEAD_DIM_K, dtype=torch.float32, device="cuda") * 0.1
    q = q_f32.to(torch.float8_e4m3fn)
    kv = kv_f32.to(torch.float8_e4m3fn)
    # Per-batch distinct index rows (each a permutation over the whole cache).
    indices = torch.stack([
        torch.randperm(num_blocks * page_block_size)[:topk] for _ in range(b)
    ]).view(b, s_q, topk).to(torch.int32).cuda()
    if invalid_mode == "tail":
        indices[:, :, (topk * 3) // 4:] = -1          # last quarter invalid, every batch
    elif invalid_mode == "all":
        indices[0, :, :] = -1                          # batch 0 fully invalid
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_K)
    return q, kv, indices, sm_scale, page_block_size


def _gather_K_per_block(kv_bytes, indices_row, page_block_size, block_idx):
    """indices_row: (topk,) int32 for ONE batch. Invalid (-1) entries are
    clamped to 0 for the gather; the caller masks their contribution. (A raw
    -1 // 64 = -1 would wrap to the LAST block under torch negative indexing
    and silently corrupt the reference.)"""
    flat_idx = indices_row.to(torch.int64)
    start = block_idx * TOPK_BLOCK_SIZE
    end = start + TOPK_BLOCK_SIZE
    blk = flat_idx[start:end]
    valid = blk >= 0
    safe = torch.where(valid, blk, torch.zeros_like(blk))
    K = kv_bytes[safe // page_block_size, safe % page_block_size, 0, :]
    return K, valid


def _reference_one_batch(Q_fp64, kv_bytes, indices_row, page_block_size, num_kblocks, sm_scale_log2):
    """fp64 online-softmax reference for ONE batch row. Invalid (-1) tokens are
    masked exactly as the kernel does: their logit -> -inf, i.e. p -> 0 (zero
    contribution to rO/rL). An all-invalid row yields rL=0 -> O=0, LSE=+inf."""
    rL = torch.zeros(H_Q, dtype=torch.float64, device="cuda")
    rM = torch.full((H_Q,), MAX_INIT_VAL, dtype=torch.float64, device="cuda")
    rO = torch.zeros(H_Q, HEAD_DIM_V, dtype=torch.float64, device="cuda")
    abs_dot = torch.zeros(H_Q, HEAD_DIM_V, dtype=torch.float64, device="cuda")
    for block_idx in range(num_kblocks):
        K_bytes, valid = _gather_K_per_block(kv_bytes, indices_row, page_block_size, block_idx)
        if not valid.any():
            continue   # fully invalid block: masked p==0 everywhere, no state change
        K_fp64 = _bytes_to_fp64_truth(K_bytes)
        V_fp64 = K_fp64[:, :HEAD_DIM_V]
        rP = Q_fp64 @ K_fp64.T
        rP_scaled = rP * sm_scale_log2
        rP_scaled = torch.where(valid.unsqueeze(0), rP_scaled,
                                torch.full_like(rP_scaled, float('-inf')))
        cur_max = rP_scaled.max(dim=1).values          # finite: block has >=1 valid token
        new_rM = torch.maximum(rM, cur_max)
        scale_for_old = torch.exp2(rM - new_rM)
        cur_rP = torch.exp2(rP_scaled - new_rM.unsqueeze(1))   # invalid -> exp2(-inf) = 0
        rS_x448 = (cur_rP.float() * FP8_P_SCALE).contiguous()
        rS_fp8 = _fp32_to_fp8_truth(rS_x448)
        rS_fp64 = _bytes_to_fp64_truth(rS_fp8.view(torch.uint8))
        rO = rO * scale_for_old.unsqueeze(1) + rS_fp64 @ V_fp64
        abs_dot = abs_dot * scale_for_old.unsqueeze(1) + rS_fp64.abs() @ V_fp64.abs()
        rL = rL * scale_for_old + cur_rP.sum(dim=1)
        rM = new_rM
    rL_fp32 = rL.float()
    o_scale = torch.where(rL_fp32 == 0.0, torch.zeros_like(rL_fp32), 1.0 / (FP8_P_SCALE * rL_fp32))
    O_scaled = (rO.float() * o_scale.unsqueeze(1)).contiguous()
    O_post_cast = O_scaled.to(torch.bfloat16).float()
    abs_dot_scaled = (abs_dot * o_scale.unsqueeze(1).abs()).float()
    LSE_out = torch.where(rL == 0.0, torch.full_like(rL, float('inf')),
                          torch.log(rL) + rM / LOG2_E).float()
    return O_scaled, O_post_cast, LSE_out, abs_dot_scaled


def _run_and_check(topk: int, seed: int, b: int = 1, invalid_mode: str = "none"):
    """Shared driver: run the kernel once, then verify EVERY batch row against
    the per-batch fp64 reference."""
    q, kv, indices, sm_scale, page_block_size = _make_inputs(
        seed=seed, topk=topk, b=b, invalid_mode=invalid_mode)
    num_kblocks = topk // TOPK_BLOCK_SIZE
    sm_scale_log2 = sm_scale * LOG2_E

    empty_i32 = torch.empty(0, dtype=torch.int32, device="cuda")
    out, lse, _tsm, _ns = fc.sparse_native_fp8_swapsab_tp2_decode_fwd(
        q, kv, indices, sm_scale, empty_i32, empty_i32)
    torch.cuda.synchronize()

    kv_bytes = kv.view(torch.uint8)
    for bi in range(b):
        Q_fp64 = _bytes_to_fp64_truth(q.view(torch.uint8)[bi, 0])
        O_ref_fp32, O_ref_post, LSE_ref, abs_dot_scaled = _reference_one_batch(
            Q_fp64, kv_bytes, indices[bi, 0], page_block_size, num_kblocks, sm_scale_log2)

        O_kernel = out[bi, 0].float()

        if torch.isinf(LSE_ref).all():
            # Fully-invalid row: kernel must produce exact zeros + LSE=+inf.
            assert (O_kernel == 0).all(), \
                f"[b={bi}] all-invalid row: expected zero output, got max|O|={O_kernel.abs().max()}"
            assert torch.isinf(lse[bi, 0].float()).all(), \
                f"[b={bi}] all-invalid row: expected LSE=+inf"
            print(f"  [batch {bi}] all-invalid row: O==0, LSE==+inf OK")
            continue

        cs_ref_floor = _cos_sim(O_ref_fp32, O_ref_post)
        cs = _cos_sim(O_kernel, O_ref_post)
        err = (O_kernel - O_ref_post).abs()
        threshold = ATOL_FLOOR + E_SAFETY_FP8_ACCUM * abs_dot_scaled
        n_violate = (err > threshold).sum().item()

        print(f"  [batch {bi}] cos_sim={cs:.7f} ref_floor={cs_ref_floor:.7f} "
              f"gap={cs_ref_floor - cs:+.4f} n_violate={n_violate}/{O_ref_post.numel()}")

        assert cs >= cs_ref_floor - FLOOR_GAP_TOLERANCE, (
            f"[b={bi}] cos_sim={cs:.7f} below ref fp8 floor {cs_ref_floor:.7f}")
        # Per acceptance review P2-3: local corruption must not hide under a
        # global cos_sim. Observed clean runs give n_violate == 0.
        assert n_violate <= 16, f"[b={bi}] n_violate={n_violate} > 16 (local corruption?)"

        lse_kernel = lse[bi, 0, :].float()
        valid = ~torch.isinf(LSE_ref)
        if valid.any():
            cs_lse = _cos_sim(lse_kernel[valid], LSE_ref[valid])
            assert cs_lse >= COS_SIM_MIN_LSE, f"[b={bi}] LSE cos_sim={cs_lse:.7f}"


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 123])
@pytest.mark.parametrize("topk", [64, 256, 2048])
def test_native_fp8_swapsab_tp2_e2e(topk: int, seed: int):
    """Original single-batch sweep."""
    print(f"\n[topk={topk} seed={seed} b=1]")
    _run_and_check(topk=topk, seed=seed, b=1)


@pytest.mark.parametrize("b", [2, 5, 17])
@pytest.mark.parametrize("topk", [256, 2048])
def test_multi_batch(topk: int, b: int):
    """P1-1: multi-batch persistent loop (cross-batch Q TMA phase alternation,
    bar_phase_k continuation, next-batch Q prefetch overlap) -- the machinery
    b=1 never exercises. Every batch row checked independently."""
    print(f"\n[topk={topk} b={b} multi-batch]")
    _run_and_check(topk=topk, seed=7, b=b)


@pytest.mark.parametrize("invalid_mode", ["tail", "all"])
def test_invalid_indices(invalid_mode: str):
    """P1-2: invalid (-1) indices. 'tail' = last quarter of every row invalid
    (producer zero-fill + softmax mask path); 'all' = batch 0 fully invalid
    (rL=0 sentinel: O=0, LSE=+inf) while batch 1 stays valid."""
    print(f"\n[invalid_mode={invalid_mode}]")
    _run_and_check(topk=2048, seed=11, b=2, invalid_mode=invalid_mode)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
