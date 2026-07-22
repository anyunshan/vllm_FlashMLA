"""
IEEE FP8 E4M3 FN truth implementation (independent of PyTorch / CUDA).

Used as ground-truth reference for fp8 cast tests, decoupling test correctness
from any specific library's fp8 implementation.

FP8 E4M3 FN spec:
  - 1 sign bit + 4 exponent bits (bias=7) + 3 mantissa bits
  - Normal: value = (-1)^s * (1 + m/8) * 2^(e-7), e in [1, 15]
  - Subnormal: value = (-1)^s * (m/8) * 2^-6 = (-1)^s * m * 2^-9, e == 0
  - Max normal: e=15, m=6 -> +/-448 (bytes 0x7E / 0xFE)
  - NaN: e=15, m=7 only (bytes 0x7F / 0xFF). No infinity (FN variant).

Cast rules:
  - Round-to-nearest-even (RNE) on mantissa
  - Saturate: |x| > 448 -> +/-448
  - NaN input -> 0x7F (positive NaN) or 0xFF (sign preserving)
  - +0 -> 0x00, -0 -> 0x80

Implementation: brute-force enumerate all 254 representable fp8 values
(sorted), use binary search to find nearest with RNE tie-breaking. Slow
but trivially correct (no subnormal/normal boundary edge case bugs).
"""

import numpy as np


def _build_fp8_e4m3fn_table():
    """Enumerate all 254 representable fp8 e4m3fn values (skip 2 NaN encodings)."""
    table = []
    for byte in range(256):
        sign = (byte >> 7) & 1
        exp_field = (byte >> 3) & 0xF
        mant_field = byte & 0x7

        if exp_field == 15 and mant_field == 7:
            continue  # NaN encoding, not a value

        if exp_field == 0:
            value = mant_field * (2.0 ** -9)
        else:
            value = (1.0 + mant_field / 8.0) * (2.0 ** (exp_field - 7))

        if sign == 1:
            value = -value
        table.append((value, byte))

    return sorted(table, key=lambda vb: vb[0])


_FP8_TABLE = _build_fp8_e4m3fn_table()
_FP8_VALUES = np.array([v for v, _ in _FP8_TABLE], dtype=np.float64)
_FP8_BYTES = np.array([b for _, b in _FP8_TABLE], dtype=np.uint8)
_FP8_MAX = 448.0

# Positive-only table for sign-separated cast: avoids the +0/-0 ambiguity in
# RNE tie-breaking near zero (sorted table has both 0x00 and 0x80 mapping to
# 0.0; tie-break by mantissa LSB picks one regardless of input sign, giving
# wrong sign on output). Cast logic strips sign, looks up |val|, then ORs
# sign bit back at end.
_POS_FP8_TABLE = sorted(
    [(v, b) for v, b in _FP8_TABLE if v >= 0.0 and b != 0x80],
    key=lambda vb: vb[0],
)
_POS_FP8_VALUES = np.array([v for v, _ in _POS_FP8_TABLE], dtype=np.float64)
_POS_FP8_BYTES = np.array([b for _, b in _POS_FP8_TABLE], dtype=np.uint8)


def fp32_to_fp8_e4m3fn_truth(x_fp32):
    """fp32 -> fp8 e4m3fn bytes (uint8). RNE saturate, IEEE compliant.

    Sign-separated to avoid +0/-0 RNE tie-break ambiguity:
      sign extracted via np.signbit (correctly handles -0.0)
      |val| looked up in positive-only fp8 table with RNE tie-break
      sign bit OR'd back at end

    Args:
      x_fp32: array-like, will be cast to fp32 internally.
    Returns:
      uint8 numpy array, same shape, raw fp8 bytes.
    """
    x = np.asarray(x_fp32, dtype=np.float32)
    orig_shape = x.shape
    x_flat = x.reshape(-1).astype(np.float64)
    out = np.zeros(x_flat.shape, dtype=np.uint8)

    for i, val in enumerate(x_flat):
        if np.isnan(val):
            out[i] = 0xFF if np.signbit(val) else 0x7F
            continue

        sign_bit = 0x80 if np.signbit(val) else 0x00
        abs_val = abs(val)

        if abs_val == 0.0:
            out[i] = sign_bit  # +0 -> 0x00, -0 -> 0x80
            continue

        if abs_val > _FP8_MAX:
            out[i] = sign_bit | 0x7E  # saturate to +/-448
            continue

        # Find nearest |fp8| value to |val| with RNE tie-breaking
        idx = np.searchsorted(_POS_FP8_VALUES, abs_val)

        if idx == 0:
            chosen = _POS_FP8_BYTES[0]
        elif idx >= len(_POS_FP8_VALUES):
            chosen = _POS_FP8_BYTES[-1]
        else:
            v_lo, v_hi = _POS_FP8_VALUES[idx - 1], _POS_FP8_VALUES[idx]
            b_lo, b_hi = _POS_FP8_BYTES[idx - 1], _POS_FP8_BYTES[idx]
            d_lo, d_hi = abs_val - v_lo, v_hi - abs_val

            if d_lo < d_hi:
                chosen = b_lo
            elif d_hi < d_lo:
                chosen = b_hi
            else:
                # RNE tie: pick byte with even mantissa (LSB = 0)
                mant_lo = b_lo & 0x7
                mant_hi = b_hi & 0x7
                if (mant_lo & 1) == 0:
                    chosen = b_lo
                elif (mant_hi & 1) == 0:
                    chosen = b_hi
                else:
                    chosen = b_lo  # both odd: should not happen

        out[i] = sign_bit | (int(chosen) & 0x7F)

    return out.reshape(orig_shape)


def fp8_e4m3fn_to_fp32_truth(byte_arr):
    """fp8 e4m3fn bytes (uint8) -> fp32 (lossless). NaN bytes -> NaN.

    Args:
      byte_arr: array-like uint8.
    Returns:
      fp32 numpy array, same shape.
    """
    b = np.asarray(byte_arr, dtype=np.uint8)
    orig_shape = b.shape
    b_flat = b.reshape(-1)
    out = np.zeros(b_flat.shape, dtype=np.float32)

    for i, byte in enumerate(b_flat):
        sign = (int(byte) >> 7) & 1
        exp_field = (int(byte) >> 3) & 0xF
        mant_field = int(byte) & 0x7

        if exp_field == 15 and mant_field == 7:
            out[i] = np.nan
        elif exp_field == 0:
            value = mant_field * (2.0 ** -9)
            out[i] = -value if sign else value
        else:
            value = (1.0 + mant_field / 8.0) * (2.0 ** (exp_field - 7))
            out[i] = -value if sign else value

    return out.reshape(orig_shape)
