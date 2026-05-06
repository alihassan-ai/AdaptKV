"""4-bit NormalFloat (NF4) quantization utilities for KV cache compression.

Falls back to a pure-PyTorch implementation when bitsandbytes is unavailable
(e.g. on a CPU-only Mac).
"""

from typing import Tuple

import torch
import torch.nn.functional as F

# NF4 codebook: 16 values optimally spaced for normally-distributed data.
# Source: QLoRA paper (Dettmers et al., 2023).
_NF4_CODEBOOK = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477344512939453,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)

_GROUP_SIZE = 64  # Tokens per quantization group


def _try_bitsandbytes_nf4(tensor: torch.Tensor):
    """Attempt to use bitsandbytes for NF4; returns None if unavailable."""
    try:
        import bitsandbytes.functional as bnb_F
        # bitsandbytes expects float16 or float32
        t = tensor.to(torch.float16)
        quant, state = bnb_F.quantize_4bit(t, quant_type="nf4", blocksize=_GROUP_SIZE)
        return quant, state, True
    except Exception:
        return None, None, False


def quantize_to_nf4(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize an FP16/FP32 tensor to 4-bit NormalFloat format.

    Returns:
        packed:  uint8 tensor, values packed 2 per byte; same leading dims,
                 last dim halved (rounded up).
        scales:  per-group absolute max, float32.
        codebook: the NF4 codebook used (for dequantization), float32.
    """
    orig_device = tensor.device
    orig_dtype = tensor.dtype
    flat = tensor.reshape(-1).float()  # work in fp32 for precision
    n = flat.shape[0]

    # Pad to multiple of GROUP_SIZE
    pad = (_GROUP_SIZE - n % _GROUP_SIZE) % _GROUP_SIZE
    if pad:
        flat = F.pad(flat, (0, pad))

    num_groups = flat.shape[0] // _GROUP_SIZE
    groups = flat.view(num_groups, _GROUP_SIZE)

    # Per-group scale = abs_max (avoid zero-division)
    scales = groups.abs().max(dim=1).values.clamp(min=1e-8)  # [num_groups]
    normalized = groups / scales.unsqueeze(1)  # [-1, 1]

    # Find nearest NF4 codebook entry for each element
    cb = _NF4_CODEBOOK.to(flat.device)  # [16]
    diffs = (normalized.unsqueeze(-1) - cb.unsqueeze(0).unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)  # [num_groups, GROUP_SIZE]

    # Pack two 4-bit indices into one uint8
    indices_flat = indices.reshape(-1)  # [num_groups * GROUP_SIZE]
    half_len = (indices_flat.shape[0] + 1) // 2
    packed = torch.zeros(half_len, dtype=torch.uint8, device=flat.device)
    packed[: indices_flat.shape[0] // 2] = (
        (indices_flat[0::2] & 0xF) | ((indices_flat[1::2] & 0xF) << 4)
    )
    if indices_flat.shape[0] % 2 == 1:
        packed[-1] = indices_flat[-1] & 0xF

    return packed.to(orig_device), scales.to(orig_device), cb.to(orig_device)


def dequantize_from_nf4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    original_shape: torch.Size,
    original_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize NF4-packed data back to a floating-point tensor.

    Args:
        packed:         uint8 tensor of packed 4-bit indices.
        scales:         per-group abs_max (float32).
        codebook:       NF4 codebook (float32).
        original_shape: target output shape.
        original_dtype: output floating-point dtype.

    Returns:
        Reconstructed tensor of shape `original_shape` and `original_dtype`.
    """
    device = packed.device

    # Unpack 4-bit indices
    total_elems = scales.shape[0] * _GROUP_SIZE
    low  = (packed & 0xF).to(torch.int64)
    high = ((packed >> 4) & 0xF).to(torch.int64)
    interleaved = torch.stack([low, high], dim=1).reshape(-1)[:total_elems]

    # Lookup codebook values
    cb = codebook.to(device)
    values = cb[interleaved]  # [total_elems]

    # Rescale by per-group abs_max
    num_groups = scales.shape[0]
    groups = values.view(num_groups, _GROUP_SIZE)
    rescaled = groups * scales.unsqueeze(1)

    # Trim padding and reshape
    n_orig = 1
    for s in original_shape:
        n_orig *= s
    return rescaled.reshape(-1)[:n_orig].reshape(original_shape).to(original_dtype)


class QuantizedKVEntry:
    """Stores a single (key, value) pair in NF4 format with metadata for dequantization."""

    __slots__ = ("k_packed", "k_scales", "k_codebook", "k_shape", "k_dtype",
                 "v_packed", "v_scales", "v_codebook", "v_shape", "v_dtype")

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Args:
            keys:   [num_heads, 1, head_dim]
            values: [num_heads, 1, head_dim]
        """
        self.k_shape = keys.shape
        self.k_dtype = keys.dtype
        self.k_packed, self.k_scales, self.k_codebook = quantize_to_nf4(keys)

        self.v_shape = values.shape
        self.v_dtype = values.dtype
        self.v_packed, self.v_scales, self.v_codebook = quantize_to_nf4(values)

    def dequantize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, values) dequantized back to original dtype."""
        keys = dequantize_from_nf4(
            self.k_packed, self.k_scales, self.k_codebook,
            self.k_shape, self.k_dtype,
        )
        values = dequantize_from_nf4(
            self.v_packed, self.v_scales, self.v_codebook,
            self.v_shape, self.v_dtype,
        )
        return keys, values

    def nbytes(self) -> int:
        return (self.k_packed.nelement() * self.k_packed.element_size()
                + self.v_packed.nelement() * self.v_packed.element_size())


# ── Group-wise 4-bit absmax quantization (used by experiments) ────────────────

_GROUP_SIZE = 128


def quantize_4bit(
    tensor: torch.Tensor,
    group_size: int = _GROUP_SIZE,
):
    """Quantize an FP16/FP32 tensor to 4-bit signed integers.

    Groups of `group_size` elements share one absmax scale.
    Quantization range: [-8, 7] (symmetric 4-bit signed).

    Returns:
        quantized: int8 tensor  [num_groups, group_size]
        scale:     FP32 per-group scale  [num_groups, 1]
        shape:     original shape (needed for dequantization)
    """
    import torch.nn.functional as F
    shape = tensor.shape
    flat  = tensor.reshape(-1).float()
    n     = flat.shape[0]

    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))

    groups = flat.reshape(-1, group_size)
    absmax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale  = absmax / 7.0
    quantized = (groups / scale).round().clamp(-8, 7).to(torch.int8)
    return quantized.reshape(-1, group_size), scale, shape


def dequantize_4bit(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Size,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct an approximate FP tensor from 4-bit quantized data."""
    n_orig = 1
    for s in shape:
        n_orig *= s
    dequant = (quantized.float() * scale).reshape(-1)[:n_orig]
    return dequant.reshape(shape).to(target_dtype)


def measure_quantization_error(original: torch.Tensor) -> dict:
    """Quantize then dequantize and measure reconstruction quality."""
    import torch.nn.functional as F
    q, sc, sh = quantize_4bit(original.float())
    recon = dequantize_4bit(q, sc, sh, original.dtype)

    orig_f = original.float()
    rec_f  = recon.float()

    mse = ((orig_f - rec_f) ** 2).mean().item()

    if orig_f.dim() >= 2:
        cos = F.cosine_similarity(
            orig_f.reshape(-1, orig_f.shape[-1]),
            rec_f.reshape(-1, rec_f.shape[-1]),
            dim=-1,
        ).mean().item()
    else:
        cos = F.cosine_similarity(orig_f.unsqueeze(0), rec_f.unsqueeze(0)).item()

    rel_err = ((orig_f - rec_f).abs() / (orig_f.abs() + 1e-6)).mean().item()

    return {
        "mse":              mse,
        "cosine_similarity": cos,
        "relative_error":   rel_err,
        "bits_saved_ratio": 0.75,  # 4-bit vs 16-bit = 75% savings
    }
