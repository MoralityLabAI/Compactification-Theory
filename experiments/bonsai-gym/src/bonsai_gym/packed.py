"""Packed-resident PTQ1_0 weights on the GPU, with LoRA in the primal basis.

The base weights stay in their 28-byte/128-weight PTQ1_0 packing on the device
(codes and FP16 scales split into two tensors). Every matmul decodes a slice of
rows to the compute dtype, multiplies, and frees it; backward decodes again
instead of saving the dense weight. Nothing dense the size of a layer is kept.

Decoding follows `ptq1_codec.decode_ptq1_block` (PrismML-Eng/llama.cpp
c96baf4, ggml-quants.c), which was checked bit-exact against the fork's C
decoder on every PTQ1_0 tensor of the 27B file (HybridTRMLDT S0 receipt
row_parity_20261031). `tests/test_packed.py` checks this decoder against it.

Activation side, as in the fork's `build_lora_mm`: base(x) = W_rot @ H(s * P x),
where P is the GDN head permutation (ssm_out only), s the explicit sign vector
for the input width and H the normalized Sylvester Walsh-Hadamard transform in
blocks of 1024. LoRA reads the unpermuted, unrotated x, so a trained adapter
exports as an ordinary GGUF LoRA for the fork.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

GROUP = 128
BLOCK_BYTES = 28
CODE_BYTES = 26


def _digit(code: int, place: int) -> int:
    return ((((code * (3 ** place)) & 0xFF) * 3) >> 8) - 1


def _tables() -> tuple[torch.Tensor, torch.Tensor]:
    lut = torch.tensor([[_digit(c, n) for n in range(5)] for c in range(256)], dtype=torch.float32)
    order = []
    for start, width in ((0, 16), (16, 8)):
        for n in range(5):
            for m in range(width):
                order.append((start + m) * 5 + n)
    for n in range(4):
        for h in range(2):
            order.append((24 + h) * 5 + n)
    assert len(order) == GROUP and len(set(order)) == GROUP
    return lut, torch.tensor(order, dtype=torch.long)


_LUT, _ORDER = _tables()
_DEVICE_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _device_tables(device, dtype):
    key = (str(device), dtype)
    if key not in _DEVICE_TABLES:
        _DEVICE_TABLES[key] = (_LUT.to(device=device, dtype=dtype), _ORDER.to(device))
    return _DEVICE_TABLES[key]


def split_blocks(raw: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 payload [rows * cols/128 * 28] -> codes [rows, G, 26] uint8, scales [rows, G] fp16."""
    if cols % GROUP:
        raise ValueError("row length must be a multiple of 128")
    g = cols // GROUP
    blocks = raw.reshape(rows, g, BLOCK_BYTES)
    codes = blocks[..., :CODE_BYTES].contiguous()
    scales = blocks[..., CODE_BYTES:].contiguous().view(torch.float16).reshape(rows, g)
    return codes, scales


def decode(codes: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """codes [R, G, 26] uint8 + scales [R, G] -> dense [R, G*128] in `dtype`."""
    lut, order = _device_tables(codes.device, dtype)
    r, g, _ = codes.shape
    digits = F.embedding(codes.to(torch.int32), lut)          # [R, G, 26, 5]
    trits = digits.reshape(r, g, CODE_BYTES * 5)[..., order]  # [R, G, 128]
    return (trits * scales.to(dtype).unsqueeze(-1)).reshape(r, g * GROUP)


def hadamard_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError("Hadamard size must be a power of two")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(n)).to(device=device, dtype=dtype)


def rotate(x: torch.Tensor, signs: torch.Tensor, hmat: torch.Tensor) -> torch.Tensor:
    """H(s * x) blockwise over the last axis (H symmetric, so x @ H == H x per block)."""
    b = hmat.shape[0]
    y = (x * signs.to(x.dtype)).reshape(*x.shape[:-1], -1, b)
    return (y @ hmat.to(x.dtype)).reshape(x.shape)


def inverse_rotate(z: torch.Tensor, signs: torch.Tensor, hmat: torch.Tensor) -> torch.Tensor:
    """s * H(z): restores a primal row from a Hadamard-latent table row."""
    b = hmat.shape[0]
    y = z.reshape(*z.shape[:-1], -1, b) @ hmat.to(z.dtype)
    return y.reshape(z.shape) * signs.to(z.dtype)


class _PackedMatmul(torch.autograd.Function):
    """z [N, in] @ W^T for packed W, decoded `chunk` rows at a time in both passes."""

    @staticmethod
    def forward(ctx, z, codes, scales, chunk):
        rows = codes.shape[0]
        if chunk >= rows:
            out = z @ decode(codes, scales, z.dtype).t()
        else:
            out = z.new_empty(z.shape[0], rows)
            for r0 in range(0, rows, chunk):
                w = decode(codes[r0:r0 + chunk], scales[r0:r0 + chunk], z.dtype)
                out[:, r0:r0 + chunk] = z @ w.t()
                del w
        ctx.save_for_backward(codes, scales)
        ctx.chunk = chunk
        ctx.zdtype = z.dtype
        return out

    @staticmethod
    def backward(ctx, gout):
        codes, scales = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        rows, chunk = codes.shape[0], ctx.chunk
        gz = None
        for r0 in range(0, rows, chunk):
            w = decode(codes[r0:r0 + chunk], scales[r0:r0 + chunk], ctx.zdtype)
            part = (gout[:, r0:r0 + chunk].to(ctx.zdtype) @ w).float()
            gz = part if gz is None else gz.add_(part)
            del w
        return gz.to(ctx.zdtype), None, None, None


def ternarize_ste(w: torch.Tensor) -> torch.Tensor:
    """Absmean ternary quantization with a straight-through gradient (BitNet b1.58 rule)."""
    scale = w.abs().mean().clamp_min(1e-8)
    q = (w / scale).round().clamp(-1, 1) * scale
    return w + (q - w).detach()


class PackedLinear(nn.Module):
    """Frozen packed PTQ1_0 projection with an optional LoRA on the primal input.

    `perm` is (hd, nk, rep) for the GDN output projection: the input arrives in
    the runtime's tiled head order and the folded weight expects grouped order.
    """

    def __init__(self, name: str, codes: torch.Tensor, scales: torch.Tensor, in_features: int,
                 signs: torch.Tensor | None, hmat: torch.Tensor | None,
                 perm: tuple[int, int, int] | None = None, chunk_elems: int = 1 << 26):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = codes.shape[0]
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.signs = signs
        self.hmat = hmat
        self.perm = perm
        self.chunk = max(1, chunk_elems // in_features)
        self.lora_A: nn.Parameter | None = None
        self.lora_B: nn.Parameter | None = None
        self.lora_scale = 0.0
        self.lora_quant = "none"

    def add_lora(self, rank: int, alpha: float, generator: torch.Generator | None = None,
                 quant: str = "none") -> int:
        dev = self.codes.device
        a = torch.empty(rank, self.in_features, dtype=torch.float32)
        bound = 1.0 / math.sqrt(self.in_features)  # kaiming_uniform(a=sqrt(5)) bound, as PEFT
        a.uniform_(-bound, bound, generator=generator)
        self.lora_A = nn.Parameter(a.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=dev))
        self.lora_scale = alpha / rank
        if quant not in ("none", "ternary_ste"):
            raise ValueError(f"unknown adapter quantization {quant}")
        self.lora_quant = quant
        return a.numel() + self.out_features * rank

    def rotated_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.perm is not None:
            hd, nk, rep = self.perm
            x = x.reshape(*x.shape[:-1], rep, nk, hd).transpose(-2, -3).reshape(x.shape)
        if self.signs is None:
            return x
        return rotate(x, self.signs, self.hmat)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        lead = x.shape[:-1]
        if z is None:
            z = self.rotated_input(x)
        y = _PackedMatmul.apply(z.reshape(-1, self.in_features), self.codes, self.scales, self.chunk)
        y = y.reshape(*lead, self.out_features)
        if self.lora_A is not None:
            a, b = self.lora_A, self.lora_B
            if self.lora_quant == "ternary_ste":
                a, b = ternarize_ste(a), ternarize_ste(b)
            y = y + F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * self.lora_scale
        return y

    def dense_weight(self, dtype=torch.float32) -> torch.Tensor:
        """W_rot as a dense matrix (tests and small slices only)."""
        return decode(self.codes, self.scales, dtype)
