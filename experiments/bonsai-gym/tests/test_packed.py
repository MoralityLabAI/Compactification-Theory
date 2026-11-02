import os
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bonsai_gym import ptq1_codec  # noqa: E402
from bonsai_gym.gdn import chunk_gated_delta_rule, recurrent_gated_delta_rule  # noqa: E402
from bonsai_gym.packed import (PackedLinear, decode, hadamard_matrix, inverse_rotate, rotate,  # noqa: E402
                               split_blocks, ternarize_ste)

GGUF = os.environ.get("BONSAI_GGUF", r"C:\Users\patri\models\ternary-bonsai-2-27b\Ternary-Bonsai-2-27B-PTQ1_0.gguf")


def _random_payload(rows, cols, seed=0):
    rng = random.Random(seed)
    trits, scales, raw = [], [], bytearray()
    for _ in range(rows):
        row, srow = [], []
        for _ in range(cols // 128):
            t = [rng.choice((-1, 0, 1)) for _ in range(128)]
            s = rng.uniform(0.001, 0.05)
            block = ptq1_codec.encode_ptq1_block(t, s)
            raw += block
            row += t
            srow.append(ptq1_codec.decode_ptq1_block(block)[1])
        trits.append(row)
        scales.append(srow)
    return bytes(raw), torch.tensor(trits, dtype=torch.float32), torch.tensor(scales, dtype=torch.float32)


def test_decode_matches_reference_codec():
    raw, trits, scales = _random_payload(5, 384)
    codes, sc = split_blocks(torch.frombuffer(bytearray(raw), dtype=torch.uint8), 5, 384)
    dense = decode(codes, sc, torch.float32)
    expect = (trits.reshape(5, 3, 128) * scales[..., None]).reshape(5, 384)
    assert torch.equal(dense, expect)


@pytest.mark.skipif(not Path(GGUF).exists(), reason="Bonsai 2 27B GGUF not on disk")
def test_decode_matches_reference_on_real_rows():
    from bonsai_gym.gguf_reader import GGUFFile
    with GGUFFile(GGUF) as g:
        for name in ("blk.0.attn_qkv.weight", "blk.63.ffn_down.weight", "output.weight"):
            t = g.tensors[name]
            raw = bytes(g.payload(name, first_row=7, rows=2))
            q, s = ptq1_codec.decode_matrix(raw, rows=2, cols=t.row_length)
            expect = torch.tensor(q, dtype=torch.float32).reshape(2, -1, 128) * torch.tensor(s)[..., None]
            codes, sc = split_blocks(torch.frombuffer(bytearray(raw), dtype=torch.uint8), 2, t.row_length)
            assert torch.equal(decode(codes, sc, torch.float32), expect.reshape(2, -1)), name


def test_rotation_is_orthogonal_and_inverts():
    h = hadamard_matrix(16, dtype=torch.float64)
    assert torch.allclose(h @ h, torch.eye(16, dtype=torch.float64), atol=1e-12)
    x = torch.randn(3, 64, dtype=torch.float64)
    s = torch.where(torch.rand(64) < 0.5, -1.0, 1.0).double()
    # the embedding inverse s*H(z) undoes the forward transform H(s*x)
    assert torch.allclose(inverse_rotate(rotate(x, s, h), s, h), x, atol=1e-12)


def _linear(rows=6, cols=256, perm=None, seed=1):
    raw, _, _ = _random_payload(rows, cols, seed)
    codes, sc = split_blocks(torch.frombuffer(bytearray(raw), dtype=torch.uint8), rows, cols)
    s = torch.where(torch.rand(cols, generator=torch.Generator().manual_seed(seed)) < 0.5, -1.0, 1.0)
    h = hadamard_matrix(128)
    return PackedLinear("blk.0.test.weight", codes, sc, cols, s, h, perm, chunk_elems=2 * cols)


def test_packed_forward_backward_match_dense():
    lin = _linear()
    lin.add_lora(4, 8.0, torch.Generator().manual_seed(0))
    torch.nn.init.normal_(lin.lora_B, std=0.1)
    x = torch.randn(5, 256, requires_grad=True)
    y = lin(x)
    w = lin.dense_weight()
    ref_x = x.detach().clone().requires_grad_(True)
    a, b = lin.lora_A.detach().clone().requires_grad_(True), lin.lora_B.detach().clone().requires_grad_(True)
    ref = rotate(ref_x, lin.signs, lin.hmat) @ w.t() + (ref_x @ a.t()) @ b.t() * lin.lora_scale
    assert torch.allclose(y, ref, atol=1e-5)
    gy = torch.randn_like(y)
    y.backward(gy)
    ref.backward(gy)
    assert torch.allclose(x.grad, ref_x.grad, atol=1e-5)
    assert torch.allclose(lin.lora_A.grad, a.grad, atol=1e-5)
    assert torch.allclose(lin.lora_B.grad, b.grad, atol=1e-5)
    assert lin.codes.grad is None


def test_ssm_out_permutation_is_tiled_to_grouped():
    lin = _linear(rows=2, cols=256, perm=(32, 2, 4))
    x = torch.arange(256, dtype=torch.float32)[None]
    lin.signs = None
    z = lin.rotated_input(x)[0].long()
    # tiled head h = r*nk + k holds x[h*32:(h+1)*32]; grouped position is k*rep + r
    for k in range(2):
        for r in range(4):
            h_tiled, h_grouped = r * 2 + k, k * 4 + r
            assert torch.equal(z[h_grouped * 32:(h_grouped + 1) * 32], torch.arange(h_tiled * 32, (h_tiled + 1) * 32))


def test_ternary_ste_passes_gradient():
    w = torch.randn(4, 4, requires_grad=True)
    q = ternarize_ste(w)
    s = w.detach().abs().mean()
    assert torch.allclose(q.detach().abs(), torch.where(q.detach() == 0, 0.0, s))
    q.sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_chunked_delta_rule_matches_recurrent():
    torch.manual_seed(0)
    b, t, h, dk, dv = 1, 100, 3, 16, 8
    q = torch.nn.functional.normalize(torch.randn(b, t, h, dk), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, t, h, dk), dim=-1)
    v = torch.randn(b, t, h, dv)
    g = -torch.rand(b, t, h) * 0.5
    beta = torch.rand(b, t, h)
    ref = recurrent_gated_delta_rule(q, k, v, g, beta)
    out = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32)
    assert torch.allclose(out, ref, atol=1e-4)
