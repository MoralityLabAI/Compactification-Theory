import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bonsai_gym.adapter_io import write_gguf_lora  # noqa: E402
from bonsai_gym.data import Example, Sampler, collate, lcs_reward, render_prompt  # noqa: E402
from bonsai_gym.gguf_reader import GGUFFile  # noqa: E402

GGUF = Path(r"C:\Users\patri\models\ternary-bonsai-2-27b\Ternary-Bonsai-2-27B-PTQ1_0.gguf")


def test_gguf_lora_round_trip(tmp_path):
    a = torch.randn(4, 256)
    b = torch.randn(512, 4)
    p = tmp_path / "a.gguf"
    write_gguf_lora({"blk.0.ffn_up.weight.lora_a": a, "blk.0.ffn_up.weight.lora_b": b}, p, alpha=8.0)
    with GGUFFile(p) as g:
        assert g.metadata["general.type"] == "adapter"
        assert g.metadata["adapter.lora.alpha"] == 8.0
        ta, tb = g.tensors["blk.0.ffn_up.weight.lora_a"], g.tensors["blk.0.ffn_up.weight.lora_b"]
        assert ta.dims == (256, 4) and tb.dims == (4, 512)
        got = np.frombuffer(g._mm[ta.offset:ta.offset + a.numel() * 4], dtype="<f4").reshape(4, 256)
        assert np.array_equal(got, a.numpy())


@pytest.mark.skipif(not GGUF.exists(), reason="GGUF not on disk")
def test_prompt_matches_gguf_template():
    jinja2 = pytest.importorskip("jinja2")
    with GGUFFile(GGUF) as g:
        tmpl = g.metadata["tokenizer.chat_template"]
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    env = jinja2.Environment()
    want = env.from_string(tmpl).render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
    assert render_prompt(msgs) == want


def test_collate_labels_shifted_and_padded():
    e1 = Example("a", [1, 2, 3], [4, 5], {})
    e2 = Example("b", [1, 2], [6], {})
    ids, lab = collate([e1, e2], 0, "cpu")
    assert ids.tolist() == [[1, 2, 3, 4, 5], [1, 2, 6, 0, 0]]
    assert lab.tolist() == [[-100, -100, 4, 5, -100], [-100, 6, -100, -100, -100]]


def test_sampler_resume_is_identical():
    s = Sampler(10, 3)
    s.next(7)
    st = s.state()
    a = s.next(9)
    t = Sampler(10, 99)
    t.load(st)
    assert t.next(9) == a


def test_lcs_reward():
    assert lcs_reward("<reversed_text>\ncba\n</reversed_text>", "cba") == 1.0
    assert lcs_reward("<reversed_text>cba", "cba") == 0.0
