"""ARG-02 Nanbeige tests on a tiny random Nanbeige built from the local snapshot's remote code.

Needs torch, transformers < 5 (the remote code targets 4.42-4.46) and the
Nanbeige4.2-3B snapshot in the HF cache; skipped otherwise (as in CI).
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock

try:
    import torch
    import transformers
    from research.algernon.algebraic_reasoning_gym.arg02 import runner
    HAVE = int(transformers.__version__.split(".")[0]) < 5 and \
        (runner.snapshot_dir(load_lock()["nanbeige"]) / "modeling_nanbeige.py").is_file()
except ImportError:
    HAVE = False


def _close(a, b, tol=1e-5):
    for k in ("lp1", "lp2", "lp3"):
        if not torch.allclose(torch.tensor(a[k]).flatten(), torch.tensor(b[k]).flatten(), atol=tol, rtol=0):
            return False
    return True


@unittest.skipUnless(HAVE, "needs transformers < 5 and the local Nanbeige4.2-3B snapshot")
class NanbeigeLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research.algernon.algebraic_reasoning_gym.arg02 import nanbeige, recurrence
        cls.nb, cls.rec = nanbeige, recurrence
        cls.model, cls.tok = runner.tiny_nanbeige()
        cls.ops = runner.op_token_ids(cls.tok)
        g = torch.Generator().manual_seed(2)
        cls.prompt = [166100] + torch.randint(1000, 20000, (33,), generator=g).tolist()
        cls.prompt2 = [166100] + torch.randint(1000, 20000, (25,), generator=g).tolist()
        cls.ids = recurrence.candidate_rows(cls.prompt, cls.ops)
        cls.ids2 = recurrence.candidate_rows(cls.prompt2, cls.ops)

    def test_loop_count_matches_native_forward(self):
        bl = self.nb.NanbeigeLoop(self.model)
        traj = bl.trajectory(self.ids, self.ops, 3, readout_steps={1, 2, 3})
        for t in (1, 2, 3):
            self.model.config.num_loops = t
            with torch.no_grad():
                ref = torch.log_softmax(self.model(input_ids=self.ids, use_cache=False).logits[:, -3:].float(), -1)
            self.assertTrue(_close(traj["records"][t], self.rec.extract(ref, self.ops)), t)
        self.model.config.num_loops = 2

    def test_ablation_identities(self):
        bl = self.nb.NanbeigeLoop(self.model)
        lp = bl.trajectory(self.ids, self.ops, 3, readout_steps={1, 2, 3}, keep_delta={2, 3}, keep_site={2},
                           keep_states=True)
        op = bl.trajectory(self.ids, self.ops, 3, mode="open", readout_steps={1, 3})
        self.assertTrue(_close(lp["records"][1], op["records"][1]))
        sh = bl.trajectory(self.ids, self.ops, 3, mode="shuffle", readout_steps={2, 3}, donor_delta=lp["deltas"])
        self.assertTrue(_close(sh["records"][3], lp["records"][3]))
        pt = bl.trajectory(self.ids, self.ops, 3, readout_steps={3}, patch={"step": 2, "site": lp["sites"][2]})
        self.assertTrue(_close(pt["records"][3], lp["records"][3]))
        donor = bl.trajectory(self.ids2, self.ops, 3, keep_delta={3})
        other = bl.trajectory(self.ids, self.ops, 3, mode="shuffle", readout_steps={3}, donor_delta=donor["deltas"])
        self.assertLess(other["records"][3]["row_spread"], 1e-5)
        self.assertLess(lp["records"][3]["row_spread"], 1e-5)

    def test_whole_stack_only(self):
        with self.assertRaises(ValueError):
            self.nb.NanbeigeLoop(self.model, (0, 1))

    def test_engine_prompt_contract(self):
        from research.algernon.algebraic_reasoning_gym.arg02 import splits
        eng = runner.Engine(self.model, self.tok, None, [], "cpu")
        row = splits.public_record(splits.test_cases()[0])
        ids = eng.prompt_ids(row)
        self.assertEqual(ids[0], 166100)
        self.assertEqual(ids[-len(eng.tail):], eng.tail)
        self.assertEqual(len(set(self.ops.values())), 4)


@unittest.skipUnless(HAVE, "needs transformers < 5 and the local Nanbeige4.2-3B snapshot")
class NanbeigePipelineTests(unittest.TestCase):
    def test_smoke_pipeline(self):
        from research.algernon.algebraic_reasoning_gym.arg02 import smoke
        with TemporaryDirectory() as tmp:
            out = smoke.run(tmp, sample=1, profile="nanbeige")
            receipt = json.loads((out / "arg02_receipt.json").read_text())
            self.assertEqual(receipt["profile"], "nanbeige")
            self.assertIsNone(receipt["ff_control"])
            self.assertIsNone(receipt["decisions"]["H3b_ff"])
            self.assertEqual(receipt["selection"]["block"], [0, 2])
            self.assertNotIn("ff2b@1", receipt["summary"])
            self.assertEqual(receipt["ensemble_sizes"], {"1": 1, "2": 2, "4": 4})
            with self.assertRaises(SystemExit):
                runner.main(["test", "--run-dir", tmp, "--smoke", "--profile", "qwen"])


if __name__ == "__main__":
    unittest.main()
