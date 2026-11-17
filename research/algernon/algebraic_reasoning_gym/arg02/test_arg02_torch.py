"""ARG-02 torch tests on a tiny random Qwen3.5 (CPU, fp32): recurrence identities,
HF parity, leakage guards and the end-to-end smoke. Skipped without torch."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

try:
    import torch
    from transformers import Qwen3_5ForCausalLM  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from spiral_ln import algebraic_reasoning_oracle as oracle


def _close(a, b, tol=1e-5):
    for k in ("lp1", "lp2", "lp3"):
        fa = torch.tensor(a[k]).flatten()
        fb = torch.tensor(b[k]).flatten()
        if not torch.allclose(fa, fb, atol=tol, rtol=0):
            return False
    return True


@unittest.skipUnless(HAVE_TORCH, "torch / transformers with Qwen3.5 not installed")
class RecurrenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research.algernon.algebraic_reasoning_gym.arg02 import recurrence, runner
        cls.rec, cls.runner = recurrence, runner
        cls.model, cls.tok = runner.tiny_model()
        cls.ops = runner.op_token_ids(cls.tok)
        g = torch.Generator().manual_seed(1)
        cls.prompt = torch.randint(0, 100, (37,), generator=g).tolist()
        cls.prompt2 = torch.randint(0, 100, (29,), generator=g).tolist()
        cls.ids = recurrence.candidate_rows(cls.prompt, cls.ops)
        cls.ids2 = recurrence.candidate_rows(cls.prompt2, cls.ops)

    def loop(self, block=(2, 4)):
        return self.rec.BlockLoop(self.model, block)

    def test_depth_one_matches_hf_forward(self):
        with torch.no_grad():
            ref = torch.log_softmax(self.model(input_ids=self.ids).logits[:, -3:].float(), -1)
        ref_rec = self.rec.extract(ref, self.ops)
        for block in ((2, 4), (0, 8), (4, 8)):
            out = self.loop(block).trajectory(self.ids, self.ops, 1, readout_steps={1})
            self.assertTrue(_close(out["records"][1], ref_rec), block)
            self.assertEqual(out["records"][1]["g3"], ref_rec["g3"])

    def test_readout_inside_trajectory_equals_standalone_depth(self):
        bl = self.loop()
        long = bl.trajectory(self.ids, self.ops, 4, readout_steps={1, 2, 3, 4})
        for d in (1, 2, 3, 4):
            short = bl.trajectory(self.ids, self.ops, d, readout_steps={d})
            self.assertTrue(_close(long["records"][d], short["records"][d]), d)
        self.assertEqual(len(long["states"]), 5)
        self.assertEqual(len(long["residual"]), 4)

    def test_open_loop_identities(self):
        bl = self.loop()
        lp = bl.trajectory(self.ids, self.ops, 3, readout_steps={1, 2, 3})
        op = bl.trajectory(self.ids, self.ops, 3, mode="open", readout_steps={1, 2, 3})
        self.assertTrue(_close(lp["records"][1], op["records"][1]))
        self.assertFalse(_close(lp["records"][3], op["records"][3], tol=1e-7))

    def test_shuffle_with_self_donor_is_the_loop(self):
        bl = self.loop()
        donor = bl.trajectory(self.ids, self.ops, 4, keep_delta={2, 4})
        sh = bl.trajectory(self.ids, self.ops, 4, mode="shuffle", readout_steps={2, 4}, donor_delta=donor["deltas"])
        lp = bl.trajectory(self.ids, self.ops, 4, readout_steps={2, 4})
        for t in (2, 4):
            self.assertTrue(_close(sh["records"][t], lp["records"][t]), t)

    def test_shuffle_with_other_case_changes_answers_and_aligns_lengths(self):
        bl = self.loop()
        donor = bl.trajectory(self.ids2, self.ops, 4, keep_delta={2, 4})
        sh = bl.trajectory(self.ids, self.ops, 4, mode="shuffle", readout_steps={4}, donor_delta=donor["deltas"])
        lp = bl.trajectory(self.ids, self.ops, 4, readout_steps={4})
        self.assertFalse(_close(sh["records"][4], lp["records"][4], tol=1e-7))
        # rows sharing a prefix must stay identical whichever case is longer
        self.assertLess(sh["records"][4]["row_spread"], 1e-5)
        donor_long = bl.trajectory(self.ids, self.ops, 4, keep_delta={4})
        sh2 = bl.trajectory(self.ids2, self.ops, 4, mode="shuffle", readout_steps={4}, donor_delta=donor_long["deltas"])
        self.assertLess(sh2["records"][4]["row_spread"], 1e-5)
        idx = self.rec.align_index(37 + 2, 29 + 2, torch.device("cpu"))
        self.assertEqual(len(idx), 39)
        self.assertEqual(idx[-2:].tolist(), [29, 30])  # candidates onto candidates
        self.assertTrue(bool((idx[:-2] <= 28).all()))  # prompt onto donor prompt only
        self.assertEqual(self.rec.align_index(10, 10, torch.device("cpu")).tolist(), list(range(10)))

    def test_self_patch_and_zero_kick_are_identities(self):
        bl = self.loop()
        donor = bl.trajectory(self.ids, self.ops, 3, keep_site={2}, keep_states=True)
        clean = bl.trajectory(self.ids, self.ops, 3, readout_steps={3})
        pt = bl.trajectory(self.ids, self.ops, 3, readout_steps={3}, patch={"step": 2, "site": donor["sites"][2]})
        self.assertTrue(_close(pt["records"][3], clean["records"][3]))
        zk = bl.trajectory(self.ids, self.ops, 3, readout_steps={3}, perturb={"step": 1, "sigma": 0.0, "seed": 3},
                           clean_states=donor["full_states"])
        self.assertTrue(_close(zk["records"][3], clean["records"][3]))
        self.assertTrue(all(d < 1e-5 for d in zk["distance"]))
        kick = bl.trajectory(self.ids, self.ops, 3, readout_steps={3}, perturb={"step": 1, "sigma": 0.5, "seed": 3},
                             clean_states=donor["full_states"])
        self.assertAlmostEqual(kick["distance"][0], kick["kick_norm"], places=3)

    def test_rows_share_prefix(self):
        out = self.loop().trajectory(self.ids, self.ops, 2, readout_steps={2})
        self.assertLess(out["records"][2]["row_spread"], 1e-5)

    def test_unsupported_model_rejected(self):
        class Fake:
            class model:
                layers = []
                class config:
                    model_type = "llama"
        with self.assertRaises(ValueError):
            self.rec.BlockLoop(Fake, (0, 1))


@unittest.skipUnless(HAVE_TORCH, "torch / transformers with Qwen3.5 not installed")
class PipelineTests(unittest.TestCase):
    def test_runner_rejects_non_public_rows(self):
        from research.algernon.algebraic_reasoning_gym.arg02 import runner
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "rows.jsonl"
            p.write_text(json.dumps({"case_id": "x", "topology": "fork", "observed_degree": 12, "demos": [],
                                     "query": [1], "public_sha256": "0" * 64, "future": [1]}) + "\n")
            with self.assertRaises(ValueError):
                runner.read_public(p)

    def test_smoke_pipeline_without_roster_access_in_runner(self):
        from research.algernon.algebraic_reasoning_gym.arg02 import analyze, runner
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            analyze.main(["export", "--run-dir", tmp, "--sample", "1"])
            with patch.object(oracle, "roster", side_effect=AssertionError("runner touched the roster")):
                runner.main(["dev", "--run-dir", tmp, "--stage", "shots", "--smoke"])
            analyze.main(["select", "--run-dir", tmp, "--stage", "shots"])
            with patch.object(oracle, "roster", side_effect=AssertionError("runner touched the roster")):
                runner.main(["dev", "--run-dir", tmp, "--stage", "blocks", "--smoke"])
            analyze.main(["select", "--run-dir", tmp, "--stage", "blocks"])
            with patch.object(oracle, "roster", side_effect=AssertionError("runner touched the roster")):
                runner.main(["test", "--run-dir", tmp, "--smoke"])
                runner.main(["test", "--run-dir", tmp, "--smoke", "--model", "ff"])
                runner.main(["test", "--run-dir", tmp, "--smoke", "--arms", "test_shuffle_a1"])
            analyze.main(["report", "--run-dir", tmp, "--out", str(out)])
            analyze.main(["replay", "--tape", str(out / "arg02_answer_tape.jsonl.gz"),
                          "--check", str(out / "arg02_receipt.json")])
            receipt = json.loads((out / "arg02_receipt.json").read_text())
            self.assertTrue(receipt["smoke"])
            from research.algernon.algebraic_reasoning_gym.arg02 import tables
            profile = json.loads((out / "arg02_profile.json").read_text())
            self.assertIn("## Decision rules", tables.render(receipt, profile))
            self.assertEqual(receipt["summary"]["loop@1"]["negative_accepted"], 0)
            self.assertIn("mechanism_claim", receipt["decisions"])
            self.assertIn("H2b_state_original_shuffle", receipt["decisions"])
            d = receipt["selection"]["depth"]
            self.assertIn(f"loop@{d}_vs_shuffle_a1@{d}", receipt["paired_future12"])
            spreads = [t["record"]["row_spread"] for t in analyze.read_tape(out / "arg02_answer_tape.jsonl.gz")
                       if t["arm"] == "shuffle_a1"]
            self.assertTrue(spreads and max(spreads) < 1e-4)
            # the depth-1 loop and the canonical ensemble member are the same forward
            self.assertEqual(receipt["summary"]["loop@1"]["future12"], receipt["summary"]["ensemble@1"]["future12"])
            # restarting reuses every part
            before = sorted(p.stat().st_mtime_ns for p in (Path(tmp) / "parts").rglob("*.json"))
            runner.main(["test", "--run-dir", tmp, "--smoke"])
            after = sorted(p.stat().st_mtime_ns for p in (Path(tmp) / "parts").rglob("*.json"))
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
