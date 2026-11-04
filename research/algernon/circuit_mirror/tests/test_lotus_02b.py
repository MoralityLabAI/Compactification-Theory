"""CPU tests for Algernon-02B. Every feature here is a manufactured fixture.

The upstream-wrapper test builds a tiny random Llama inside the pinned LOTUS
`Lotus` class; it runs only when ALGERNON_LOTUS_UPSTREAM points at that clean
checkout and transformers 4.46.2 is installed. A pass never shows the
published 3B checkpoint running.
"""
import json
import os
import types
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:  # CI installs neither
    np = None
    torch = None

from research.algernon.circuit_mirror.lotus_bridge.counterfactual import (
    build_counterfactual, relabel, validate_counterfactual,
)
from research.algernon.circuit_mirror.lotus_bridge.roster import build_roster, prompt_for

UPSTREAM = os.environ.get("ALGERNON_LOTUS_UPSTREAM")


def _transformers_446():
    try:
        import transformers
        return transformers.__version__ == "4.46.2"
    except ImportError:
        return False


class CounterfactualTests(unittest.TestCase):
    def test_relabel_is_equivariant_and_validated(self):
        roster = build_roster(train=20, calibration=8, test_per_family=6)
        cf = build_counterfactual(roster)
        self.assertEqual(validate_counterfactual(cf, roster), 12)
        fitted = {(tuple(r["successors"]), r["initial"]) for r in roster["cases"] if r["split"] != "test"}
        parents = {r["id"]: r for r in roster["cases"]}
        for row in cf["cases"]:
            parent = parents[row["parent_id"]]
            self.assertEqual(row["trace"], [row["sigma"][s] for s in parent["trace"]])
            self.assertNotIn((tuple(row["successors"]), row["initial"]), fitted)
            self.assertTrue(all(i != v for i, v in enumerate(row["sigma"])))
        bad = json.loads(json.dumps(cf))
        bad["cases"][0]["trace"][-1] = (bad["cases"][0]["trace"][-1] + 1) % 6
        with self.assertRaises(ValueError):
            validate_counterfactual(bad, roster)

    def test_relabel_inverse(self):
        row = {"successors": [1, 2, 0, 0], "initial": 3, "trace": [3, 0, 1, 2]}
        succ, start, trace = relabel(row, [1, 0, 3, 2])
        self.assertEqual((start, trace), (2, [2, 1, 0, 3]))
        state, got = start, [start]
        for _ in range(3):
            state = succ[state]
            got.append(state)
        self.assertEqual(got, trace)

    def test_frozen_counterfactual_matches_frozen_roster(self):
        here = Path(__file__).resolve().parents[1] / "lotus_bridge"
        roster = json.loads((here / "frozen_roster_6x16.json").read_text(encoding="utf-8"))
        cf = json.loads((here / "frozen_counterfactual_6x16.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_counterfactual(cf, roster), 48)
        self.assertEqual(cf, json.loads(json.dumps(build_counterfactual(roster))))


STUB_EVAL = '''import argparse, json
p = argparse.ArgumentParser()
for a in ("--model_id", "--datasets", "--n_looped_iters", "--c_thought", "--device", "--save_preds"):
    p.add_argument(a)
p.add_argument("--bf16", action="store_true")
a = p.parse_args()
rows = json.load(open(a.datasets))
preds = [{"idx": i, "pred": r["answer"], "answer": r["answer"], "correct": int(r["answer"]) % 2 == 0}
         for i, r in enumerate(rows)]
print("  peak GPU mem:  allocated=1.000 GB, reserved=1.000 GB")
json.dump({a.datasets: preds}, open(a.save_preds, "w"))
'''


class Gsm8kPlumbingTests(unittest.TestCase):
    def test_reads_upstream_eval_predictions(self):
        import tempfile
        from research.algernon.circuit_mirror.lotus_bridge.gpu_run import gsm8k_repro
        with tempfile.TemporaryDirectory() as d:
            up = Path(d) / "lotus"
            (up / "scripts").mkdir(parents=True)
            (up / "data").mkdir()
            (up / "data" / "gsm_test.json").write_text(json.dumps(
                [{"question": f"q{i}", "steps": [], "answer": str(i)} for i in range(30)]))
            (up / "scripts" / "eval.py").write_text(STUB_EVAL)
            run = Path(d) / "run"
            run.mkdir()
            args = types.SimpleNamespace(upstream=up, seed=5, gsm8k_n=9, device="cpu", hf_revision="r1",
                                         checkpoint_files_sha256={"model.safetensors": "aa"})
            rec = gsm8k_repro(args, Path(d) / "snapshot", run)
            sample = json.loads((run / "gsm8k_sample.json").read_text())
            self.assertEqual([s["idx"] for s in sample], sorted(s["idx"] for s in sample))
            self.assertEqual((rec["n"], rec["scored_rows"]), (9, 9))
            self.assertEqual(rec["correct"], sum(int(s["answer"]) % 2 == 0 for s in sample))
            self.assertEqual(len(rec["upstream_peak_memory_line"]), 1)
            # A restart reuses the receipt; a changed revision or predictions file reruns eval.py.
            self.assertEqual(gsm8k_repro(args, Path(d) / "snapshot", run), rec | {"reused_from_earlier_attempt": True})
            args.hf_revision = "r2"
            self.assertEqual(gsm8k_repro(args, Path(d) / "snapshot", run) | {"seconds": rec["seconds"]}, rec)
            (run / "gsm8k_preds.json").write_text("{}")
            self.assertNotIn("reused_from_earlier_attempt", gsm8k_repro(args, Path(d) / "snapshot", run))


@unittest.skipIf(np is None, "optional numpy unavailable")
class ResumePartsTests(unittest.TestCase):
    def test_part_is_reused_only_under_its_key_and_hash(self):
        import tempfile
        from research.algernon.circuit_mirror.lotus_bridge.gpu_run import Parts
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "parts"
            arrays = {"blocks": np.random.default_rng(0).standard_normal((3, 2, 2, 4)).astype(np.float32),
                      "readout": np.arange(6, dtype=np.int32)}
            Parts(root, {"rev": 1}).save("full", "test/sink/0", {"input_sha256": "ab"}, arrays)
            value, got = Parts(root, {"rev": 1}).load("full", "test/sink/0")
            self.assertEqual(value, {"input_sha256": "ab"})
            for k, v in arrays.items():
                self.assertEqual(got[k].dtype, v.dtype)
                self.assertTrue(np.array_equal(got[k], v))
            self.assertIsNone(Parts(root, {"rev": 2}).load("full", "test/sink/0"))
            self.assertIsNone(Parts(root, {"rev": 1}).load("full", "test/sink/1"))
            self.assertIsNone(Parts(root, {"rev": 1}).load("cf", "test/sink/0"))
            self.assertIsNone(Parts(None, {"rev": 1}).load("full", "test/sink/0"))
            Parts(root, {"rev": 1}).save("behaviour_loops6", "test/sink/0", {"answer": "3"})
            self.assertEqual(Parts(root, {"rev": 1}).load("behaviour_loops6", "test/sink/0"), ({"answer": "3"}, None))
            self.assertEqual(sorted(p.name for p in (root / "full").iterdir()), ["test_sink_0.json", "test_sink_0.npz"])
            npz = root / "full" / "test_sink_0.npz"
            npz.write_bytes(npz.read_bytes()[:-8])
            self.assertIsNone(Parts(root, {"rev": 1}).load("full", "test/sink/0"))


@unittest.skipIf(torch is None, "optional torch/numpy unavailable")
class BlockCaptureTests(unittest.TestCase):
    def test_select_blocks_geometry_and_legacy_pools(self):
        from research.algernon.circuit_mirror.lotus_bridge.capture import select_blocks, select_hidden
        # prompt 0..2, start token 3, latents 4..9 (3 blocks of 2), end token 10
        h = torch.randn(1, 11, 5)
        logits = torch.randn(1, 11, 9)
        kw = dict(first_latent=4, last_latent=9, loop_start=3, loop_end=10, c_thought=2)
        a = select_blocks(h[:, 3:10], logits[:, 3:10], **kw)
        b = select_blocks(h[:, :10], logits[:, :10], **kw)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        blocks, static, readout = a
        self.assertEqual(tuple(blocks.shape), (3, 2, 5))
        self.assertTrue(torch.equal(blocks[1, 0], h[0, 7]))
        self.assertTrue(torch.allclose(blocks[2, 1], h[0, 8:10].mean(0)))
        self.assertTrue(torch.equal(static, h[0, 3]))
        self.assertTrue(torch.equal(readout, logits[0, 3:10].argmax(-1).to(torch.int32)))
        legacy = select_hidden(h[:, 3:10], first_latent=4, last_latent=9, loop_start=3, loop_end=10, c_thought=2)
        self.assertTrue(torch.allclose(legacy, blocks[-1], atol=1e-6))
        with self.assertRaises(ValueError):
            select_blocks(h[:, 3:10], None, first_latent=4, last_latent=8, loop_start=3, loop_end=9, c_thought=2)

    def test_capture_case_with_stub(self):
        from research.algernon.circuit_mirror.lotus_bridge.capture import capture_case
        module = types.SimpleNamespace()

        def efficient(hidden):
            return types.SimpleNamespace(last_hidden_state=hidden,
                                         logits=torch.zeros((*hidden.shape[:2], 13)).index_fill_(2, torch.tensor([4]), 1.0))
        module.efficient_forward = efficient

        class Stub:
            start_latent_id, latent_token_id, end_latent_id = 10, 11, 12
            base_causallm = torch.nn.Linear(2, 2)

            def forward(self, input_ids, attention_mask, labels, position_ids, n_looped_iters=0):
                ix = (input_ids[0] == 11).nonzero().flatten()
                start, end = int(ix[0]) - 1, int(ix[-1]) + 1
                hidden = torch.arange((end - start) * 3, dtype=torch.float32).reshape(1, end - start, 3)
                for k in range(n_looped_iters + 1):
                    out = module.efficient_forward(hidden + k * 100.0)
                    self.assert_suppressed = out.logits.shape[-1] == 1
                return out

        tok = types.SimpleNamespace(encode=lambda q, add_special_tokens=True: [1, 2, 3])
        res, meta = capture_case(Stub(), module, tok, "q", loops=3, c_thought=2)
        self.assertEqual(res["blocks"].shape, (4, 3, 2, 3))
        self.assertEqual(res["readout"].shape, (4, 7))
        self.assertTrue((res["readout"] == 4).all())
        np.testing.assert_allclose(res["blocks"][2, 0, 0] - res["blocks"][0, 0, 0], 200.0)
        np.testing.assert_allclose(res["static"][1] - res["static"][0], 100.0)
        self.assertIs(module.efficient_forward, efficient)
        self.assertEqual(meta["sequence_tokens"], 3 + 1 + 6 + 1)


def _fixture(roster, cf, kind, hidden=72, seed=0):
    """Manufactured captures. recurrent: block b holds trace[b+1] once pass >= b+1.
    static: every block holds only the prompt (one-hot map and start) plus a
    position code, never the state."""
    rng = np.random.default_rng(seed)

    def make(cases):
        n, steps = len(cases), 16
        blocks = rng.normal(0, 0.3, (n, steps + 1, steps, 2, hidden)).astype(np.float32)
        static = np.repeat(rng.normal(0, 0.3, (n, 1, hidden)).astype(np.float32), steps + 1, 1)
        for i, r in enumerate(cases):
            if kind == "recurrent":
                for p in range(steps + 1):
                    for b in range(min(p, steps)):
                        blocks[i, p, b, :, r["trace"][b + 1]] += 4.0
            else:
                code = np.zeros(hidden, dtype=np.float32)
                for src, dst in enumerate(r["successors"]):
                    code[6 * src + dst] = 2.0
                code[36 + r["initial"]] = 2.0
                static[i] += code
                blocks[i] += code
                for b in range(steps):
                    blocks[i, :, b, :, 42 + b] += 2.0
        return {"blocks": blocks, "static": static}
    return make(roster["cases"]), make(cf["cases"])


@unittest.skipIf(torch is None, "optional torch/numpy unavailable")
class AnalysisGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roster = build_roster(train=40, calibration=12, test_per_family=10)
        cls.cf = build_counterfactual(cls.roster)

    def _run(self, kind):
        from research.algernon.circuit_mirror.lotus_bridge.analysis import analyze
        cap, cf_cap = _fixture(self.roster, self.cf, kind)
        return analyze(self.roster, self.cf, cap, cf_cap, seed=3)

    def test_recurrent_fixture_passes_gate(self):
        summary, records = self._run("recurrent")
        res, gate = summary["results"], summary["gate"]
        self.assertTrue(gate["passed"], gate)
        self.assertEqual(gate["classification"], "recurrence_specific_state")
        self.assertGreater(res[f"view:{gate['primary_view']}"]["all_test"]["heldout_steps_9_16_accuracy"], 0.95)
        self.assertLess(res["view:pass0"]["long_cycle"]["heldout_steps_9_16_accuracy"], 0.5)
        self.assertGreater(res["counterfactual:relabelled_truth"]["all_test"]["heldout_steps_9_16_accuracy"], 0.95)
        self.assertEqual(res["baseline:exact_solver"]["all_test"]["exact_trace_valid_full_rate"], 1.0)
        self.assertTrue(all(len(r["decoded"]) == 17 for r in records if "decoded" in r))

    def test_static_fixture_fails_gate(self):
        summary, _ = self._run("static")
        gate = summary["gate"]
        self.assertFalse(gate["passed"], gate)
        self.assertNotEqual(gate["classification"], "recurrence_specific_state")
        self.assertLess(summary["results"][f"view:{gate['primary_view']}"]["long_cycle"]["heldout_steps_9_16_accuracy"], 0.5)
        probes = summary["results"]["input_encoding"]["static_token"]
        self.assertGreater(probes["start"]["heldout_accuracy"], probes["start"]["train_majority_accuracy"] + 0.4)
        self.assertGreater(probes["successor[0]"]["heldout_accuracy"], probes["successor[0]"]["train_majority_accuracy"])


@unittest.skipUnless(UPSTREAM and torch is not None and _transformers_446(),
                     "needs ALGERNON_LOTUS_UPSTREAM and transformers 4.46.2")
class UpstreamWrapperParityTests(unittest.TestCase):
    def test_parity_suite_on_tiny_random_llama(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from research.algernon.circuit_mirror.lotus_bridge.capture import capture_case, import_lotus_module
        from research.algernon.circuit_mirror.lotus_bridge.gpu_run import parity_suite, weights_fingerprint
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                          tie_word_embeddings=True)
        base = LlamaForCausalLM(cfg).eval()
        module = import_lotus_module(Path(UPSTREAM))
        model = module.Lotus(base, latent_token_id=39, start_latent_id=37, end_latent_id=38,
                             eos_token_id=2, pad_token_id=2, c_thought=3).eval()
        tok = types.SimpleNamespace(encode=lambda q, add_special_tokens=True: [1] + [5 + (ord(c) % 30) for c in q[:20]])
        before = weights_fingerprint(base)
        report = parity_suite(model, module, tok, prompt_for(build_roster(train=1, calibration=1,
                              test_per_family=1)["cases"][0]), loops=4, c_thought=3)
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["static_token_max_drift_across_passes"], 0.0)
        res, _ = capture_case(model, module, tok, "abc", loops=4, c_thought=3)
        self.assertEqual(res["blocks"].shape, (5, 4, 2, 32))
        self.assertEqual(weights_fingerprint(base), before)


if __name__ == "__main__":
    unittest.main()
