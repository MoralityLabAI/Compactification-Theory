"""CPU tests for Algernon-02C (LOTUS recipe on the pointer task). Fixtures only.

The trainer tests build a tiny random GPT-2 with the real GPT-2 tokenizer and run
the pinned upstream LOTUS code; they need ALGERNON_LOTUS_UPSTREAM, transformers
4.46.2 and the pinned GPT-2 snapshot in the Hugging Face cache. A pass shows the
loop, checkpoints and resume work; it says nothing about what the real run learns.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # CI installs no torch
    torch = None

from research.algernon.circuit_mirror.lotus_bridge.counterfactual import build_counterfactual
from research.algernon.circuit_mirror.lotus_bridge.pointer_cot import build, step_text
from research.algernon.circuit_mirror.lotus_bridge.roster import build_roster, prompt_for

UPSTREAM = os.environ.get("ALGERNON_LOTUS_UPSTREAM")


def _gpt2_snapshot():
    try:
        from huggingface_hub import snapshot_download
        from research.algernon.circuit_mirror.lotus_bridge.train_lotus import GPT2_REVISION
        return snapshot_download("openai-community/gpt2", revision=GPT2_REVISION, local_files_only=True)
    except Exception:
        return None


def _small_data(n_train=48, n_val=16):
    roster = build_roster(train=12, calibration=6, test_per_family=6)
    cf = build_counterfactual(roster)
    return roster, cf, build(roster=roster, counterfactual=cf, n_train=n_train, n_val=n_val)


class PointerCotTests(unittest.TestCase):
    def test_rows_exclude_the_frozen_sets_and_match_the_capture_prompt(self):
        roster, cf, (train, val, meta) = _small_data()
        banned = {(tuple(c["successors"]), c["initial"]) for c in roster["cases"] + cf["cases"]}
        keys = [(tuple(r["successors"]), r["initial"]) for r in train + val]
        self.assertEqual(len(keys), len(set(keys)), "train and validation pairs are distinct")
        self.assertFalse(banned & set(keys))
        self.assertEqual([r["depth"] for r in train[:9]], [1, 2, 3, 4, 5, 6, 7, 8, 1])
        for r in train + val:
            self.assertEqual(r["question"] + "\n", prompt_for(r))
            self.assertEqual(len(r["steps"]), r["depth"])
            self.assertEqual(r["steps"][-1], step_text(r["trace"][-2], r["trace"][-1]))
            self.assertEqual(r["answer"], str(r["trace"][-1]))
            self.assertEqual(r["trace"][-1], r["trace"][0] if r["depth"] == 0 else r["successors"][r["trace"][-2]])
        self.assertEqual(meta["parameters"]["roster_sha256"], roster["roster_sha256"])


@unittest.skipIf(torch is None, "optional numpy/torch unavailable")
class DepthEvalTests(unittest.TestCase):
    def test_rows_ask_for_d_steps_and_the_summary_scores_cases(self):
        from research.algernon.circuit_mirror.lotus_bridge.depth_eval import depth_rows, summarize
        roster = build_roster(train=12, calibration=6, test_per_family=6)
        rows = depth_rows(roster)
        self.assertEqual(len(rows), (6 + 12) * 16)
        cases = {c["id"]: c for c in roster["cases"]}
        for r in rows:
            c = cases[r["case"]]
            self.assertEqual(r["answer"], str(c["trace"][r["depth"]]))
            self.assertIn(f"exactly {r['depth']} times", r["question"])
            self.assertEqual(r["question"] + "\n", prompt_for({**c, "trace": c["trace"][:r["depth"] + 1]}))
        s = summarize([{**r, "correct": r["depth"] <= 8 or r["family"] == "sink"} for r in rows])
        self.assertEqual((s["heldout_9_16"]["accuracy"], s["heldout_9_16"]["cases"]), (0.5, 12))
        self.assertEqual(s["long_cycle_9_16"]["accuracy"], 0.0)
        self.assertEqual(s["accuracy_by_depth"]["permutation"][16], 0.0)


@unittest.skipIf(torch is None or not UPSTREAM or _gpt2_snapshot() is None,
                 "needs torch, ALGERNON_LOTUS_UPSTREAM and the pinned GPT-2 snapshot")
class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.base = root / "tiny-gpt2"
        torch.manual_seed(0)
        GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=512)).save_pretrained(cls.base)
        AutoTokenizer.from_pretrained(_gpt2_snapshot()).save_pretrained(cls.base)
        roster, cf, (train, val, meta) = _small_data(n_train=48, n_val=16)
        cls.data = root / "data"
        cls.data.mkdir()
        for name, obj in (("pointer_train.json", train), ("pointer_val.json", val), ("pointer_data.json", meta)):
            (cls.data / name).write_text(json.dumps(obj))
        cls.val = val
        torch.cuda.synchronize = lambda *a, **k: None  # upstream Lotus.generate times itself with CUDA syncs

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_trainer(self, out, *extra):
        from research.algernon.circuit_mirror.lotus_bridge import train_lotus
        argv = ["train_lotus", "--upstream", UPSTREAM, "--base", str(self.base), "--data-dir", str(self.data),
                "--out", str(out), "--device", "cpu", "--batch", "8", "--val-every-epoch", "4", *extra]
        old = sys.argv
        sys.argv = argv
        try:
            train_lotus.main()
        finally:
            sys.argv = old
        return json.loads((Path(out) / "train_manifest.json").read_text())

    def test_upstream_tokenization_equals_the_capture_input(self):
        from research.algernon.circuit_mirror.lotus_bridge.train_lotus import (import_upstream_dataset,
                                                                               single_process_map, tokenizer_and_base)
        single_process_map()
        ds = import_upstream_dataset(Path(UPSTREAM))
        tok, _ = tokenizer_and_base(self.base)
        data = ds.get_dataset(str(self.data / "pointer_train.json"), tok)  # upstream asserts its own round trip
        rows = json.loads((self.data / "pointer_train.json").read_text())
        for i in range(8):
            self.assertEqual(data[i]["question_tokenized"], tok.encode(prompt_for(rows[i]), add_special_tokens=True))
            self.assertTrue(all(len(s) <= 6 for s in data[i]["steps_tokenized"]), "a step fits one c_thought=6 block")

    def test_cot_then_both_arms_and_a_resumed_run_matches_a_straight_run(self):
        root = Path(self.tmp.name)
        cot = self.run_trainer(root / "cot", "--phase", "cot", "--epochs", "2")
        self.assertEqual(cot["status"], "complete")
        init = root / "cot" / "final.pt"
        lotus = ["--phase", "lotus", "--init", str(init), "--epochs", "3", "--max-latent-stage", "2"]
        straight = self.run_trainer(root / "b_straight", *lotus, "--after-loop", "false")
        self.assertEqual(straight["status"], "complete")
        self.assertEqual([e["stage"] for e in straight["epochs_done"]], [0, 1, 2])
        part = self.run_trainer(root / "b_resumed", *lotus, "--after-loop", "false", "--max-steps", "8")
        self.assertEqual(part["status"], "running")
        resumed = self.run_trainer(root / "b_resumed", *lotus, "--after-loop", "false")
        self.assertEqual((resumed["status"], resumed["attempts"]), ("complete", 2))
        a = torch.load(root / "b_straight" / "final.pt", weights_only=True)
        b = torch.load(root / "b_resumed" / "final.pt", weights_only=True)
        self.assertEqual(a.keys(), b.keys())
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a), "resume mid-epoch is bitwise equal to a straight run")
        arm_a = self.run_trainer(root / "a", *lotus, "--after-loop", "true", "--micro-batch", "4")
        self.assertEqual((arm_a["status"], arm_a["after_loop"]), ("complete", True))
        self.assertFalse(all(torch.equal(a[k], v) for k, v in torch.load(root / "a" / "final.pt", weights_only=True).items()),
                         "the two arms train different weights")

    def test_trained_checkpoints_load_and_answer(self):
        from research.algernon.circuit_mirror.lotus_bridge.gpu_run import upstream_extract_answer
        from research.algernon.circuit_mirror.lotus_bridge.train_lotus import answers, load_trained
        root = Path(self.tmp.name)
        if not (root / "b_straight" / "final.pt").exists():
            self.test_cot_then_both_arms_and_a_resumed_run_matches_a_straight_run()
        extract = upstream_extract_answer(Path(UPSTREAM))
        model, _, tok = load_trained(Path(UPSTREAM), self.base, root / "b_straight" / "final.pt", device="cpu")
        out = answers(model, tok, self.val[:2], kind="lotus", blocks=4, loops=4, c_thought=6, extract=extract)
        self.assertEqual(len(out), 2)
        cot, _, ctok = load_trained(Path(UPSTREAM), self.base, root / "cot" / "final.pt", device="cpu")
        self.assertEqual(len(answers(cot, ctok, self.val[:2], kind="cot", max_new_tokens=8, extract=extract)), 2)
