"""CPU tests for Algernon-02D (loop count as the only depth signal). Fixtures only.

The trainer test needs ALGERNON_LOTUS_UPSTREAM, transformers 4.46.2 and the pinned
GPT-2 snapshot; a pass shows the loop phase, its batches and resume work, not what
the real run learns.
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

from research.algernon.circuit_mirror.lotus_bridge.pointer_loop import (PATCH_FROM, PATCH_TO, convert,
                                                                       prompt_nocount)
from research.algernon.circuit_mirror.lotus_bridge.roster import build_roster, prompt_for
from research.algernon.circuit_mirror.tests.test_lotus_02c import UPSTREAM, _gpt2_snapshot, _small_data, torch


class PromptTests(unittest.TestCase):
    def test_nocount_prompt_is_the_count_prompt_without_the_count(self):
        roster = build_roster(train=12, calibration=6, test_per_family=6)
        for c in roster["cases"][:20]:
            q = prompt_nocount(c)
            self.assertNotIn("times", q)
            self.assertNotIn("steps", q)
            self.assertTrue(q.endswith("follow the successor once per loop. Which state is reached?\n"))
            self.assertEqual(q.split("Starting at")[0], prompt_for(c).split("Starting at")[0])
            self.assertEqual(re.findall(r"Starting at state (\d+)", q), [str(c["initial"])])
        rows = convert([{"successors": [1, 0], "initial": 0, "trace": [0, 1], "question": "x", "depth": 1}])
        self.assertEqual(rows[0]["question"] + "\n", prompt_nocount(rows[0]))

    @unittest.skipIf(not UPSTREAM, "needs ALGERNON_LOTUS_UPSTREAM")
    def test_the_patch_changes_exactly_one_line(self):
        from research.algernon.circuit_mirror.lotus_bridge.pointer_loop import patched_lotus_module, patched_source
        orig = (Path(UPSTREAM) / "scripts" / "lotus.py").read_text(encoding="utf-8")
        new = patched_source(Path(UPSTREAM))
        diff = [(a, b) for a, b in zip(orig.splitlines(), new.splitlines()) if a != b]
        self.assertEqual(diff, [(PATCH_FROM.rstrip("\n"), PATCH_TO.rstrip("\n"))])
        module, sha = patched_lotus_module(Path(UPSTREAM))
        self.assertTrue(hasattr(module, "Lotus") and len(sha) == 64)


@unittest.skipIf(torch is None or not UPSTREAM or _gpt2_snapshot() is None,
                 "needs torch, ALGERNON_LOTUS_UPSTREAM and the pinned GPT-2 snapshot")
class LoopTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.base = root / "tiny-gpt2"
        torch.manual_seed(0)
        GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=512)).save_pretrained(cls.base)
        AutoTokenizer.from_pretrained(_gpt2_snapshot()).save_pretrained(cls.base)
        _, _, (train, val, meta) = _small_data(n_train=96, n_val=16)
        cls.data = root / "data"
        cls.data.mkdir()
        for name, obj in (("pointer_train.json", convert(train)), ("pointer_val.json", convert(val)),
                          ("pointer_data.json", meta)):
            (cls.data / name).write_text(json.dumps(obj))
        cls.init = root / "init.pt"
        torch.save(GPT2LMHeadModel.from_pretrained(cls.base).state_dict(), cls.init)
        torch.cuda.synchronize = lambda *a, **k: None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_trainer(self, out, *extra):
        from research.algernon.circuit_mirror.lotus_bridge import train_lotus
        argv = ["train_lotus", "--upstream", UPSTREAM, "--base", str(self.base), "--data-dir", str(self.data),
                "--out", str(out), "--device", "cpu", "--batch", "4", "--micro-batch", "2", "--val-every-epoch", "8",
                "--phase", "loop", "--init", str(self.init), "--epochs", "2", *extra]
        old, sys.argv = sys.argv, argv
        try:
            train_lotus.main()
        finally:
            sys.argv = old
        return json.loads((Path(out) / "train_manifest.json").read_text())

    def test_loop_phase_trains_resumes_bitwise_and_the_arms_differ(self):
        root = Path(self.tmp.name)
        straight = self.run_trainer(root / "d1", "--intermediate-loss-weight", "0.05")
        self.assertEqual((straight["status"], straight["phase"], straight["fixed_slots"]), ("complete", "loop", 6))
        self.assertEqual(len(straight["patched_lotus_sha256"]), 64)
        self.assertEqual([e["stage"] for e in straight["epochs_done"]], ["loop", "loop"])
        logged = [json.loads(l) for l in (root / "d1" / "train_log.jsonl").read_text().splitlines()]
        self.assertTrue({r["stage"] for r in logged} <= set(range(1, 9)), "each logged batch ran its own depth in loops")
        part = self.run_trainer(root / "d1_resumed", "--intermediate-loss-weight", "0.05", "--max-steps", "30")
        self.assertEqual(part["status"], "running")
        resumed = self.run_trainer(root / "d1_resumed", "--intermediate-loss-weight", "0.05")
        self.assertEqual((resumed["status"], resumed["attempts"]), ("complete", 2))
        a = torch.load(root / "d1" / "final.pt", weights_only=True)
        b = torch.load(root / "d1_resumed" / "final.pt", weights_only=True)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a), "a resumed loop run equals a straight one")
        d0 = self.run_trainer(root / "d0", "--intermediate-loss-weight", "0")
        c = torch.load(root / "d0" / "final.pt", weights_only=True)
        self.assertEqual(d0["intermediate_loss_weight"], 0.0)
        self.assertFalse(all(torch.equal(a[k], c[k]) for k in a), "per-loop supervision changes the weights")

    def test_fixed_slot_capture_has_one_block_per_pass(self):
        from research.algernon.circuit_mirror.lotus_bridge.capture import capture_case, import_lotus_module
        from research.algernon.circuit_mirror.lotus_bridge.train_lotus import load_trained
        root = Path(self.tmp.name)
        self.run_trainer(root / "for_capture", "--intermediate-loss-weight", "0.05", "--epochs", "1")
        model, module, tok = load_trained(Path(UPSTREAM), self.base, root / "for_capture" / "final.pt", device="cpu")
        q = prompt_nocount({"successors": [1, 2, 0, 3, 4, 5], "initial": 0})
        res, meta = capture_case(model, module, tok, q, loops=5, c_thought=6, slots=6)
        self.assertEqual(res["blocks"].shape[:3], (6, 1, 2))
        self.assertEqual(meta["latent_tokens"], 6)
