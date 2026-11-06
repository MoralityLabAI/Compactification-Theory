"""CPU tests for Algernon-02E (depth beyond 16, causal slot patching). Fixtures only.

The model tests need ALGERNON_LOTUS_UPSTREAM, transformers 4.46.2 and the pinned GPT-2
snapshot. A pass shows the hook patches exactly the slots at the chosen loop and that a
self-patch changes nothing, not what the real D1 weights do.
"""
import json
import random
import unittest
from pathlib import Path

from research.algernon.circuit_mirror.lotus_bridge.slot_patch import choose_donor, iterate
from research.algernon.circuit_mirror.tests.test_lotus_02c import UPSTREAM, _gpt2_snapshot, torch
from research.algernon.circuit_mirror.tests import test_lotus_02d

ROSTER = Path(__file__).resolve().parents[1] / "lotus_bridge" / "frozen_roster_6x16.json"


class PureTests(unittest.TestCase):
    def test_iterate_matches_the_roster_traces(self):
        for c in json.loads(ROSTER.read_text(encoding="utf-8"))["cases"]:
            self.assertEqual([iterate(c["successors"], c["initial"], d) for d in range(17)], c["trace"])

    def test_donor_predictions_are_three_different_states(self):
        cases = [c for c in json.loads(ROSTER.read_text(encoding="utf-8"))["cases"] if c["split"] != "train"]
        a = next(c for c in cases if c["family"] == "long_cycle")
        for k in (4, 8, 12):
            b, pred = choose_donor(a, k, 16, cases, random.Random(k))
            self.assertNotEqual(b["id"], a["id"])
            self.assertEqual(len(set(pred.values())), 3)
            self.assertEqual(pred["follow"], iterate(a["successors"], b["trace"][k], 16 - k))


@unittest.skipIf(torch is None or not UPSTREAM or _gpt2_snapshot() is None,
                 "needs torch, ALGERNON_LOTUS_UPSTREAM and the pinned GPT-2 snapshot")
class SlotHookTests(unittest.TestCase):
    run_trainer = test_lotus_02d.LoopTrainerTests.run_trainer

    @classmethod
    def setUpClass(cls):
        test_lotus_02d.LoopTrainerTests.setUpClass.__func__(cls)
        from research.algernon.circuit_mirror.lotus_bridge.gpu_run import upstream_extract_answer
        from research.algernon.circuit_mirror.lotus_bridge.train_lotus import load_trained
        root = Path(cls.tmp.name)
        cls.run_trainer(cls, root / "d1", "--intermediate-loss-weight", "0.05", "--epochs", "1")
        cls.model, cls.lm, cls.tok = load_trained(Path(UPSTREAM), cls.base, root / "d1" / "final.pt", device="cpu")
        cls.extract = staticmethod(upstream_extract_answer(Path(UPSTREAM)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, q, **kw):
        from research.algernon.circuit_mirror.lotus_bridge.slot_patch import run_case
        return run_case(self.model, self.lm, self.tok, q, loops=5, extract=self.extract, record=True, **kw)

    def test_layout_matches_the_depth_curve(self):
        from research.algernon.circuit_mirror.lotus_bridge.pointer_loop import prompt_nocount
        from research.algernon.circuit_mirror.lotus_bridge.train_lotus import answers
        row = {"successors": [1, 2, 0, 3, 4, 5], "initial": 0}
        q = prompt_nocount(row).rstrip("\n")
        res, states = self.run_case(q)
        ref = answers(self.model, self.tok, [{"question": q, "answer": "0"}], kind="lotus", blocks=1, loops=5,
                      c_thought=6, max_new_tokens=16, extract=self.extract)[0]
        self.assertEqual(res["text"], ref["answer_text"])
        self.assertEqual((len(states), *states[0].shape[:1]), (6, 6))

    def test_self_patch_changes_nothing_and_a_donor_patch_lands_at_loop_k(self):
        from research.algernon.circuit_mirror.lotus_bridge.pointer_loop import prompt_nocount
        qa = prompt_nocount({"successors": [1, 2, 0, 3, 4, 5], "initial": 0}).rstrip("\n")
        qb = prompt_nocount({"successors": [5, 4, 3, 2, 1, 0], "initial": 3}).rstrip("\n")
        a, sa = self.run_case(qa)
        _, sb = self.run_case(qb)
        same, s_same = self.run_case(qa, patch=(2, sa[2]))
        self.assertEqual(same["text"], a["text"])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(sa, s_same)))
        _, s_pat = self.run_case(qa, patch=(2, sb[2]))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(sa[:2], s_pat[:2])), "loops before k untouched")
        self.assertTrue(torch.equal(s_pat[2], sb[2]), "the slots after loop k are the donor's")
        self.assertFalse(torch.equal(s_pat[3], sa[3]), "the next loop reads the patched slots")


if __name__ == "__main__":
    unittest.main()
