"""CPU tests for Algernon-03 (H-A3, constraint-guided correction under slot noise). Fixtures only.

The model tests need ALGERNON_LOTUS_UPSTREAM, transformers 4.46.2 and the pinned GPT-2
snapshot. A pass shows the arms are wired as registered (zero-initialised correctors change
nothing, noise lands at loop j only, gradients reach the corrector through F), not what the
real D1 weights do.
"""
import json
import unittest
from pathlib import Path

from research.algernon.circuit_mirror.tests import test_lotus_02d
from research.algernon.circuit_mirror.tests.test_lotus_02c import UPSTREAM, _gpt2_snapshot, torch

ROSTER = Path(__file__).resolve().parents[1] / "lotus_bridge" / "frozen_roster_6x16.json"


class PureTests(unittest.TestCase):
    def test_exact_checker_accepts_the_true_trace_only(self):
        from research.algernon.circuit_mirror.lotus_bridge.correct import trace_valid
        c = next(c for c in json.loads(ROSTER.read_text(encoding="utf-8"))["cases"] if c["family"] == "long_cycle")
        self.assertTrue(trace_valid(c, c["trace"]))
        bad = list(c["trace"])
        bad[7] = (bad[7] + 1) % 6
        self.assertFalse(trace_valid(c, bad))

    def test_generated_cases_avoid_the_roster_and_repeat(self):
        from research.algernon.circuit_mirror.lotus_bridge.correct import generated
        roster = {(tuple(c["successors"]), c["initial"]) for c in json.loads(ROSTER.read_text(encoding="utf-8"))["cases"]}
        a, b = generated(20261107, 40, 8), generated(20261107, 40, 8)
        self.assertEqual(a, b)
        self.assertFalse({(tuple(r["successors"]), r["initial"]) for r in a} & roster)
        self.assertNotEqual(a, generated(20261108, 40, 8))

    @unittest.skipIf(torch is None, "needs torch")
    def test_noise_is_seeded_per_id_and_correctors_start_at_zero(self):
        from research.algernon.circuit_mirror.lotus_bridge.correct import corrector, eps_for, features
        self.assertTrue(torch.equal(eps_for("a:j4", 8), eps_for("a:j4", 8)))
        self.assertFalse(torch.equal(eps_for("test/long_cycle/1:primary:j4:s0", 8),
                                     eps_for("test/long_cycle/2:primary:j4:s0", 8)))
        e, p = torch.rand(6), torch.rand(6)
        for arm, width in (("C", 6), ("I", 12), ("N", 6)):
            x = features(arm, e, p)
            self.assertEqual(x.shape[0], width)
            self.assertFalse(corrector(arm, 8)(x).any())
        self.assertIsNone(corrector("P", 8))

    @unittest.skipIf(torch is None, "needs torch")
    def test_bias_free_correctors_map_zero_to_zero(self):
        """H-A3c: without a bias, a satisfied constraint (e = p, so C's input is 0) gives no correction."""
        from research.algernon.circuit_mirror.lotus_bridge.correct import corrector, features
        for arm, width in (("C", 6), ("I", 12), ("N", 6)):
            G = corrector(arm, 8, bias=False)
            self.assertIsNone(G.bias)
            self.assertEqual(sum(p.numel() for p in G.parameters()), width * 6 * 8)
            with torch.no_grad():
                G.weight.normal_()
            self.assertFalse(G(torch.zeros(width)).any())
        p = torch.softmax(torch.randn(6), -1)
        with torch.no_grad():
            G = corrector("C", 8, bias=False)
            G.weight.normal_()
            self.assertFalse(G(features("C", p, p)).any())
        self.assertIsNotNone(corrector("C", 8).bias, "the default keeps H-A3's and H-A3b's bias")


@unittest.skipIf(torch is None or not UPSTREAM or _gpt2_snapshot() is None,
                 "needs torch, ALGERNON_LOTUS_UPSTREAM and the pinned GPT-2 snapshot")
class ArmTests(unittest.TestCase):
    run_trainer = test_lotus_02d.LoopTrainerTests.run_trainer

    @classmethod
    def setUpClass(cls):
        test_lotus_02d.LoopTrainerTests.setUpClass.__func__(cls)
        from research.algernon.circuit_mirror.lotus_bridge.correct import digit_ids
        from research.algernon.circuit_mirror.lotus_bridge.gpu_run import upstream_extract_answer
        from research.algernon.circuit_mirror.lotus_bridge.train_lotus import load_trained
        root = Path(cls.tmp.name)
        cls.run_trainer(cls, root / "d1", "--intermediate-loss-weight", "0.05", "--epochs", "1")
        cls.model, cls.lm, cls.tok = load_trained(Path(UPSTREAM), cls.base, root / "d1" / "final.pt", device="cpu")
        for p in cls.model.parameters():
            p.requires_grad_(False)
        cls.digits = digit_ids(cls.tok)
        cls.extract = staticmethod(upstream_extract_answer(Path(UPSTREAM)))
        cls.row = {"successors": [1, 2, 0, 3, 4, 5], "initial": 0}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def answer(self, **kw):
        from research.algernon.circuit_mirror.lotus_bridge.correct import answer
        return answer(self.model, self.lm, self.tok, self.digits, self.row, loops=5, extract=self.extract, **kw)

    def test_plain_arm_matches_the_unhooked_run_and_zero_correctors_change_nothing(self):
        from research.algernon.circuit_mirror.lotus_bridge.correct import corrector
        from research.algernon.circuit_mirror.lotus_bridge.pointer_loop import prompt_nocount
        from research.algernon.circuit_mirror.lotus_bridge.slot_patch import run_case
        ref, _ = run_case(self.model, self.lm, self.tok, prompt_nocount(self.row).rstrip("\n"), loops=5,
                          extract=self.extract)
        _, text, trace = self.answer()
        self.assertEqual(text, ref["text"])
        self.assertEqual(len(trace), 6)
        for arm in ("C", "I", "N"):
            self.assertEqual(self.answer(arm=arm, G=corrector(arm, 32))[1], text)

    def test_noise_lands_at_loop_j_only(self):
        from research.algernon.circuit_mirror.lotus_bridge.correct import Recurrence, eps_for, layout
        from research.algernon.circuit_mirror.lotus_bridge.pointer_loop import prompt_nocount
        ids, first, last = layout(self.model, self.tok, prompt_nocount(self.row).rstrip("\n"))
        x = torch.tensor([ids])
        runs = []
        for noise in (None, {"j": 3, "sigma": 4.0, "eps": eps_for("t", 32)}):
            rec = Recurrence(self.model, self.lm, self.digits, self.row["successors"], 0, noise=noise)
            with torch.inference_mode(), rec.hooked(first, last):
                self.model.generate(x, torch.ones_like(x), max_new_tokens=4, n_looped_iters=5)
            runs.append(rec.probs)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(runs[0][:2], runs[1][:2])), "loops before j untouched")
        self.assertFalse(torch.equal(runs[0][2], runs[1][2]), "the noise changes loop j")

    def test_output_scale_multiplies_the_correction(self):
        """H-A3b: kappa = 0 reproduces the plain run; a nonzero kappa moves pi after the first loop."""
        from research.algernon.circuit_mirror.lotus_bridge.correct import Recurrence, corrector, layout
        from research.algernon.circuit_mirror.lotus_bridge.pointer_loop import prompt_nocount
        ids, first, last = layout(self.model, self.tok, prompt_nocount(self.row).rstrip("\n"))
        x = torch.tensor([ids])
        G = corrector("I", 32)
        with torch.no_grad():
            G.weight.copy_(torch.randn(G.weight.shape, generator=torch.Generator().manual_seed(0)))
        runs = []
        for arm, g, scale in (("P", None, 1.0), ("I", G, 0.0), ("I", G, 5.0)):
            rec = Recurrence(self.model, self.lm, self.digits, self.row["successors"], 0, arm=arm, G=g, scale=scale)
            with torch.inference_mode(), rec.hooked(first, last):
                self.model.generate(x, torch.ones_like(x), max_new_tokens=4, n_looped_iters=5)
            runs.append(rec.probs)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(runs[0], runs[1])))
        self.assertFalse(torch.equal(runs[0][1], runs[2][1]))

    def test_slot_rms_is_recorded_per_loop_and_deterministic(self):
        from research.algernon.circuit_mirror.lotus_bridge.correct import slot_rms
        a = slot_rms(self.model, self.lm, self.tok, self.digits, self.row, loops=5)
        self.assertEqual(len(a), 5)
        self.assertTrue(all(v > 0 for v in a))
        self.assertEqual(a, slot_rms(self.model, self.lm, self.tok, self.digits, self.row, loops=5))

    def test_the_local_loss_reaches_the_corrector_through_f(self):
        from research.algernon.circuit_mirror.lotus_bridge.correct import corrector, eps_for, train_step_loss
        G = corrector("C", 32)
        loss = train_step_loss(self.model, self.lm, self.tok, self.digits, self.row, 2, arm="C", G=G,
                               noise={"j": 2, "sigma": 2.0, "eps": eps_for("g", 32)})
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(G.weight.grad.abs().sum() > 0)
        self.assertFalse(any(p.grad is not None for p in self.model.parameters()), "F stays frozen")


if __name__ == "__main__":
    unittest.main()
