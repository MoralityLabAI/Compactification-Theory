"""Executable SIML-01 runner checks, including failure paths and trusted replay."""
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
import json
import tempfile
import unittest

from microlearner import (GRAPH, GRAPH_SHA, MicroLearner, all_bit_states,
                          apply_bit_action, diagnose, digest, exact_accuracy,
                          fit_bit_transitions, fit_shear_transitions,
                          heuristic_proposal, sample, WEIGHTS)
from promotion import Broker, ProtocolError, run_demo


class MicroLearnerTests(unittest.TestCase):
    def test_graph_weights_and_hash(self):
        self.assertEqual(sum(F(str(v)) for v in WEIGHTS.values()), F(1))
        self.assertEqual(digest(GRAPH), GRAPH_SHA)
        self.assertTrue(all(e["kind"].endswith("_hypothesis") for e in GRAPH["edges"]))

    def test_teacher_one_step_full_code(self):
        table = fit_bit_transitions(3)
        for x in all_bit_states():
            for a in ("identity", "flip_y", "flip_b", "reveal_b", "reveal_c"):
                self.assertEqual(table[(x, a)], apply_bit_action(x, a))

    def test_task_only_cannot_retain_hidden_b(self):
        baseline = MicroLearner()
        self.assertEqual(exact_accuracy(baseline, "T0"), 1)
        self.assertEqual(exact_accuracy(baseline, "T1"), 0.5)
        self.assertTrue(diagnose(baseline)["b_future_collision"])

    def test_first_b_hill_climb_generalizes_to_compositions(self):
        b = MicroLearner().update("retain_b")
        self.assertEqual(exact_accuracy(b, "T1"), 1)
        self.assertEqual(exact_accuracy(b, "T0"), 1)
        self.assertFalse(diagnose(b)["b_future_collision"])

    def test_full_state_is_closed_for_all_five_actions(self):
        b = MicroLearner(3)
        self.assertEqual(len(fit_bit_transitions(3)), 8*5)
        self.assertEqual(exact_accuracy(b, "T1"), 1)

    def test_shear_capacity_and_exact_recoverability(self):
        baseline = MicroLearner(2, False)
        expanded = MicroLearner(2, True)
        self.assertEqual(exact_accuracy(baseline, "T4"), 1/31)
        self.assertEqual(exact_accuracy(expanded, "T4"), 1)
        self.assertEqual(len(fit_shear_transitions(False)), 31)
        self.assertEqual(len(fit_shear_transitions(True)), 961)
        self.assertEqual((31-1).bit_length(), 5)
        self.assertEqual((961-1).bit_length(), 10)

    def test_diagnostic_proposals_are_pinned(self):
        base = MicroLearner()
        p = heuristic_proposal(base)
        self.assertEqual((p["operator"], p["target"]), ("retain_b", "T1"))
        self.assertEqual(p["diagnostics"]["checkpoint"], base.checkpoint)
        p2 = heuristic_proposal(base.update("retain_b"))
        self.assertEqual((p2["operator"], p2["target"]), ("install_shear", "T4"))
        self.assertIsNone(heuristic_proposal(base.update("retain_b").update("install_shear")))

    def test_inapplicable_updates_fail(self):
        with self.assertRaises(ValueError):
            MicroLearner().update("install_unknown")
        with self.assertRaises(ValueError):
            MicroLearner().update("retain_c")
        with self.assertRaises(ValueError):
            MicroLearner(3, True).update("install_shear")


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.broker = Broker(self.root)

    def make_promotion(self):
        base = self.broker.incumbent
        p = heuristic_proposal(base)
        return self.broker.evaluate(p, base.update(p["operator"]), blocks=4096)

    def test_promotion_and_private_seed_replay(self):
        receipt = self.make_promotion()
        self.assertEqual(receipt["status"], "promoted")
        self.assertGreater(receipt["intervals"]["target"][0], 0.05)
        self.assertEqual(self.broker.incumbent.retained_bits, 2)
        self.assertTrue(self.broker.verify_scoring())
        self.assertEqual(self.broker.state["anchors"]["T0"]["retained_bits"], 1)
        self.assertEqual(self.broker.state["anchors"]["T1"]["retained_bits"], 2)

    def test_two_step_graph_hill_climb(self):
        result = run_demo(self.root, blocks=4096)
        self.assertEqual([a["status"] for a in result["attempts"]], ["promoted", "promoted"])
        self.assertEqual(result["exact_accuracy"], {"T0":1, "T1":1, "T4":1})
        self.assertTrue(result["chain_verified"])
        self.assertEqual(self.broker._load()["attempt_next"], 3)
        self.assertEqual(set(self.broker.state["anchors"]), {"T0", "T1", "T4"})

    def test_regressive_candidate_rejected_against_anchor(self):
        self.make_promotion()
        old = self.broker.incumbent
        p = {"operator": "drop_b", "target": "T1",
             "diagnostics": diagnose(old), "source": "regression-test",
             "competing_explanation": "none", "predicted_potential_gain": -0.2}
        receipt = self.broker.evaluate(p, old.update("drop_b"), blocks=4096)
        self.assertEqual(receipt["status"], "rejected")
        self.assertEqual(self.broker.incumbent, old)
        self.assertLess(receipt["intervals"]["aggregate"][1], 0.025)
        self.assertTrue(self.broker.verify_scoring())

    def test_missing_diagnostics_invalid_consumes_attempt(self):
        p = heuristic_proposal(self.broker.incumbent)
        del p["diagnostics"]
        r = self.broker.evaluate(p, self.broker.incumbent.update("retain_b"), blocks=4096)
        self.assertEqual(r["status"], "invalid")
        self.assertEqual(self.broker.state["attempt_next"], 2)
        self.assertEqual(self.broker.incumbent, MicroLearner())
        self.assertTrue(self.broker.verify_chain())

    def test_candidate_must_be_frozen_expected_update(self):
        p = heuristic_proposal(self.broker.incumbent)
        r = self.broker.evaluate(p, MicroLearner(3, True), blocks=4096)
        self.assertEqual(r["status"], "invalid")
        self.assertIn("mutation", r["reason"])

    def test_budget_rejected_before_acquisition(self):
        p = heuristic_proposal(self.broker.incumbent)
        r = self.broker.evaluate(p, self.broker.incumbent.update("retain_b"), blocks=100000)
        self.assertEqual(r["status"], "invalid")
        self.assertFalse((self.root / "broker_private" / "attempt_0001.key").exists())

    def test_event_mutation_detected(self):
        self.make_promotion()
        log = self.root / "events.jsonl"
        contents = log.read_text().replace('"status":"promoted"', '"status":"rejected"')
        self.assertNotEqual(log.read_text(), contents)
        log.write_text(contents)
        with self.assertRaises(ProtocolError):
            self.broker.verify_chain()

    def test_seed_mutation_detected(self):
        self.make_promotion()
        k = self.root / "broker_private" / "attempt_0001.key"
        k.write_bytes(b"0" * 32)
        with self.assertRaisesRegex(ProtocolError, "seed"):
            self.broker.verify_scoring()

    def test_source_or_graph_mismatch_blocks_reuse(self):
        state = json.loads((self.root / "state.json").read_text())
        state["graph_sha"] = "0" * 64
        (self.root / "state.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ProtocolError, "graph or scorer"):
            Broker(self.root)

    def test_inconclusive_candidate_stays_off_incumbent(self):
        self.make_promotion()
        old = self.broker.incumbent
        p = {"operator": "retain_c", "target": "T1",
             "diagnostics": diagnose(old), "source": "unit",
             "competing_explanation": "unused variable", "predicted_potential_gain": 0}
        receipt = self.broker.evaluate(p, old.update("retain_c"), blocks=4096)
        self.assertEqual(receipt["status"], "insufficient_evidence")
        self.assertEqual(self.broker.incumbent, old)

    def test_restart_retains_attempt_and_anchor_history(self):
        self.make_promotion()
        restarted = Broker(self.root)
        self.assertEqual(restarted.incumbent.retained_bits, 2)
        p = heuristic_proposal(restarted.incumbent)
        r = restarted.evaluate(p, restarted.incumbent.update("install_shear"), blocks=4096)
        self.assertEqual(r["status"], "promoted")
        self.assertEqual(r["attempt"], 2)
        self.assertTrue(restarted.verify_scoring())


if __name__ == "__main__":
    unittest.main()
