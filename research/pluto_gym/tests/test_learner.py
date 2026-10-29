"""Behavioral checks for bounded replay and synthetic student training."""

import copy
import importlib.util
import json
import math
import unittest

from research.pluto_gym.replay import DRAMReplay
from research.pluto_gym.learner import Student, collect_demonstrations, evaluate


class ReplayTests(unittest.TestCase):
    def test_wrap_retains_newest_and_fixed_allocation(self):
        replay = DRAMReplay(capacity=3, feature_dim=2, max_bytes=51)
        allocation = replay.ledger()["allocated_payload_bytes"]
        for j in range(7):
            replay.append([j, j + 1], j % 2)
        self.assertEqual(len(replay), 3)
        self.assertEqual([row["features"][0] for row in replay.rows()], [4, 5, 6])
        self.assertEqual(replay.ledger()["overwritten_rows"], 4)
        self.assertEqual(replay.ledger()["allocated_payload_bytes"], allocation)
        self.assertEqual(allocation, 51)

    def test_sampling_is_reproducible_and_copies_rows(self):
        replay = DRAMReplay(capacity=8)
        replay.extend({"features": [j, j + 1], "target": j % 2} for j in range(8))
        self.assertEqual(replay.sample(50, seed=9), replay.sample(50, seed=9))
        self.assertNotEqual(replay.sample(50, seed=9), replay.sample(50, seed=10))
        sampled = replay.sample(1, seed=2)
        sampled[0]["features"][0] = 100
        self.assertTrue(all(row["features"][0] < 100 for row in replay.rows()))

    def test_invalid_append_preserves_ring(self):
        replay = DRAMReplay(capacity=1)
        replay.append([0, 1], 1)
        before = replay.rows(), replay.ledger()
        for features, target in [([0], 1), ([True, 1], 0), ([float("nan"), 0], 1),
                                 ([0, 1], True), ([0, 1], 0.9), ([0, 1], 2),
                                 ([0, "1"], 0), ([float("inf"), 0], 0)]:
            with self.subTest(features=features, target=target):
                with self.assertRaises(ValueError):
                    replay.append(features, target)
                self.assertEqual(before, (replay.rows(), replay.ledger()))

    def test_budget_and_empty_validation(self):
        for kwargs in ({"capacity": 0}, {"capacity": True}, {"capacity": 1.2},
                       {"capacity": 100, "max_bytes": 100}, {"feature_dim": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DRAMReplay(**kwargs)
        with self.assertRaises(ValueError):
            DRAMReplay(capacity=1).sample(1)


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "NumPy is optional for student training")
class StudentTests(unittest.TestCase):
    @staticmethod
    def xor_examples():
        return [{"features": [cue, rule], "target": cue ^ rule}
                for cue in range(2) for rule in range(2)]

    def test_learning_changes_weights_and_fits_xor(self):
        model = Student(seed=0, width=8)
        before = model.checkpoint_sha256()
        replay = DRAMReplay(capacity=4)
        replay.extend(self.xor_examples())
        metrics = model.train(replay, updates=400, batch_size=16, seed=1)
        self.assertNotEqual(before, model.checkpoint_sha256())
        self.assertEqual(metrics["examples_drawn"], 6400)
        self.assertEqual(metrics["replay"]["allocated_payload_bytes"], 68)
        self.assertLess(metrics["final_corpus_cross_entropy"], 0.02)
        self.assertLess(metrics["final_corpus_cross_entropy"], metrics["initial_corpus_cross_entropy"])
        for row in self.xor_examples():
            self.assertEqual(int(model.predict_probability(*row["features"]) >= 0.5), row["target"])

    def test_checkpoint_roundtrip_and_clone_independence(self):
        model = Student(seed=6)
        model.train(self.xor_examples(), updates=100, batch_size=8, seed=1)
        state = json.loads(json.dumps(model.state_dict(), allow_nan=False))
        clone = Student.from_state_dict(state)
        self.assertEqual(model.checkpoint_sha256(), clone.checkpoint_sha256())
        for row in self.xor_examples():
            self.assertEqual(model.predict_probability(*row["features"]), clone.predict_probability(*row["features"]))
        clone.train(self.xor_examples(), updates=1, batch_size=8, seed=1)
        self.assertNotEqual(model.checkpoint_sha256(), clone.checkpoint_sha256())
        self.assertEqual(model.training_updates, 100)

    def test_checkpoint_rejects_bad_values_shapes_and_schema(self):
        original = Student().state_dict()
        invalid = []
        state = copy.deepcopy(original)
        state["weights"]["b2"] = [float("nan")]
        invalid.append(state)
        state = copy.deepcopy(original)
        state["weights"]["b2"] = [True]
        invalid.append(state)
        state = copy.deepcopy(original)
        state["weights"]["b2"] = ["0"]
        invalid.append(state)
        state = copy.deepcopy(original)
        state["weights"]["W1"] = [[0]]
        invalid.append(state)
        state = copy.deepcopy(original)
        state["schema_version"] = True
        invalid.append(state)
        state = copy.deepcopy(original)
        state["extra"] = 1
        invalid.append(state)
        for state in invalid:
            with self.subTest(state=state), self.assertRaises(ValueError):
                Student.from_state_dict(state)

    def test_seeded_training_reproduces_parameters(self):
        left = Student(seed=8, width=4)
        right = left.clone()
        left.train(self.xor_examples(), updates=80, batch_size=16, seed=99)
        right.train(self.xor_examples(), updates=80, batch_size=16, seed=99)
        self.assertEqual(left.state_dict(), right.state_dict())

    def test_replay_contents_change_training_result(self):
        left = Student(seed=2)
        right = left.clone()
        replay_a = DRAMReplay(capacity=4)
        replay_b = DRAMReplay(capacity=4)
        replay_a.extend(self.xor_examples())
        replay_b.extend({"features": x["features"], "target": 1 - x["target"]}
                        for x in self.xor_examples())
        a = left.train(replay_a, updates=300, batch_size=16, seed=0)
        b = right.train(replay_b, updates=300, batch_size=16, seed=0)
        self.assertNotEqual(a["replay_rows_sha256"], b["replay_rows_sha256"])
        self.assertGreater(abs(left.predict_probability(0, 0) - right.predict_probability(0, 0)), 0.9)

    def test_zero_updates_and_rejections_preserve_model(self):
        model = Student()
        before = model.state_dict()
        model.train(self.xor_examples(), updates=0)
        self.assertEqual(before, model.state_dict())
        for kwargs in ({"updates": True}, {"batch_size": 0}, {"learning_rate": float("inf")},
                       {"learning_rate": True}, {"updates": -1}, {"seed": 0.2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.train(self.xor_examples(), **kwargs)
            self.assertEqual(before, model.state_dict())
        with self.assertRaises(ValueError):
            model.train([{"features": [2, 0], "target": 0}], updates=1)
        self.assertEqual(before, model.state_dict())

    def test_environment_demo_and_frozen_evaluation(self):
        from research.pluto_gym.env import InterfaceSpec, RuleSpec
        rules = RuleSpec()
        interface = InterfaceSpec()
        demos = collect_demonstrations(rules, interface, range(10))
        self.assertTrue(demos)
        self.assertTrue(all(row["source"] == "synthetic_exact_oracle" for row in demos))
        model = Student(seed=4)
        model.train(demos, updates=150, batch_size=16, seed=3)
        checkpoint = model.checkpoint_sha256()
        results = evaluate(model, rules, interface, range(100, 120))
        self.assertEqual(model.checkpoint_sha256(), checkpoint)
        self.assertEqual(results["target_error_rate"], 0)
        self.assertAlmostEqual(results["mean_reward"], results["interface_oracle_value"])
        self.assertEqual(results["mean_interface_regret"], 0)
        self.assertEqual(len(results["episodes"]), 20)
        self.assertLess(results["brier_score"], 0.01)
        self.assertEqual(model.ledger()["trainable_parameters"], 33)

    def test_policy_ignores_hidden_cues_when_scheduling(self):
        from research.pluto_gym.env import InterfaceSpec, MultiFrontEnv, RuleSpec
        env = MultiFrontEnv(RuleSpec(), InterfaceSpec())
        observation, _ = env.reset(seed=10)
        model = Student(seed=0)
        action = model.act(observation)
        self.assertEqual(action["kind"], "attack")
        altered = copy.deepcopy(observation)
        for front, cue in enumerate(altered["visible_cues"]):
            if cue is None:
                altered["visible_cues"][front] = 1
        self.assertEqual(action, model.act(altered))


if __name__ == "__main__":
    unittest.main()
