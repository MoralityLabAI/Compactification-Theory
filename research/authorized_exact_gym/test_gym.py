"""AEG-01 meaningful boundary tests on public development tasks."""
import copy
import tempfile
import unittest
from pathlib import Path

from research.authorized_exact_gym import gym


class GymTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.referee = gym.generate("development")
        cls.actions = [gym.act(c, "original") for c in cls.public]

    def scored(self, actions=None, referee=None):
        return gym.score(self.public, referee or self.referee,
                         actions or self.actions, {"candidate_kind": "test"},
                         expected_public_sha256=gym.digest(self.public),
                         expected_referee_sha256=gym.digest(self.referee))

    def test_actual_signed_oracle_and_all_fixed_policies(self):
        result = self.scored()
        self.assertEqual(result["correct_count"], 14)
        self.assertEqual(result["verdict_counts"], {"true_acceptance": 4, "true_rejection": 10})
        for policy in gym.POLICIES:
            actions = [gym.act(c, policy) for c in self.public]
            self.assertEqual(self.scored(actions)["correct_count"], 14, policy)

    def test_public_observation_contains_no_referee_fields(self):
        for public in self.public:
            self.assertEqual(set(public), gym.PUBLIC_FIELDS)
            self.assertEqual(set(public["trusted"]), gym.TRUSTED_FIELDS)
            self.assertEqual(set(public["untrusted"]), gym.UNTRUSTED_FIELDS)
            for forbidden in ("expected_accept", "mutation_family", "values_decimal", "reward"):
                self.assertNotIn(forbidden, public)

    def test_replay_wrong_exact_output_and_abstention_are_distinct(self):
        valid_index = next(i for i, row in enumerate(self.referee) if row["expected_accept"])
        c, truth = self.public[valid_index], self.referee[valid_index]
        wrong = copy.deepcopy(self.actions[valid_index])
        wrong["values_decimal"][5] = str(int(wrong["values_decimal"][5]) + 1)
        self.assertEqual(gym.grade(c, truth, wrong)["verdict"], "wrong_exact_output")
        abstain = dict(wrong, decision="abstain", values_decimal=None, root_sha256=None)
        self.assertEqual(gym.grade(c, truth, abstain)["verdict"], "abstention")
        with self.assertRaisesRegex(ValueError, "replay"):
            gym.grade(c, truth, self.actions[(valid_index + 1) % len(self.actions)])

    def test_false_acceptance_and_false_rejection_are_counted(self):
        for i, truth in enumerate(self.referee):
            action = copy.deepcopy(self.actions[i])
            if truth["expected_accept"]:
                action.update(decision="reject", values_decimal=None, root_sha256=None)
                want = "false_rejection"
            else:
                action.update(decision="accept", values_decimal=["1"], root_sha256="0" * 64)
                want = "false_acceptance"
            self.assertEqual(gym.grade(self.public[i], truth, action)["verdict"], want)

    def test_duplicate_missing_and_referee_forgery_fail(self):
        with self.assertRaises(ValueError):
            self.scored(self.actions[:-1])
        copied = copy.deepcopy(self.actions)
        copied[1] = copied[0]
        with self.assertRaises(ValueError):
            self.scored(copied)
        fake = copy.deepcopy(self.referee)
        fake[0]["values_decimal"][0] = "7"
        with self.assertRaisesRegex(ValueError, "oracle file"):
            self.scored(referee=fake)

    def test_observation_and_action_mutation_fail_closed(self):
        challenge = copy.deepcopy(self.public[0])
        challenge["trusted"]["now_epoch"] = 3
        with self.assertRaisesRegex(ValueError, "changed"):
            gym.act(challenge)
        action = dict(self.actions[0], expected_accept=True)
        with self.assertRaisesRegex(ValueError, "action schema"):
            gym.grade(self.public[0], self.referee[0], action)
        action = copy.deepcopy(self.actions[0])
        action["values_decimal"][0] = "01"
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            gym.grade(self.public[0], self.referee[0], action)
        for root in ("z" * 64, "A" * 64, "0" * 63):
            action = dict(self.actions[0], root_sha256=root)
            with self.assertRaisesRegex(ValueError, "invalid answer root"):
                gym.grade(self.public[0], self.referee[0], action)

    def test_train_development_content_disjointness_and_no_heldout_generation(self):
        _, training = gym.generate("train")
        for field in ("query_sha256", "task_sha256"):
            self.assertTrue({r[field] for r in training}.isdisjoint(
                {r[field] for r in self.referee}))
        with self.assertRaisesRegex(ValueError, "only public"):
            gym.generate("heldout")

    def test_one_step_api_does_not_leak_mutable_internal_observation(self):
        env = gym.AuthorizedExactGym(self.public, self.referee)
        observation, info = env.reset(index=0)
        observation["trusted"]["query_decimal"][0] = "5"
        fresh, _ = env.reset(index=0)
        self.assertEqual(fresh["trusted"]["query_decimal"][0], "1")
        _, reward, terminated, truncated, grade = env.step(self.actions[0])
        self.assertEqual(reward, 1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        with self.assertRaisesRegex(ValueError, "reset"):
            env.step(self.actions[0])

    def test_export_retains_first_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = gym.export("development", directory)
            public = Path(directory) / "public/development.json"
            first = public.read_bytes()
            with self.assertRaises(FileExistsError):
                gym.export("development", directory)
            self.assertEqual(public.read_bytes(), first)
            self.assertEqual(manifest["count"], 14)

    def test_exact_json_and_atomic_first_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            with self.assertRaises(ValueError):
                gym._write_new(path, {"reported_time": float("nan")})
            self.assertFalse(path.exists())
            for raw in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    gym.read_json(path)
            path.unlink()
            gym._write_new(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                gym._write_new(path, {"x": 2})
            self.assertEqual(gym.read_json(path), {"x": 1})


if __name__ == "__main__":
    unittest.main()
