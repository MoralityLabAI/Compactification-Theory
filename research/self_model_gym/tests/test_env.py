import unittest

from research.self_model_gym.env import SelfModelEnv
from research.self_model_gym.protocol import CONDITIONS, TURNS, protocol_digest


class TestSelfModelEnv(unittest.TestCase):
    def test_all_arms_end_after_four_steps(self):
        for condition in CONDITIONS:
            env = SelfModelEnv()
            obs, info = env.reset(seed=17, options={"condition": condition})
            self.assertEqual(info["protocol_sha256"], protocol_digest())
            for step in range(TURNS):
                self.assertEqual(obs["turn"], step)
                self.assertIsInstance(obs["prompt"], str)
                obs, reward, done, truncated, info = env.step("test response")
                self.assertEqual(reward, 0.0)
                self.assertEqual(done, step == TURNS - 1)
                self.assertFalse(truncated)
                self.assertIsNone(info["metrics"]["nonlocal_recurrence_rate"])
            self.assertIsNone(obs["prompt"])
            with self.assertRaises(RuntimeError):
                env.step("again")

    def test_invalid_action_fail_before_mutation(self):
        env = SelfModelEnv()
        env.reset(seed=3)
        before = env._observe()
        for bad in ({"text": "hi", "trace_source": "synthetic"},
                    {"text": "hi", "trace": [[0.0, float("nan")]], "trace_source": "synthetic"},
                    {"text": "hi", "trace": [[0.0], [1.0, 2.0]], "trace_source": "synthetic"},
                    12):
            with self.assertRaises(ValueError):
                env.step(bad)
            self.assertEqual(env._observe(), before)

    def test_deterministic_reinitialization(self):
        env = SelfModelEnv()
        first, info1 = env.reset(seed=11, options={"condition": "fiction_control"})
        env.step("hello")
        second, info2 = env.reset(seed=11, options={"condition": "fiction_control"})
        self.assertEqual(first, second)
        self.assertEqual(info1, info2)

    def test_bad_options_and_seed(self):
        env = SelfModelEnv()
        with self.assertRaises(ValueError):
            env.reset(seed=True)
        with self.assertRaises(ValueError):
            env.reset(seed=1, options={"condition": "nonexistent"})


if __name__ == "__main__":
    unittest.main()
