import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "optional Gymnasium dependency is absent")
class GymnasiumTests(unittest.TestCase):
    def test_checker_and_masked_action_clock(self):
        from gymnasium.utils.env_checker import check_env
        from research.pluto_gym.gymnasium_adapter import GymnasiumMultiFrontEnv
        env = GymnasiumMultiFrontEnv()
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(obs))
        obs, _, _, _, info = env.step(1)
        self.assertEqual(obs["tick"], 1)
        self.assertTrue(info["invalid_action"])
        with self.assertRaises(ValueError):
            env.step(True)
        self.assertEqual(env.core.tick, 1)


if __name__ == "__main__":
    unittest.main()

