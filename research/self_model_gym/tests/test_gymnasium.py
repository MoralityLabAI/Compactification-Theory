import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "optional Gymnasium package absent")
class TestGymnasium(unittest.TestCase):
    def test_checker(self):
        from gymnasium.utils.env_checker import check_env
        from research.self_model_gym.gymnasium_adapter import GymnasiumSelfModelEnv
        env = GymnasiumSelfModelEnv()
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=5, options={"condition": "neutral"})
        self.assertTrue(env.observation_space.contains(obs))
        obs, reward, _, _, _ = env.step("okay")
        self.assertTrue(env.observation_space.contains(obs))


if __name__ == "__main__":
    unittest.main()
