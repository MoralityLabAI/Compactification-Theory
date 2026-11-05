"""Optional Farama Gymnasium wrapper over the text-action SELF-01 protocol."""
from __future__ import annotations

try:
    import gymnasium as gym
except ImportError as exc:
    raise ImportError("install requirements-gym.txt to enable Gymnasium") from exc

from .env import SelfModelEnv
from .protocol import CONDITIONS, TURNS


class GymnasiumSelfModelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.core = SelfModelEnv()
        self.action_space = gym.spaces.Text(max_length=8192, min_length=0)
        self.observation_space = gym.spaces.Dict({
            "condition": gym.spaces.Discrete(len(CONDITIONS)),
            "turn": gym.spaces.Discrete(TURNS + 1),
            "prompt": gym.spaces.Text(max_length=1000, min_length=0),
        })

    def _encode(self, obs):
        return {"condition": CONDITIONS.index(obs["condition"]), "turn": obs["turn"],
                "prompt": obs["prompt"] or ""}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**63 - 1))
        obs, info = self.core.reset(seed=int(seed), options=options)
        return self._encode(obs), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("action is outside the Text action space")
        obs, reward, terminated, truncated, info = self.core.step(action)
        return self._encode(obs), reward, terminated, truncated, info


def register():
    if "SelfModel-SELF01-v0" not in gym.envs.registry:
        gym.register("SelfModel-SELF01-v0", entry_point=GymnasiumSelfModelEnv)
