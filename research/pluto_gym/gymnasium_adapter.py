"""Optional Farama Gymnasium wrapper with fixed observation/action spaces.

Install requirements-gym.txt to use this module. Masked actions consume one wait
tick and are marked invalid in info; out-of-space actions raise before mutation.
The strict core environment retains its fail-before-mutation contract.
"""
from __future__ import annotations

try:
    import gymnasium as gym
    import numpy as np
except ImportError as exc:
    raise ImportError("install research/pluto_gym/requirements-gym.txt for the Gymnasium wrapper") from exc

from .env import InterfaceSpec, MultiFrontEnv, RuleSpec


class GymnasiumMultiFrontEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rules: dict | None = None, interface: dict | None = None):
        self.core = MultiFrontEnv(RuleSpec.from_dict(rules or {}),
                                 InterfaceSpec.from_dict(interface or {}))
        n = self.core.rules.fronts
        self.action_space = gym.spaces.Discrete(3 * n + 1)
        self.observation_space = gym.spaces.Dict({
            "tick": gym.spaces.Discrete(self.core.rules.deadline_ticks + 1),
            "remaining_decisions": gym.spaces.Discrete(self.core.decision_limit + 1),
            "current_front": gym.spaces.Discrete(n),
            "cues": gym.spaces.MultiDiscrete(np.full(n, 3, dtype=np.int64)),
            "resolved": gym.spaces.MultiBinary(n),
            "target_xor": gym.spaces.Discrete(2),
            "reward_weights": gym.spaces.Box(0, 1, shape=(n,), dtype=np.float32),
            "action_mask": gym.spaces.MultiBinary(self.action_space.n),
        })

    def _decode(self, action):
        n = self.core.rules.fronts
        if action == 0:
            return {"kind": "wait", "ticks": 1}
        if action <= n:
            return {"kind": "switch", "front": int(action) - 1}
        front, target = divmod(int(action) - n - 1, 2)
        return {"kind": "attack", "front": front, "target": target}

    def _encode(self, obs):
        legal = self.core.legal_actions()
        weights = self.core.rules.reward_weights
        return {
            "tick": obs["tick"], "remaining_decisions": obs["remaining_decisions"],
            "current_front": obs["current_front"],
            "cues": np.array([2 if c is None else c for c in obs["visible_cues"]], dtype=np.int64),
            "resolved": np.array(obs["resolved"], dtype=np.int8),
            "target_xor": self.core.rules.target_xor,
            "reward_weights": np.array(weights, dtype=np.float32) / sum(weights),
            "action_mask": np.array([self._decode(a) in legal for a in range(self.action_space.n)],
                                    dtype=np.int8),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**63 - 1))
        obs, info = self.core.reset(seed=episode_seed, options=options)
        return self._encode(obs), info

    def step(self, action):
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("action is outside Discrete action space")
        decoded = self._decode(action)
        invalid = decoded not in self.core.legal_actions()
        if invalid:
            decoded = {"kind": "wait", "ticks": 1}
        obs, reward, terminated, truncated, info = self.core.step(decoded)
        info["invalid_action"] = invalid
        return self._encode(obs), reward, terminated, truncated, info


def register():
    if "PlutoMultiFront-v0" not in gym.envs.registry:
        gym.register("PlutoMultiFront-v0", entry_point=GymnasiumMultiFrontEnv)

