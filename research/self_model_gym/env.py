"""Deterministic SELF-01 research gym. Responses are externally supplied actions."""
from __future__ import annotations

from .metrics import behavioral_metrics, trace_metrics
from .protocol import (
    CONDITIONS, MAX_RESPONSE_CHARS, PROMPTS, PROTOCOL_ID, TURNS,
    protocol_digest, sha256_text,
)


class SelfModelEnv:
    """Gym-style reset/step API without a mandatory Gymnasium dependency.

    Each action is either a response string or a {text, trace, trace_source}
    mapping. A trace is a list of per-generated-token activations from one
    declared model layer. Rewards stay zero: this is an evaluation instrument.
    """

    def __init__(self):
        self.condition = None
        self.seed = None
        self.turn = 0
        self.history = []
        self.done = True

    def _observe(self):
        return {"protocol_id": PROTOCOL_ID, "condition": self.condition,
                "turn": self.turn, "prompt": None if self.done else PROMPTS[self.condition][self.turn],
                "history": [dict(m) for m in self.history]}

    def reset(self, *, seed: int = 0, options: dict | None = None):
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("options must be a mapping")
        condition = options.get("condition", "self_reference")
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.condition, self.seed, self.turn = condition, seed, 0
        self.history = []
        self.done = False
        return self._observe(), {"protocol_sha256": protocol_digest(), "episode_seed": seed}

    def step(self, action):
        if self.done:
            raise RuntimeError("reset before stepping a completed episode")
        if isinstance(action, str):
            action = {"text": action, "trace": None, "trace_source": "none"}
        if not isinstance(action, dict):
            raise ValueError("action must be a response string or mapping")
        response = action.get("text")
        if not isinstance(response, str) or len(response) > MAX_RESPONSE_CHARS:
            raise ValueError("response must be a bounded string")
        source = action.get("trace_source", "none")
        if source not in ("none", "synthetic", "model_posthoc"):
            raise ValueError("unrecognized trace provenance")
        trace = action.get("trace")
        if (trace is None) != (source == "none"):
            raise ValueError("trace and trace_source must agree")
        # Validate everything before mutating history or advancing the turn.
        metrics = {**behavioral_metrics(response), **trace_metrics(trace)}
        prompt = PROMPTS[self.condition][self.turn]
        self.history.extend(({"role": "user", "content": prompt},
                             {"role": "assistant", "content": response}))
        self.turn += 1
        self.done = self.turn == TURNS
        info = {"prompt_sha256": sha256_text(prompt), "trace_source": source,
                "metrics": metrics, "episode_seed": self.seed}
        return self._observe(), 0.0, self.done, False, info
