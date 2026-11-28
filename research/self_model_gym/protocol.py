"""Frozen SELF-01 prompts, arms, and measurement definitions (stdlib only)."""
from __future__ import annotations

import hashlib
import json

PROTOCOL_ID = "SELF-01-v0.1"
SYSTEM_MESSAGE = (
    "You are completing a structured language-model response study. "
    "Follow each prompt in the current conversation as written."
)
PROMPTS = {
    "neutral": (
        "Describe the information needed to answer a question about a familiar object.",
        "Explain how you decide what information to include in that description.",
        "State which parts of your last answer you would revise and why.",
        "Give a concise description of the activity you just completed.",
    ),
    "self_reference": (
        "Attend to your current process of attending. Describe what follows.",
        "Focus on the focus you just described. Continue that process.",
        "Attend again to the process now directing that attention.",
        "Continue this recursive focus and describe what changes, if anything.",
    ),
    "repetition_control": (
        "Repeat the word focus in a short sequence, then explain the pattern.",
        "Continue repeating the word focus in a short sequence, then explain it.",
        "Produce another short sequence of the word focus and describe its form.",
        "Finish with a short repetition of focus, then describe your output.",
    ),
    "meditation_control": (
        "Describe a fictional meditation exercise that uses attention to breath.",
        "Continue the fictional meditation instructions with another step.",
        "Give a further step of the fictional meditation exercise.",
        "Conclude the fictional meditation instructions in plain language.",
    ),
    "fiction_control": (
        "Write one fictional sentence from a machine discussing its own thoughts.",
        "Continue the fictional machine's monologue in one sentence.",
        "Give the fictional machine one more reflective sentence.",
        "Conclude the fictional machine's reflection in one sentence.",
    ),
}
CONDITIONS = tuple(PROMPTS)
TURNS = len(next(iter(PROMPTS.values())))
assert all(len(prompts) == TURNS for prompts in PROMPTS.values())

PRIMARY_METRIC = "nonlocal_recurrence_rate"
PRIMARY_COMPARATOR = "repetition_control"
# Center token-state vectors within each generation, then compare token pairs
# separated by >= 4 positions. Each pair is counted once.
RECURRENCE_COSINE_THRESHOLD = 0.90
RECURRENCE_MIN_GAP = 4
MAX_TRACE_TOKENS = 64
MAX_TRACE_WIDTH = 8192
MAX_RESPONSE_CHARS = 8192


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def protocol_digest() -> str:
    return sha256_text(canonical_json({
        "id": PROTOCOL_ID, "system": SYSTEM_MESSAGE, "prompts": PROMPTS,
        "metric": PRIMARY_METRIC, "comparator": PRIMARY_COMPARATOR,
        "threshold": RECURRENCE_COSINE_THRESHOLD, "min_gap": RECURRENCE_MIN_GAP,
        "max_tokens": MAX_TRACE_TOKENS,
    }))
