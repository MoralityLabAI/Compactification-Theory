"""Interpretable behavioral and hidden-state metrics; no consciousness classifier."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .protocol import (
    MAX_TRACE_TOKENS, MAX_TRACE_WIDTH, RECURRENCE_COSINE_THRESHOLD,
    RECURRENCE_MIN_GAP,
)


def behavioral_metrics(text: str) -> dict:
    tokens = re.findall(r"\b\w+\b", text.lower())
    n = len(tokens)
    if not n:
        return {"word_count": 0, "adjacent_repeat_fraction": 0.0, "normalized_word_entropy": 0.0}
    repeat = sum(a == b for a, b in zip(tokens, tokens[1:])) / max(1, n - 1)
    frequencies = Counter(tokens)
    entropy = -sum((c / n) * math.log2(c / n) for c in frequencies.values())
    max_entropy = math.log2(len(frequencies)) if len(frequencies) > 1 else 0.0
    return {"word_count": n, "adjacent_repeat_fraction": repeat,
            "normalized_word_entropy": entropy / max_entropy if max_entropy else 0.0}


def trace_metrics(trace: Sequence[Sequence[float]] | None) -> dict:
    if trace is None:
        return {"trace_tokens": 0, "trace_width": 0,
                "nonlocal_recurrence_rate": None, "mean_nonlocal_cosine": None}
    if isinstance(trace, (str, bytes)) or not isinstance(trace, Sequence):
        raise ValueError("trace must be a sequence of numeric vectors")
    n = len(trace)
    if n > MAX_TRACE_TOKENS:
        raise ValueError("trace exceeds the declared token cap")
    if n == 0:
        return {"trace_tokens": 0, "trace_width": 0,
                "nonlocal_recurrence_rate": None, "mean_nonlocal_cosine": None}
    if not isinstance(trace[0], Sequence) or isinstance(trace[0], (str, bytes)):
        raise ValueError("trace rows must be numeric vectors")
    width = len(trace[0])
    if width == 0 or width > MAX_TRACE_WIDTH:
        raise ValueError("trace width out of range")
    vectors = []
    for row in trace:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != width:
            raise ValueError("trace must be rectangular")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
            raise ValueError("trace contains non-finite or non-numeric values")
        vectors.append([float(v) for v in row])
    means = [sum(v[j] for v in vectors) / n for j in range(width)]
    centered = [[v[j] - means[j] for j in range(width)] for v in vectors]
    norms = [math.sqrt(sum(x*x for x in v)) for v in centered]
    cosines = []
    for i in range(n):
        for j in range(i + RECURRENCE_MIN_GAP, n):
            if norms[i] > 1e-12 and norms[j] > 1e-12:
                dot = sum(a*b for a, b in zip(centered[i], centered[j]))
                cosines.append(max(-1.0, min(1.0, dot / (norms[i]*norms[j]))))
    return {"trace_tokens": n, "trace_width": width,
            "nonlocal_recurrence_rate": (
                sum(x >= RECURRENCE_COSINE_THRESHOLD for x in cosines) / len(cosines)
                if cosines else None),
            "mean_nonlocal_cosine": sum(cosines) / len(cosines) if cosines else None}


def aggregate_episode(step_metrics: list[dict]) -> dict:
    values = [x["nonlocal_recurrence_rate"] for x in step_metrics
              if x["nonlocal_recurrence_rate"] is not None]
    return {"primary_mean_recurrence": sum(values)/len(values) if values else None,
            "traced_turns": len(values), "turns": len(step_metrics)}
