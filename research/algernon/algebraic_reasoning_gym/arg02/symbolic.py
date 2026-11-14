"""ARG-01 CPU baselines re-scored per case through the unmodified ARG-01 gym.

Arms: certified class-size order (12 checks), complete certified class
search, exhaustive 64-program search, frozen RRC-19 GRU and matched FF.
`gym.score_case` is called as is; its source hashes stay pinned.
"""
from __future__ import annotations

import time

from research.algernon.algebraic_reasoning_gym import gym

ARMS = ("certified_mass12", "certified_mass_full", "exhaustive_64", "frozen_gru", "frozen_ff")


def run(cases):
    t0 = time.perf_counter_ns()
    models = gym.checkpoints()
    load_ns = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns()
    classes = gym.compile_classes()
    compile_ns = time.perf_counter_ns() - t0
    out = {arm: {} for arm in ARMS}
    for case in cases:
        for arm in ARMS:
            r = gym.score_case(case, arm, models, classes)
            out[arm][case.identifier] = {
                "accepted": r["grade"]["accepted"], "future_correct": r["grade"]["future_correct"],
                "false_future": r["grade"]["false_future_acceptance"], "queries": r["queries"],
                "integer_loop_proxy": r["integer_loop_proxy"], "model_macs": r["model_macs"],
                "order_ns": r["order_ns"], "verify_ns": r["verify_ns"],
                "trace": [list(p) for p in r["trace"]]}
    return out, {"checkpoint_load_ns": load_ns, "class_compile_ns": compile_ns}
