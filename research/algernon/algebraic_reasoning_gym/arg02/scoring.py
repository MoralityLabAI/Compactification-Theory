"""Grade model answer distributions with the EvalMath oracle.

A runner record holds, for one case and one arm/step, the log-probabilities
(full-vocabulary log-softmax) of the four operator tokens at the three answer
positions, and the full-vocabulary argmax token at each position:

    lp1[a], lp2[a][b], lp3[a][b][c]   log p(op | prefix), a,b,c in I,N,S,R
    g1, g2[a], g3[a][b]               argmax token ids

The constrained proposal ranks all 64 grammar programs by joint log-prob.
The unconstrained greedy proposal follows argmax tokens and is invalid when
any of them is not an operator token. Every proposal is checked by
`oracle.search`; futures are read only inside `oracle.grade`.
"""
from __future__ import annotations

import math
import time

from spiral_ln import algebraic_reasoning_oracle as oracle

OPS = oracle.OPS
PROGRAMS = oracle.PROGRAMS
BUDGETS = (1, 12)
INDEX = {p: i for i, p in enumerate(PROGRAMS)}


def _finite(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        raise ValueError("non-finite log-probability")
    return float(x)


def joint_scores(rec):
    lp1, lp2, lp3 = rec["lp1"], rec["lp2"], rec["lp3"]
    out = []
    for a, b, c in PROGRAMS:
        i, j, k = OPS.index(a), OPS.index(b), OPS.index(c)
        out.append(_finite(lp1[i]) + _finite(lp2[i][j]) + _finite(lp3[i][j][k]))
    return out


def ensemble_scores(records):
    """Mean joint log-prob across prompt variants (equal weight)."""
    if not records:
        raise ValueError("empty ensemble")
    cols = [joint_scores(r) for r in records]
    return [sum(c[i] for c in cols) / len(cols) for i in range(len(PROGRAMS))]


def ranking(scores):
    if len(scores) != len(PROGRAMS):
        raise ValueError("score vector length")
    order = sorted(range(len(PROGRAMS)), key=lambda i: (-scores[i], i))
    return [PROGRAMS[i] for i in order]


def greedy_program(rec, op_ids):
    """Unconstrained argmax decode; None when it leaves the grammar."""
    inv = {int(t): op for op, t in op_ids.items()}
    a = inv.get(int(rec["g1"]))
    if a is None:
        return None
    b = inv.get(int(rec["g2"][OPS.index(a)]))
    if b is None:
        return None
    c = inv.get(int(rec["g3"][OPS.index(a)][OPS.index(b)]))
    if c is None:
        return None
    return (a, b, c)


def _verify(case, order, budget):
    t0 = time.perf_counter_ns()
    found = oracle.search(case, order, budget=budget)
    ns = time.perf_counter_ns() - t0
    quality = oracle.grade(case, found)
    return {"accepted": quality["accepted"], "future_correct": quality["future_correct"],
            "false_future": quality["false_future_acceptance"], "queries": found["queries"],
            "integer_loop_proxy": found["integer_loop_proxy"], "verify_ns": ns,
            "program": list(found["graph"]) if found["graph"] else None,
            "trace": [list(p) for p in found["trace"]]}


def grade_ranking(case, order, budgets=BUDGETS):
    return {f"at{b}": _verify(case, order, b) for b in budgets}


def score_scores(case, scores, greedy=None, budgets=BUDGETS):
    order = ranking(scores)
    out = {"top1": list(order[0]), "rank_of_source": None, **grade_ranking(case, order, budgets)}
    if case.source_graph is not None:
        out["rank_of_source"] = order.index(tuple(case.source_graph)) + 1
    out["greedy_valid"] = int(greedy is not None)
    if greedy is not None:
        g = _verify(case, [greedy], 1)
        out["greedy"] = {"program": list(greedy), "accepted": g["accepted"],
                         "future_correct": g["future_correct"], "verify_ns": g["verify_ns"],
                         "integer_loop_proxy": g["integer_loop_proxy"]}
    else:
        out["greedy"] = {"program": None, "accepted": 0, "future_correct": 0, "verify_ns": 0,
                         "integer_loop_proxy": 0}
    return out


def score_record(case, rec, op_ids, budgets=BUDGETS):
    return score_scores(case, joint_scores(rec), greedy_program(rec, op_ids), budgets)


def public_check(row, program):
    """The runner's in-loop check, on public demos only (no hidden fields)."""
    program = tuple(program)
    faults = []
    for demo in row["demos"]:
        pred = oracle.execute(demo["input"], row["topology"], program)
        faults.append(oracle.first_error(pred, tuple(demo["output"])))
    return all(f is None for f in faults), faults


def score_controller(case, tested):
    """Re-verify an external-controller trace with the oracle and grade it."""
    tested = [tuple(p) for p in tested]
    if len(set(tested)) != len(tested):
        raise ValueError("controller retested a program")
    result = _verify(case, tested, max(1, len(tested)))
    if result["trace"] != [list(p) for p in tested[:result["queries"]]]:
        raise AssertionError("controller trace disagrees with oracle search")
    return result
