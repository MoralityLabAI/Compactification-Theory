"""Exact, CPU-only finite-state closure and intervention-transfer falsifier.

Run: python research/dynamical_closure/recoverability_experiment.py
No randomness, external packages, or fitted model on held-out words are used.
"""
import json
from collections import defaultdict
from fractions import Fraction
from itertools import product

from task_closure import FiniteSystem, bayes_risk, is_task_closed, run_word


STATES = tuple(product(range(2), repeat=3))  # (visible y, latent b, latent c)


def build_system():
    functions = {
        "identity": lambda y, b, c: (y, b, c),
        "flip_y": lambda y, b, c: (1 - y, b, c),
        "flip_b": lambda y, b, c: (y, 1 - b, c),
        "reveal_b": lambda y, b, c: (b, b, c),
        "reveal_c": lambda y, b, c: (c, b, c),
    }
    actions = tuple(functions)
    maps = tuple(tuple(STATES.index(functions[name](*state)) for state in STATES)
                 for name in actions)
    return FiniteSystem(maps, tuple(s[0] for s in STATES)), actions


def code_path():
    # Each code is a deterministic function of its predecessor.
    return (tuple(STATES), tuple(s[:2] for s in STATES),
            tuple(s[:1] for s in STATES), tuple(() for _ in STATES))


def fit_transition(codes, transition):
    """Empirical squared-loss optimum on the eight *training* states only."""
    buckets = defaultdict(list)
    for x, z in enumerate(codes):
        buckets[z].append(codes[transition[x]])
    return {z: tuple(Fraction(sum(next_z[j] for next_z in values), len(values))
                     for j in range(len(z)))
            for z, values in buckets.items()}


def fitted_mse(system, codes, tables, word):
    """Evaluate a held-out composition of individually trained action tables."""
    if len(codes[0]) == 0:
        # The only task prediction from the constant code is its training mean.
        prediction = Fraction(sum(system.task), len(system.task))
        return sum((Fraction(system.task[run_word(system, x, word)]) - prediction) ** 2
                   for x in range(len(codes))) / len(codes)
    loss = Fraction()
    for x, z in enumerate(codes):
        for action in word:
            z = tables[action][z]
        prediction = z[0]  # learned readout is the visible first coordinate
        target = system.task[run_word(system, x, word)]
        loss += (Fraction(target) - prediction) ** 2
    return loss / len(codes)


def first_failure(risks, tolerance=Fraction(1, 16)):
    return next((k for k, risk in enumerate(risks) if risk > tolerance), None)


def run():
    system, names = build_system()
    idx = {name: i for i, name in enumerate(names)}
    codes = code_path()
    # Only single-action tables are fitted. These two sequences are held out.
    fitted_actions = tuple(idx[a] for a in ("identity", "flip_y", "flip_b", "reveal_b"))
    heldout = {"flip_b_then_reveal_b": (idx["flip_b"], idx["reveal_b"]),
               "flip_y_then_reveal_b": (idx["flip_y"], idx["reveal_b"])}
    pilot = (idx["reveal_b"],)
    unseen_action = (idx["reveal_c"],)  # diagnostic only: not fitted

    def target(word):
        return tuple(system.task[run_word(system, x, word)] for x in range(len(STATES)))

    def risks(word):
        return tuple(bayes_risk(code, target(word)) for code in codes)

    tables = [{a: fit_transition(code, system.transitions[a]) for a in fitted_actions}
              for code in codes]
    def record(word, fit=False):
        rs = risks(word)
        result = {"oracle_bayes_risk": [str(v) for v in rs],
                  "threshold_k": first_failure(rs)}
        if fit:
            result["fitted_composition_mse"] = [str(fitted_mse(system, code, table, word))
                                                for code, table in zip(codes, tables)]
        return result

    observational = (idx["flip_b"], idx["flip_y"])
    return {
        "states": len(STATES), "measure": "uniform over all (y,b,c) in {0,1}^3",
        "code_path": ["(y,b,c)", "(y,b)", "(y)", "()"],
        "tolerance": "1/16", "task": "current y",
        "training": {"one_step_actions": [names[i] for i in fitted_actions],
                     "task_only_training_word": [names[i] for i in observational],
                     "task_only_training_risk": str(bayes_risk(codes[2], target(observational)))},
        "pilot_reveal_b": record(pilot),
        "heldout_compositions": {name: record(word, fit=True) for name, word in heldout.items()},
        "unseen_action_reveal_c": record(unseen_action),
        "closed_on_fitted_actions": [is_task_closed(FiniteSystem(
            tuple(system.transitions[a] for a in fitted_actions), system.task), code)
            for code in codes],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
