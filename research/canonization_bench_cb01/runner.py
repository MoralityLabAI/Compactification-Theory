"""CB01: a bounded, deterministic companion oracle and Lean reuse metrics.

This tiny oracle is only a finite fixture. EvalMath owns the general IRC-02
stochastic reference. Metrics report kernel acceptance only after a real Lake
build, never from source inspection alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from itertools import product
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

ROOT = Path(__file__).resolve().parent
ACTIONS = ("read", "switch")
STATES = tuple(range(6))


@dataclass(frozen=True)
class Step:
    next_state: int
    signal: str
    reward: int
    ticks: int
    next_mode: str = "default"
    done: bool = False


# Two bisimilar pairs and a delayed distinguisher between state 0 and state 4.
# A switch costs two ticks, so the separating observation is possible at d=3.
TRANSITIONS: dict[tuple[int, str], Step] = {
    (0, "read"): Step(0, "idle", 0, 1),
    (0, "switch"): Step(1, "switch", 0, 2),
    (1, "read"): Step(1, "alert", 1, 1),
    (1, "switch"): Step(0, "switch", 0, 2),
    (2, "read"): Step(2, "idle", 0, 1),
    (2, "switch"): Step(3, "switch", 0, 2),
    (3, "read"): Step(3, "alert", 1, 1),
    (3, "switch"): Step(2, "switch", 0, 2),
    (4, "read"): Step(4, "idle", 0, 1),
    (4, "switch"): Step(5, "switch", 0, 2),
    (5, "read"): Step(5, "alarm", 1, 1),
    (5, "switch"): Step(4, "switch", 0, 2),
}


def validate_transitions(transitions: dict[tuple[int, str], Step]) -> None:
    for (state, action), edge in transitions.items():
        if state not in STATES or edge.next_state not in STATES:
            raise ValueError("transition leaves declared state set")
        if action not in ACTIONS or edge.ticks <= 0:
            raise ValueError("illegal action or nonpositive duration")


def action_words(actions: tuple[str, ...], budget: int) -> Iterable[tuple[str, ...]]:
    if budget < 0 or not actions:
        raise ValueError("positive action grammar and nonnegative budget required")
    for length in range(budget + 1):
        yield from product(actions, repeat=length)


def timed_trace(start: int, word: tuple[str, ...], budget: int,
                transitions: dict[tuple[int, str], Step] = TRANSITIONS) -> tuple[tuple, ...]:
    """Return all publicly visible events; stop once the finite clock expires."""
    if start not in STATES or budget < 0:
        raise ValueError("invalid starting state or budget")
    state = start
    out: list[tuple] = []
    for action in word:
        if budget == 0:
            break
        edge = transitions.get((state, action))
        if edge is None:
            out.append(("denied", action))
            break
        if edge.ticks > budget:
            out.append(("timeout", action, budget))
            break
        out.append((edge.signal, edge.reward, edge.ticks, edge.next_mode, edge.done))
        budget -= edge.ticks
        state = edge.next_state
        if edge.done:
            break
    return tuple(out)


def signature(state: int, budget: int, actions: tuple[str, ...] = ACTIONS,
              transitions: dict[tuple[int, str], Step] = TRANSITIONS) -> tuple:
    return tuple(timed_trace(state, word, budget, transitions)
                 for word in action_words(actions, budget))


def classes(budget: int, actions: tuple[str, ...] = ACTIONS,
            transitions: dict[tuple[int, str], Step] = TRANSITIONS) -> list[list[int]]:
    groups: dict[tuple, list[int]] = {}
    for state in STATES:
        groups.setdefault(signature(state, budget, actions, transitions), []).append(state)
    return list(groups.values())


def shortest_witness(x: int, y: int, budget: int,
                     actions: tuple[str, ...] = ACTIONS,
                     transitions: dict[tuple[int, str], Step] = TRANSITIONS) -> list[str] | None:
    for word in action_words(actions, budget):
        if timed_trace(x, word, budget, transitions) != timed_trace(y, word, budget, transitions):
            return list(word)
    return None


def memory_bits(class_count: int) -> int:
    if class_count < 1:
        raise ValueError("at least one class required")
    return (class_count - 1).bit_length()


def code_lines(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(bool(line.strip()) and not line.lstrip().startswith(("--", "/-", "-/", "*"))
               for line in lines)


def source_metrics() -> dict:
    paths = {
        "baseline_A": "CB01/BaselineA.lean",
        "baseline_B": "CB01/BaselineB.lean",
        "canonical": "CB01/Canonical.lean",
        "consumers": "CB01/Consumers.lean",
        "heldout": "CB01/Heldout.lean",
    }
    sizes = {k: code_lines(ROOT / v) for k, v in paths.items()}
    sources = {k: (ROOT / v).read_text(encoding="utf-8") for k, v in paths.items()}
    # Source inspection only: these are measured call sites, not a Lean kernel verdict.
    calls = {name: len(re.findall(r"\bCanonical\.(?:precompose|postcompose|pair)\b", sources[name]))
             for name in ("consumers", "heldout")}
    if calls["consumers"] != 2 or calls["heldout"] != 3:
        raise AssertionError("unexpected canonical reuse count")
    for path in ROOT.rglob("*.lean"):
        # Defense in depth; the Lean kernel and axiom audit remain authoritative.
        without_comments = re.sub(r"/-.*?-/|--[^\n]*", "", path.read_text(encoding="utf-8"),
                                  flags=re.DOTALL)
        if re.search(r"\b(sorry|admit|axiom|native_decide)\b", without_comments):
            raise AssertionError(f"proof placeholder or unsafe axiom token: {path}")
    return {
        "nonblank_source_lines": sizes,
        "canonical_call_sites": calls,
        "duplicate_baseline_proofs": 2,
        "canonical_precomposition_proofs": 1,
        "baseline_plus_clients_lines": sizes["baseline_A"] + sizes["baseline_B"],
        "canonical_plus_clients_lines": sizes["canonical"] + sizes["consumers"],
        "line_count_caution": "Raw source lines include signatures/imports; smaller is not assumed better.",
    }


def lake_status(require_lean: bool) -> str:
    if shutil.which("lake") is None:
        if require_lean:
            raise RuntimeError("Lake unavailable: cannot certify kernel acceptance")
        return "not_run_no_lake"
    process = subprocess.run(["lake", "build", "CB01"], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError("Lean kernel/build failed:\n" + process.stdout[-3000:] + process.stderr[-3000:])
    return "lake_build_passed"


def receipt(require_lean: bool = False) -> dict:
    validate_transitions(TRANSITIONS)
    partition = classes(3)
    refined = classes(3, ("read",))
    delayed = shortest_witness(0, 4, 3)
    mutation = dict(TRANSITIONS)
    mutation[(2, "read")] = replace(mutation[(2, "read")], ticks=2)
    counterexample = shortest_witness(0, 2, 3, transitions=mutation)
    checks = {
        "partition": partition == [[0, 2], [1, 3], [4], [5]],
        "memory_bits": memory_bits(len(partition)) == 2,
        "delayed_witness": delayed == ["switch", "read"],
        "duration_mutation": counterexample == ["read"],
        "grammar_refinement": len(refined) <= len(partition),
        "clock_refinement": len(classes(2)) <= len(partition),
    }
    if not all(checks.values()):
        raise AssertionError(f"finite oracle rejected check(s): {checks}")
    return {
        "schema": "cb01-receipt-v1",
        "kind": "synthetic_deterministic_finite_oracle",
        "clock_ticks": 3,
        "actions": list(ACTIONS),
        "equivalence_classes": partition,
        "class_count": len(partition),
        "minimum_state_code_bits_given_public_clock": memory_bits(len(partition)),
        "read_only_class_count": len(refined),
        "clock_2_class_count": len(classes(2)),
        "delayed_distinguishing_word_0_4": delayed,
        "mutated_duration_distinguishing_word_0_2": counterexample,
        "checks": checks,
        "source_metrics": source_metrics(),
        "lean_kernel_status": lake_status(require_lean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-lean", action="store_true",
                        help="fail if the Lean compiler is unavailable")
    parser.add_argument("--json", type=Path, help="write deterministic receipt to path")
    args = parser.parse_args()
    result = receipt(require_lean=args.require_lean)
    body = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(body, encoding="utf-8")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
