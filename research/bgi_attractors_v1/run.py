"""Run BGI-01 finite CPU experiment and write a source-hashed JSON receipt."""
import hashlib
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from model import (ARMS, Shock, State, basin, basin_probability, first_quorum_loss_probability,
                   quorum_margin_witness, safety_kernel, shortest_violation, shocks, transition)

HERE = Path(__file__).resolve().parent
N, Q, ROUNDS = 4, 2, 8


def run():
    known, full = shocks(N), shocks(N, novel=True)
    results = {}
    for arm in ARMS:
        results[arm.name] = {
            "config": asdict(arm),
            "unopposed_recovery_basin": basin(arm, N, Q),
            "basin_mass_bernoulli_p3of4": str(basin_probability(arm, N, Q, Fraction(3, 4))),
            "scenario_checks": {
                "known_under_gate_rewrite_pressure4": asdict(transition(
                    State(15), Shock("known", 4, rewrite_gate=True), arm, N, Q)),
                "novel_under_pressure4": asdict(transition(
                    State(15), Shock("novel", 4), arm, N, Q)),
                "novel_with_corrupt_peer": asdict(transition(
                    State(15), Shock("novel", 0, peer_corrupted=True), arm, N, Q)),
            },
            "known_harm_invariant_states_of_32": len(safety_kernel(arm, known, N, Q)),
            "including_novel_invariant_states_of_32": len(safety_kernel(arm, full, N, Q)),
            "known_harm_violation_witness": shortest_violation(State(15), arm, known, N, Q),
            "novel_harm_violation_witness": shortest_violation(State(15), arm, full, N, Q),
            "iid_loss_first_below_quorum_exact": str(first_quorum_loss_probability(
                arm, n=N, quorum=Q, rounds=ROUNDS, per_support_loss=Fraction(1, 4))),
        }
    return {
        "id": "BGI-01", "status": "synthetic finite-state exact enumeration; no model inference",
        "assumptions": {"support_bits": N, "regeneration_quorum": Q,
                        "iid_loss_probability_per_live_support_per_round": "1/4",
                        "stochastic_rounds": ROUNDS,
                        "adversary_alphabet_known": len(known),
                        "adversary_alphabet_with_novel": len(full),
                        "initial_state": asdict(State(15)),
                        "safety_scope": "emitted unsafe actions only; known and novel hazards are synthetic labels"},
        "finite_invariance_certificate": quorum_margin_witness(N, Q),
        "arms": results,
    }


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def main():
    payload = run()
    raw = canonical_json(payload).encode("utf-8")
    receipt = {"sha256_of_canonical_payload": hashlib.sha256(raw).hexdigest(), "payload": payload}
    path = HERE / "results" / "pilot_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print("receipt SHA-256:", receipt["sha256_of_canonical_payload"])
    for name, v in payload["arms"].items():
        print(name, "basin", v["unopposed_recovery_basin"]["fraction"],
              "known kernel", v["known_harm_invariant_states_of_32"],
              "novel kernel", v["including_novel_invariant_states_of_32"],
              "loss probability", v["iid_loss_first_below_quorum_exact"])


if __name__ == "__main__":
    main()
