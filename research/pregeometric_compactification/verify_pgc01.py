"""Independent algebraic checks for the PGC-01 operational influence gym."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from pgc01 import (BooleanSystem, accumulated_relation, affine_system,
                   all_relabelings, anf_powers, const, derivative,
                   fixture_systems, poly_evaluate, profiles,
                   relabel_state, relation_diagnostics, simulate, var)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_profile(system, profile, polynomials, counts):
    n = system.width
    counts["system_lag_profiles"] += 1
    for state, measured in enumerate(profile.outputs):
        expected = sum(poly_evaluate(poly, state) << j for j, poly in enumerate(polynomials))
        require(measured == expected, f"ANF trajectory mismatch: {system.name}")
        counts["trajectory_output_bits"] += n
    for source in range(n):
        for target in range(n):
            dp = derivative(polynomials[target], source)
            require(profile.maximum[source][target] == int(bool(dp)), "ANF support theorem")
            counts["support_entries"] += 1
            absolute = 0
            signed = 0
            for context in range(1 << n):
                if (context >> source) & 1:
                    continue
                y0 = (profile.outputs[context] >> target) & 1
                y1 = (profile.outputs[context | (1 << source)] >> target) & 1
                expected = poly_evaluate(dp, context)
                require((y0 ^ y1) == expected, "context-dependent Boolean derivative")
                absolute += expected
                signed += poly_evaluate(polynomials[target], context | (1 << source)) - poly_evaluate(polynomials[target], context)
                counts["paired_context_bit_responses"] += 1
            require(profile.absolute_counts[source][target] == absolute, "mean absolute response")
            require(profile.signed_counts[source][target] == signed, "mean signed response")


def independent_affine_rows(width, matrix, horizon):
    """Bit-matrix multiplication supplies an oracle separate from Boolean ASTs."""
    rows = [(matrix >> (width * j)) & ((1 << width) - 1) for j in range(width)]
    current = [1 << i for i in range(width)]
    result = []
    for _ in range(horizon):
        following = []
        for row in rows:
            value = 0
            for k in range(width):
                if (row >> k) & 1:
                    value ^= current[k]
            following.append(value)
        current = following
        result.append(tuple(current))
    return result


def verify_all_affine():
    counts = Counter()
    digest = sha256()
    horizon = 3
    for width in (1, 2, 3):
        for matrix in range(1 << (width * width)):
            expected_rows = independent_affine_rows(width, matrix, horizon)
            for bias in range(1 << width):
                system = affine_system(width, matrix, bias)
                measured = profiles(system, horizon)
                polynomials = anf_powers(system, horizon)
                counts["systems"] += 1
                counts[f"systems_width_{width}"] += 1
                for profile, polys, rows in zip(measured, polynomials, expected_rows):
                    check_profile(system, profile, polys, counts)
                    for source in range(width):
                        for target in range(width):
                            require(profile.maximum[source][target] == ((rows[target] >> source) & 1),
                                    "independent matrix-power oracle")
                            counts["matrix_power_support_entries"] += 1
                    digest.update(json.dumps([width, matrix, bias, profile.as_dict()],
                                             sort_keys=True, separators=(",", ":")).encode())
    return {"horizon": horizon, "counts": dict(sorted(counts.items())),
            "case_digest_sha256": digest.hexdigest()}


def verify_fixtures_and_relabelings():
    counts = Counter()
    fixtures = []
    digest = sha256()
    for system in fixture_systems():
        counts["systems"] += 1
        measured = profiles(system, 4)
        polynomials = anf_powers(system, 4)
        for profile, polys in zip(measured, polynomials):
            check_profile(system, profile, polys, counts)
        for permutation, renamed in all_relabelings(system):
            transformed = profiles(renamed, 4)
            counts["relabeling_cases"] += 1
            for old, new in zip(measured, transformed):
                for state, output in enumerate(old.outputs):
                    require(new.outputs[relabel_state(state, permutation)] == relabel_state(output, permutation),
                            "trajectory relabeling covariance")
                    counts["relabeling_trajectory_states"] += 1
                for i in range(system.width):
                    for j in range(system.width):
                        pi, pj = permutation[i], permutation[j]
                        require((new.maximum[pi][pj], new.absolute_counts[pi][pj], new.signed_counts[pi][pj]) ==
                                (old.maximum[i][j], old.absolute_counts[i][j], old.signed_counts[i][j]),
                                "influence relabeling covariance")
                        counts["relabeling_response_entries"] += 1
                digest.update(json.dumps([system.name, permutation, new.as_dict()],
                                         sort_keys=True, separators=(",", ":")).encode())
        fixtures.append({"system": system.as_dict(),
                         "profiles": [p.as_dict() for p in measured],
                         "first_lag_order": relation_diagnostics(measured[0].maximum),
                         "accumulated_order": relation_diagnostics(accumulated_relation(measured))})
    return {"counts": dict(sorted(counts.items())), "fixtures": fixtures,
            "case_digest_sha256": digest.hexdigest()}


def verify_named_obstructions():
    systems = {s.name: s for s in fixture_systems()}
    chain = profiles(systems["chain"], 4)
    require(not relation_diagnostics(chain[0].maximum)["transitive"], "one-step chain is intransitive")
    require(relation_diagnostics(accumulated_relation(chain))["strict_partial_order"], "chain accumulation")

    diamond = systems["xor_diamond"]
    diamond_profiles = profiles(diamond, 4)
    drel = relation_diagnostics(accumulated_relation(diamond_profiles))
    require((0, 3) in drel["closure_adds_pairs"], "cancelling structural path witness")
    require(diamond_profiles[0].maximum[0][1] == 1 and diamond_profiles[0].maximum[1][3] == 1,
            "two composable event response pairs")
    require(diamond_profiles[1].maximum[0][3] == 0, "two equal paths cancel")
    plain = [((simulate(diamond, initial, 2) >> 3) & 1) for initial in (0, 1)]
    controlled = [((simulate(diamond, initial, 2, {1: {2: 0}}) >> 3) & 1) for initial in (0, 1)]
    require(plain == [0, 0] and controlled == [0, 1], "operation-class change reveals path")

    state_effects = {}
    for name in ("and_context", "or_context"):
        system = systems[name]
        values = []
        for other in (0, 1):
            y0 = (system.step(other << 1) >> 2) & 1
            y1 = (system.step((other << 1) | 1) >> 2) & 1
            values.append(y1 - y0)
        state_effects[name] = values
    require(state_effects == {"and_context": [0, 1], "or_context": [1, 0]}, "context dependence")

    mask = profiles(systems["xor_mask"], 1)[0]
    require(mask.maximum[0][2] == 1 and mask.absolute_counts[0][2] == mask.contexts_per_source,
            "every XOR context responds")
    require(mask.signed_counts[0][2] == 0, "marginal XOR effect cancels")
    marginal = []
    for setting in (0, 1):
        values = [(systems["xor_mask"].step(setting | (other << 1)) >> 2) & 1 for other in (0, 1)]
        marginal.append(str(Fraction(sum(values), len(values))))
    require(marginal == ["1/2", "1/2"], "identical target marginals")

    swap = relation_diagnostics(profiles(systems["feedback_swap"], 1)[0].maximum)
    require(swap["cycle_registers"] == [0, 1], "feedback register cycle")

    causal = systems["causal_observation_model"]
    common = systems["common_cause_observation_model"]
    baseline = []
    do_distributions = {}
    for model in (causal, common):
        require(all(model.step(7 * u) == 7 * u for u in (0, 1)), "unperturbed observational fixed points")
        observed = [((model.step(7 * u) >> 1) & 1, (model.step(7 * u) >> 2) & 1) for u in (0, 1)]
        baseline.append(observed)
        forced = [((model.step((7 * u) & ~(1 << 1)) >> 2) & 1) for u in (0, 1)]
        do_distributions[model.name] = str(Fraction(sum(forced), 2))
    require(baseline[0] == baseline[1] == [(0, 0), (1, 1)], "observational alias")
    require(do_distributions == {"causal_observation_model": "0", "common_cause_observation_model": "1/2"},
            "intervention resolves observational alias")
    return {
        "one_step_chain": {"transitive": False, "accumulated_strict_partial_order": True},
        "xor_diamond": {"structural_path": [0, 1, 3], "second_path": [0, 2, 3],
                        "source_zero_one_outputs_at_lag_2": plain,
                        "same_outputs_with_middle_register_2_clamped": controlled,
                        "event_transitivity_witness": {
                            "event_format": "[register, tick]",
                            "responding_pairs": [[[0, 0], [1, 1]], [[1, 1], [3, 2]]],
                            "absent_composite_pair": [[0, 0], [3, 2]]},
                        "time_collapsed_relation_transitive": drel["transitive"]},
        "context_effects_for_other_register_0_then_1": state_effects,
        "xor_mask": {"maximum_response": 1, "mean_absolute_response": "1",
                     "mean_signed_response": "0", "target_one_probability_under_do_0_then_1": marginal},
        "feedback_swap": {"register_cycle": [0, 1, 0],
                          "event_time_order": "tick labels are supplied and every update advances one tick"},
        "observational_alias": {"observed_support_A_B": baseline[0], "equal_support_probabilities": ["1/2", "1/2"],
                               "P_B_one_after_do_A_zero": do_distributions,
                               "total_variation_after_intervention": "1/2"},
    }


def verify_horizon_ambiguity():
    rows = []
    for horizon in (1, 2, 3, 4, 6, 8):
        width = horizon + 2
        updates = (const(0), *(var(i - 1) for i in range(1, width)))
        connected = BooleanSystem(f"delay_chain_{horizon}", updates)
        severed = BooleanSystem(f"severed_delay_chain_{horizon}", updates[:-1] + (const(0),))
        traces = []
        for model in (connected, severed):
            traces.append([[((simulate(model, source, lag) >> (width - 1)) & 1)
                            for lag in range(1, horizon + 2)] for source in (0, 1)])
        for source in (0, 1):
            require(traces[0][source][:horizon] == traces[1][source][:horizon], "finite probe alias")
        require(traces[0][1][horizon] == 1 and traces[1][1][horizon] == 0, "one later probe separates")
        rows.append({"observed_horizon": horizon, "registers": width,
                     "intervened_register": 0, "observed_register": width - 1,
                     "other_initial_registers": "all zero",
                     "candidate_count_after_observed_horizon": 2,
                     "candidate_count_after_next_tick": 1,
                     "connected_traces_do_0_then_1": traces[0],
                     "severed_traces_do_0_then_1": traces[1]})
    return {"protocol": "set register 0 at tick 0; initialize others to zero; observe only last register",
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("receipt.json"))
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    root = Path(__file__).resolve().parent
    result = {
        "experiment": "PGC-01", "schema_version": 1,
        "scope": "exact finite operational response and identifiability with supplied registers and ticks",
        "affine_family": verify_all_affine(),
        "named_fixtures": verify_fixtures_and_relabelings(),
        "obstructions": verify_named_obstructions(),
        "finite_horizon_ambiguity": verify_horizon_ambiguity(),
        "assumptions": json.loads((root / "ASSUMPTIONS.json").read_text()),
        "source_sha256": {name: sha256((root / name).read_bytes()).hexdigest()
                          for name in ("pgc01.py", "verify_pgc01.py", "FIXTURES.json", "ASSUMPTIONS.json")},
        "status": "PASS",
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        expected = json.loads(args.check.read_text())
        require(json.loads(serialized) == expected, "canonical receipt mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    runtime = {"elapsed_seconds": perf_counter() - started, "timing_is_noncanonical": True}
    args.output.with_suffix(".runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "affine_counts": result["affine_family"]["counts"],
                      "fixture_counts": result["named_fixtures"]["counts"],
                      "ambiguity_cases": len(result["finite_horizon_ambiguity"]["cases"]),
                      **runtime}, indent=2))


if __name__ == "__main__":
    main()
