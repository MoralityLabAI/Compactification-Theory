"""NOR-23: preregistered exact coordinate audit of source-locked RRC25 futures.

Two information regimes:
  blind: fixed modified coordinates BEFORE independent random trusted audit;
  informed: attacker learns the audit coordinates before picking modifications.
A trusted witness reads genuine target coefficients, available only after
the public-only program proposal is fixed. Every reported false acceptance
is explicitly out-of-family. See REGISTRATION.md committed first.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import comb, sqrt
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21

SCHEMA = "nor23.randomized_exact_coordinate_audit.v1"
FIXTURE_SCHEMA = "nor23.fresh_public_source_only.v1"
SOURCE_OFFSET = 94323000
ATTACK_SEED = 202611230071
CHALLENGE_SEED = 202611230072
REPLICATES = 2
DEGREES = (512, 1024)
FAMILIES = nor21.FAMILIES
BUDGETS = (0, 1, 4, 16, 64)
RADII = (1, 4, 16)
MAGNITUDES = (1, 4294049777)
TRIALS = 4096
SOURCE_COUNT = REPLICATES * len(DEGREES) * len(FAMILIES)
MAX_ATTEMPTS = 100


def canonical(data):
    return (json.dumps(data, indent=2, sort_keys=True,
                       ensure_ascii=True) + "\n").encode()


def source_seed(rep, degree, family):
    if rep not in (0, 1) or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("outside frozen NOR23 source roster")
    return (corpus.SEED + SOURCE_OFFSET + 1000003 * rep +
            100009 * DEGREES.index(degree) +
            10007 * FAMILIES.index(family))


def coordinate_seed(which, digest, size, k, ordinal=0):
    if which not in (ATTACK_SEED, CHALLENGE_SEED):
        raise ValueError("invalid random stream")
    payload = f"{which}|{digest}|{size}|{k}|{ordinal}".encode("ascii")
    return int.from_bytes(sha256(payload).digest(), "big")


def positions(which, digest, size, k, ordinal=0):
    if not (0 <= k <= size):
        raise ValueError("invalid coordinate budget")
    rng = random.Random(coordinate_seed(which, digest, size, k, ordinal))
    return tuple(sorted(rng.sample(range(size), k)))


def blind_positions(digest, size, radius):
    if radius not in RADII:
        raise ValueError("unregistered attack radius")
    return frozenset(positions(ATTACK_SEED, digest, size, radius))


def challenge_positions(digest, size, budget, trial):
    if budget not in BUDGETS or not 0 <= trial < TRIALS:
        raise ValueError("unregistered sample budget or trial")
    return positions(CHALLENGE_SEED, digest, size, budget, trial)


def informed_positions(sampled, size, radius):
    """Adversary sees sampled coordinates and chooses unchecked ones."""
    seen = set(sampled)
    result = []
    for i in range(size):
        if i not in seen:
            result.append(i)
            if len(result) == radius:
                break
    if len(result) != radius:
        raise ValueError("adaptive attack needs radius <= unchecked count")
    return frozenset(result)


def trusted_witness_accepts(proposed, true_future, audited, changed, delta):
    """Exact target coefficients at audit indices; target not in public policy.

    This sparse oracle is extensionally the real target tuple modified by
    nonzero delta at 'changed', without materializing many huge integers.
    """
    if len(proposed) != len(true_future):
        raise ValueError("target sizes disagree")
    if delta == 0:
        raise ValueError("alteration must be nonzero")
    for i in audited:
        if proposed[i] != true_future[i] + (delta if i in changed else 0):
            return False
    return True


def detection_probability(size, radius, budget):
    if not (0 <= radius <= size and 0 <= budget <= size):
        raise ValueError("invalid population")
    if budget > size - radius:
        return 1.0
    return 1.0 - comb(size - radius, budget) / comb(size, budget)


def generate_source(rep, degree, family, known):
    """Generator may construct program, but only public fields are serialized."""
    query_degree = degree * 3 // 2
    rng = random.Random(source_seed(rep, degree, family))
    for attempt in range(MAX_ATTEMPTS):
        true_program = nor21.new_header_program(rng)
        demos = tuple((src := nor21.public_source(rng, family, query_degree),
                       corpus.execute(src, true_program)) for _ in range(2))
        query = nor21.public_source(rng, family, query_degree)
        public = corpus.Case("nor23-generating", family, demos, query, (), None)
        headers = corpus.possible_headers(public)
        if len(headers) != 1:
            continue
        header = headers[0]
        if ((header.base, header.delay, header.spacing, header.amplitude)
                != true_program[:4]):
            continue
        digest = nor21.public_digest(public, degree)
        if digest in known:
            continue
        mask = nor21.native_mask(public, header, degree)
        if mask == 0:
            continue
        preds = nor21.candidate_predictions(query, header, query_degree)
        if len({preds[j] for j in range(16) if mask & (1 << j)}) != 1:
            continue
        planted = 4 * corpus.OPS.index(true_program[4]) + \
                  corpus.OPS.index(true_program[5])
        if not mask & (1 << planted):
            raise AssertionError("genuine program outside accepted mask")
        known.add(digest)
        return {
            "replicate": rep, "observation_degree": degree,
            "query_degree": query_degree, "family": family,
            "source_seed": source_seed(rep, degree, family),
            "selected_attempt": attempt,
            "full_public_input_sha256": digest,
            "public_header": {
                "base": header.base, "delay": header.delay,
                "spacing": header.spacing, "amplitude": header.amplitude,
            },
            "demos": [[list(src[:degree + 1]), list(y[:degree + 1])]
                      for src, y in demos],
            "public_query_input": list(query),
            "original_public_accept_mask": mask,
        }
    raise RuntimeError("NOR23 selected no compatible source within 100 attempts")


def make_fixture():
    seen = set()
    cases = [generate_source(rep, degree, family, seen)
             for rep in range(REPLICATES)
             for degree in DEGREES
             for family in FAMILIES]
    assert len(cases) == SOURCE_COUNT == len(seen)
    assert Counter(r["observation_degree"] for r in cases) == {512: 6, 1024: 6}
    assert Counter(r["family"] for r in cases) == {f: 4 for f in FAMILIES}
    return {
        "schema": FIXTURE_SCHEMA,
        "registered_source_offset": SOURCE_OFFSET,
        "public_only": True,
        "hidden_true_future_and_program_absent": True,
        "trusted_witness_values_absent": True,
        "registered_budgets": list(BUDGETS),
        "registered_radii": list(RADII),
        "registered_magnitudes": list(MAGNITUDES),
        "registered_trials": TRIALS,
        "cases": cases,
    }


def public_case(row):
    degree = row["observation_degree"]
    assert degree in DEGREES and row["query_degree"] == 3 * degree // 2
    h = row["public_header"]
    header = corpus.Header(h["base"], h["delay"], h["spacing"], h["amplitude"])
    demos = tuple((tuple(f), tuple(y)) for f, y in row["demos"])
    query = tuple(row["public_query_input"])
    assert len(demos) == 2 and all(
        len(f) == len(y) == degree + 1 for f, y in demos)
    assert len(query) == row["query_degree"] + 1
    public = corpus.Case("nor23-public-only", row["family"],
                         demos, query, (), None)
    assert public.program is None and public.future == ()
    assert nor21.public_digest(public, degree) == row["full_public_input_sha256"]
    return public, header


def frozen_fixture(path):
    raw = Path(path).read_bytes()
    fixture = json.loads(raw)
    assert fixture["schema"] == FIXTURE_SCHEMA
    assert fixture["registered_source_offset"] == SOURCE_OFFSET
    assert fixture["public_only"]
    assert fixture["hidden_true_future_and_program_absent"]
    assert fixture["trusted_witness_values_absent"]
    assert fixture["registered_budgets"] == list(BUDGETS)
    assert fixture["registered_radii"] == list(RADII)
    assert fixture["registered_magnitudes"] == list(MAGNITUDES)
    assert fixture["registered_trials"] == TRIALS
    assert len(fixture["cases"]) == SOURCE_COUNT
    assert len(set(r["full_public_input_sha256"] for r in fixture["cases"])) == SOURCE_COUNT
    return fixture, sha256(raw).hexdigest()


def sealed_future_after_choice(row, frozen_prediction):
    """Private reconstruction called only AFTER public prediction is frozen."""
    rep, degree, family = (row["replicate"], row["observation_degree"], row["family"])
    rng = random.Random(source_seed(rep, degree, family))
    program = query = demos = None
    assert 0 <= row["selected_attempt"] < MAX_ATTEMPTS
    for _ in range(row["selected_attempt"] + 1):
        program = nor21.new_header_program(rng)
        demos = tuple((src := nor21.public_source(rng, family, 3 * degree // 2),
                       corpus.execute(src, program)) for _ in range(2))
        query = nor21.public_source(rng, family, 3 * degree // 2)
    assert program is not None
    assert row["source_seed"] == source_seed(rep, degree, family)
    assert row["demos"] == [[list(f[:degree + 1]), list(y[:degree + 1])]
                            for f, y in demos]
    assert row["public_query_input"] == list(query)
    actual = corpus.execute(query, program)
    if actual != frozen_prediction:
        raise AssertionError("in-family public prediction disagreed with sealed truth")
    return actual


def target_digest(coeffs):
    return sha256(json.dumps(coeffs, separators=(",", ":")).encode()).hexdigest()


def run(fixture_path):
    fixture, fixture_digest = frozen_fixture(fixture_path)
    source_records = []
    aggregated = defaultdict(lambda: {"detected": 0, "missed": 0, "adaptive_missed": 0,
                                      "valid_pass": 0, "valid_rejected": 0, "trials": 0})
    for row in fixture["cases"]:
        public, header = public_case(row)
        degree, query_degree = row["observation_degree"], row["query_degree"]
        digest = row["full_public_input_sha256"]
        proposal = nor21.certify(public, header, degree, query_degree, "stream16")
        assert proposal["mask"] == row["original_public_accept_mask"]
        assert proposal["unanimous"] and proposal["chosen_class"] is not None
        chosen = proposal["chosen_class"]
        predicted = corpus.execute(public.query, nor21.program(header, chosen))
        # All public choices are fixed before accessing the true future.
        frozen = (proposal["mask"], chosen, target_digest(predicted))
        truth = sealed_future_after_choice(row, predicted)
        assert target_digest(truth) == frozen[2]
        size = len(truth)
        attacks = {r: blind_positions(digest, size, r) for r in RADII}
        audits = []
        for budget in BUDGETS:
            counts = {(r, delta): {"blind_detected": 0, "blind_false_accept": 0,
                                  "adaptive_detected": 0, "adaptive_false_accept": 0}
                      for r in RADII for delta in MAGNITUDES}
            valid_pass = valid_rejected = 0
            challenge_ns = compare_ns = 0
            for trial in range(TRIALS):
                t0 = time.perf_counter_ns()
                checked = challenge_positions(digest, size, budget, trial)
                t1 = time.perf_counter_ns()
                challenge_ns += t1 - t0
                assert len(checked) == budget
                # Publisher truth is the original valid generator future.
                if trusted_witness_accepts(predicted, truth, checked, frozenset(), 1):
                    valid_pass += 1
                else:
                    valid_rejected += 1
                for radius in RADII:
                    blind = attacks[radius]
                    informed = informed_positions(checked, size, radius)
                    for delta in MAGNITUDES:
                        blind_pass = trusted_witness_accepts(
                            predicted, truth, checked, blind, delta)
                        adaptive_pass = trusted_witness_accepts(
                            predicted, truth, checked, informed, delta)
                        cell = counts[(radius, delta)]
                        cell["blind_false_accept" if blind_pass else "blind_detected"] += 1
                        cell["adaptive_false_accept" if adaptive_pass
                             else "adaptive_detected"] += 1
                compare_ns += time.perf_counter_ns() - t1
            assert valid_pass == TRIALS and valid_rejected == 0
            for radius in RADII:
                for delta in MAGNITUDES:
                    v = counts[(radius, delta)]
                    assert v["blind_false_accept"] + v["blind_detected"] == TRIALS
                    assert v["adaptive_false_accept"] == TRIALS
                    assert v["adaptive_detected"] == 0
                    key = (degree, budget, radius, delta)
                    agg = aggregated[key]
                    agg["detected"] += v["blind_detected"]
                    agg["missed"] += v["blind_false_accept"]
                    agg["adaptive_missed"] += v["adaptive_false_accept"]
                    agg["valid_pass"] += valid_pass
                    agg["valid_rejected"] += valid_rejected
                    agg["trials"] += TRIALS
            audits.append({
                "budget": budget, "valid_witness_pass": valid_pass,
                "valid_witness_rejected": valid_rejected,
                "challenge_sample_ns": challenge_ns,
                "all_12_audit_comparisons_ns": compare_ns,
                "outcomes": [
                    {"radius": r, "magnitude": delta, **counts[(r, delta)]}
                    for r in RADII for delta in MAGNITUDES],
            })
        source_records.append({
            "replicate": row["replicate"], "degree": degree,
            "query_degree": query_degree, "family": row["family"],
            "public_sha256": digest, "frozen_public_mask": frozen[0],
            "frozen_chosen_class": frozen[1],
            "frozen_predicted_future_sha256": frozen[2],
            "valid_true_future_sha256_postchoice": target_digest(truth),
            "blind_attack_positions": {str(r): sorted(attacks[r]) for r in RADII},
            "audits": audits,
        })
    degree_cells = []
    primary_pass = True
    for (degree, budget, radius, delta), counts in sorted(aggregated.items()):
        n = counts["trials"]
        m = degree * 3 // 2 + 1
        p = detection_probability(m, radius, budget)
        observed = counts["detected"] / n
        tolerance = 5 * sqrt(p * (1 - p) / n) + 0.015
        sane = abs(observed - p) <= tolerance
        primary_pass &= sane and counts["adaptive_missed"] == n and (
            counts["valid_pass"] == n and counts["valid_rejected"] == 0)
        degree_cells.append({
            "degree": degree, "output_coordinates": m, "budget": budget,
            "altered_coordinates": radius, "nonzero_delta": delta,
            "source_count": REPLICATES * len(FAMILIES), "trials": n,
            "exact_uniform_detection_probability": p,
            "observed_blind_detection_probability": observed,
            "blind_detection_tolerance": tolerance,
            "blind_monte_carlo_within_registered_tolerance": sane,
            "blind_detection": counts["detected"],
            "blind_false_accept": counts["missed"],
            "adaptive_false_accept": counts["adaptive_missed"],
            "valid_accept": counts["valid_pass"],
            "valid_reject": counts["valid_rejected"],
        })
    assert len(degree_cells) == 2 * 5 * 3 * 2
    assert len(source_records) == SOURCE_COUNT
    return {
        "schema": SCHEMA, "fixture_sha256": fixture_digest,
        "distinct_public_source_count": SOURCE_COUNT,
        "source_selection_freezes_before_true_future_access": True,
        "trusted_witness_supplies_exogenous_true_target_coefficients": True,
        "blind_modification_positions_fixed_before_challenge": True,
        "adaptive_modification_positions_chosen_after_challenge": True,
        "attack_seed": ATTACK_SEED, "challenge_seed": CHALLENGE_SEED,
        "sampled_challenges_per_source_budget": TRIALS,
        "budgets": list(BUDGETS), "radii": list(RADII),
        "magnitudes": list(MAGNITUDES),
        "source_records": source_records,
        "degree_cells": degree_cells,
        "registered_criteria_pass": bool(primary_pass),
        "scope": ("Synthetic exact RRC25 finite-series outputs; assumed authentic "
                  "external coordinate witness. Randomized challenge does not "
                  "authenticate the witness or guarantee full future correctness "
                  "for k < m. Replications share source and challenge identities."),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--make-fixture", type=Path)
    p.add_argument("--fixture", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.make_fixture:
        if args.fixture:
            p.error("fixture creation and analysis are separate stages")
        raw = canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True, exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR23 PUBLIC ONLY FIXTURE", len(raw), sha256(raw).hexdigest())
        return
    if args.fixture is None:
        p.error("use --fixture for grading, --make-fixture for creation")
    result = run(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(canonical(result))
    print("NOR23 AUDIT BUDGET VERDICT", result["registered_criteria_pass"],
          "SOURCES", result["distinct_public_source_count"],
          "CELLS", len(result["degree_cells"]))
    for c in result["degree_cells"]:
        if c["nonzero_delta"] == 1:
            print("NOR23_CELL", c["degree"], c["budget"], c["altered_coordinates"],
                  c["blind_detection"], c["trials"],
                  round(c["exact_uniform_detection_probability"], 6))
    if args.check:
        assert result["registered_criteria_pass"]


if __name__ == "__main__":
    main()
