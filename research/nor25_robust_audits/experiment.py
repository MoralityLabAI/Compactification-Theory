"""NOR-25 source-locked robustness test. Registration precedes this implementation.

The NOR-24 frozen prior is an input. All NOR-25 source identities are fresh.
The external witness is exact; SHA256-seeded trials are reproducible and
not claimed to supply cryptographic unpredictability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import islice
from math import comb
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.nor24_adaptive_audits import experiment as prior24
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

SCHEMA = "nor25.frozen_prior_robust_blind_coordinate_audit.v1"
FIXTURE_SCHEMA = "nor25.source_only_12_fresh_test.v1"
SOURCE_OFFSET = 99325000
REPLICATES = (4, 5)
DEGREES = prior24.DEGREES
FAMILIES = nor21.FAMILIES
SOURCE_COUNT = len(REPLICATES) * len(DEGREES) * len(FAMILIES)
KINDS = prior24.KINDS
ALL_KINDS = prior24.ALL_KINDS
PRIOR_NUMERATORS = (295, 246, 153, 78)
PRIOR_DENOMINATOR = 772
PRIOR = {kind: num / PRIOR_DENOMINATOR
         for kind, num in zip(KINDS, PRIOR_NUMERATORS)}
EVAL_WEIGHTS = prior24.PRIOR_WEIGHTS
RADIUS = (1, 4, 16)
BUDGETS = (4, 16, 64)
TRIALS = 1024
DELTAS = (1, 4294049777)
ATTACK_STREAM = 202611250072
AUDIT_STREAM = 202611250073
MIXED_FRACTIONS = {"hybrid_25": 1, "hybrid_50": 2, "hybrid_75": 3}
ARMS = ("uniform", "static_prior", "posterior", "shuffled_prior",
        "hybrid_25", "hybrid_50", "hybrid_75", "oracle_support")


def source_seed(rep, degree, family):
    if rep not in REPLICATES or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("source outside NOR-25 registration")
    return (corpus.SEED + SOURCE_OFFSET + 1000003 * rep +
            100009 * DEGREES.index(degree) +
            10007 * FAMILIES.index(family))


def generate_source(rep, degree, family, seen):
    q = 3 * degree // 2
    rng = random.Random(source_seed(rep, degree, family))
    for attempt in range(100):
        planted = nor21.new_header_program(rng)
        demos = tuple((f := nor21.public_source(rng, family, q),
                       corpus.execute(f, planted)) for _ in range(2))
        query = nor21.public_source(rng, family, q)
        public = corpus.Case("nor25-public-source", family, demos, query, (), None)
        headers = corpus.possible_headers(public)
        if len(headers) != 1:
            continue
        header = headers[0]
        if (header.base, header.delay, header.spacing, header.amplitude) != planted[:4]:
            continue
        digest = nor21.public_digest(public, degree)
        if digest in seen:
            continue
        mask = nor21.native_mask(public, header, degree)
        if mask == 0:
            continue
        preds = nor21.candidate_predictions(query, header, q)
        if len({preds[j] for j in range(16) if mask & (1 << j)}) != 1:
            continue
        planted_j = corpus.OPS.index(planted[4]) * 4 + corpus.OPS.index(planted[5])
        assert mask & (1 << planted_j)
        seen.add(digest)
        return {
            "split": "TEST", "replicate": rep,
            "observation_degree": degree, "query_degree": q,
            "family": family, "source_seed": source_seed(rep, degree, family),
            "selected_attempt": attempt, "full_public_input_sha256": digest,
            "public_header": {"base": header.base, "delay": header.delay,
                              "spacing": header.spacing, "amplitude": header.amplitude},
            "demos": [[list(x[:degree+1]), list(y[:degree+1])]
                      for x, y in demos],
            "public_query_input": list(query),
            "original_public_accept_mask": mask,
        }
    raise RuntimeError("NOR25 exhausted first-100 deterministic source candidates")


def make_fixture():
    seen = set()
    rows = [generate_source(rep, degree, family, seen)
            for rep in REPLICATES for degree in DEGREES for family in FAMILIES]
    assert len(seen) == len(rows) == SOURCE_COUNT == 12
    return {
        "schema": FIXTURE_SCHEMA, "source_offset": SOURCE_OFFSET,
        "replicates": list(REPLICATES), "source_only": True,
        "private_truth_absent": True, "attack_sets_absent": True,
        "frozen_prior_numerators": list(PRIOR_NUMERATORS),
        "frozen_prior_denominator": PRIOR_DENOMINATOR,
        "challenge_trials": TRIALS, "cases": rows,
    }


def load_fixture(path):
    data = Path(path).read_bytes()
    fixture = __import__("json").loads(data)
    assert fixture["schema"] == FIXTURE_SCHEMA
    assert fixture["source_offset"] == SOURCE_OFFSET
    assert fixture["replicates"] == list(REPLICATES)
    assert fixture["source_only"] and fixture["private_truth_absent"]
    assert fixture["attack_sets_absent"]
    assert fixture["frozen_prior_numerators"] == list(PRIOR_NUMERATORS)
    assert fixture["frozen_prior_denominator"] == PRIOR_DENOMINATOR
    assert fixture["challenge_trials"] == TRIALS
    rows = fixture["cases"]
    assert len(rows) == SOURCE_COUNT
    assert len({r["full_public_input_sha256"] for r in rows}) == SOURCE_COUNT
    assert {(r["replicate"], r["observation_degree"], r["family"]) for r in rows} == {
        (rep, degree, family) for rep in REPLICATES
        for degree in DEGREES for family in FAMILIES}
    assert all(r["split"] == "TEST" and
               not any(w in r for w in ("program", "future", "attack", "witness"))
               for r in rows)
    return fixture, sha256(data).hexdigest()


def sealed_truth_after_public_proposal(row, predicted):
    """Reconstruct private original program after the public proposal is fixed."""
    rep, d, family = row["replicate"], row["observation_degree"], row["family"]
    rng = random.Random(source_seed(rep, d, family))
    program = demos = query = None
    for _ in range(row["selected_attempt"] + 1):
        program = nor21.new_header_program(rng)
        demos = tuple((x := nor21.public_source(rng, family, 3*d//2),
                       corpus.execute(x, program)) for _ in range(2))
        query = nor21.public_source(rng, family, 3*d//2)
    assert row["source_seed"] == source_seed(rep, d, family)
    assert row["demos"] == [[list(x[:d+1]), list(y[:d+1])] for x, y in demos]
    assert row["public_query_input"] == list(query)
    truth = corpus.execute(query, program)
    assert tuple(predicted) == tuple(truth)
    return truth


def blind_attack(digest, kind, radius, regions):
    if kind not in ALL_KINDS or radius not in RADIUS:
        raise ValueError("unregistered blind attack")
    rng = random.Random(prior24.stream_seed(ATTACK_STREAM, digest, kind, radius))
    return frozenset(rng.sample(sorted(regions[kind]), radius))


def reserve_size(policy, budget):
    if policy not in MIXED_FRACTIONS or budget not in BUDGETS:
        raise ValueError("unregistered NOR25 allocation")
    return budget * MIXED_FRACTIONS[policy] // 4


def hybrid_schedule(static_order, m, digest, policy, budget, trial):
    """Fixed risk prefix T, then uniform size-u sample from U minus T."""
    u = reserve_size(policy, budget)
    t = budget-u
    top = tuple(static_order[:t])
    assert len(set(top)) == t
    chosen = set(top)
    remaining = [i for i in range(m) if i not in chosen]
    rng = random.Random(prior24.stream_seed(
        AUDIT_STREAM, digest, policy, budget, trial))
    tail = tuple(rng.sample(remaining, u))
    plan = top + tail
    assert len(plan) == len(set(plan)) == budget
    return plan


def policy_schedules(regions, digest, trial):
    """Baseline controls preserve NOR24 policies; hybrids are independent."""
    baseline = prior24.policy_schedules(regions, PRIOR, digest, trial)
    m = len(regions["uniform"])
    results = {}
    for budget in BUDGETS:
        for arm in ARMS:
            if arm in MIXED_FRACTIONS:
                plan = hybrid_schedule(baseline["static_prior"], m, digest,
                                       arm, budget, trial)
            elif arm == "oracle_support":
                for kind in ALL_KINDS:
                    results[(arm, kind, budget)] = baseline[(arm, kind)][:budget]
                continue
            else:
                plan = baseline[arm][:budget]
            assert len(plan) == len(set(plan)) == budget
            results[(arm, budget)] = plan
    return results


def blind_floor(m, budget, reserve, radius):
    """The exact minimax blind detection probability for a fixed risk prefix."""
    t = budget-reserve
    if not (0 <= t < m and 0 <= reserve <= m-t and 1 <= radius <= m-t):
        raise ValueError("out-of-domain blind guarantee")
    return Fraction(1) - Fraction(comb(m-t-radius, reserve), comb(m-t, reserve))


def exact_detection(proposed, true, inspected, changed, delta):
    return not prior24.exact_witness_accepts(
        proposed, true, inspected, changed, delta)


def evaluate(path):
    fixture, fixture_hash = load_fixture(path)
    # Freeze ALL source-only proposals before restoring any private targets.
    proposals = {row["full_public_input_sha256"]: prior24.public_proposal(row)
                 for row in fixture["cases"]}
    assert len(proposals) == SOURCE_COUNT
    records = []
    for row in fixture["cases"]:
        digest = row["full_public_input_sha256"]
        proposal = proposals[digest]
        true = sealed_truth_after_public_proposal(row, proposal["predicted"])
        predicted = proposal["predicted"]
        regions = prior24.supports(predicted, proposal["header"])
        # Fix blind attacks for this public source BEFORE any audit plans.
        attacks = {(kind, radius): blind_attack(digest, kind, radius, regions)
                   for kind in ALL_KINDS for radius in RADIUS}
        cells = {(kind, radius, budget, arm): 0
                 for kind in ALL_KINDS for radius in RADIUS
                 for budget in BUDGETS for arm in ARMS}
        informed_fails = 0
        valid_target_fails = 0
        magnitude_disagreements = 0
        for trial in range(TRIALS):
            plans = policy_schedules(regions, digest, trial)
            for kind in ALL_KINDS:
                for budget in BUDGETS:
                    for arm in ARMS:
                        checked = plans[(arm, kind, budget)] if arm == "oracle_support" \
                                  else plans[(arm, budget)]
                        if not prior24.exact_witness_accepts(
                                predicted, true, checked, frozenset(), DELTAS[0]):
                            valid_target_fails += 1
                        # Adaptive attacker sees the exact realized schedule.
                        for radius in RADIUS:
                            attack = attacks[(kind, radius)]
                            d1 = exact_detection(predicted, true, checked, attack, DELTAS[0])
                            d2 = exact_detection(predicted, true, checked, attack, DELTAS[1])
                            magnitude_disagreements += d1 != d2
                            cells[(kind, radius, budget, arm)] += int(d1)
                            informed = prior24.informed_attack(
                                checked, regions[kind], radius)
                            informed_fails += exact_detection(
                                predicted, true, checked, informed, DELTAS[0])
        assert valid_target_fails == informed_fails == magnitude_disagreements == 0
        observations = [
            {"kind": kind, "radius": radius, "budget": budget,
             "policy": arm, "detected": cells[(kind, radius, budget, arm)],
             "missed": TRIALS - cells[(kind, radius, budget, arm)]}
            for kind in ALL_KINDS for radius in RADIUS
            for budget in BUDGETS for arm in ARMS
        ]
        floors = {}
        for arm in MIXED_FRACTIONS:
            for budget in BUDGETS:
                for radius in RADIUS:
                    u = reserve_size(arm, budget)
                    bound = blind_floor(len(predicted), budget, u, radius)
                    floors[f"{arm}|{budget}|{radius}"] = [
                        bound.numerator, bound.denominator]
        records.append({
            "public_sha256": digest, "replicate": row["replicate"],
            "observation_degree": row["observation_degree"],
            "family": row["family"], "public_mask": proposal["mask"],
            "chosen_class": proposal["chosen_class"],
            "public_prediction_sha256": proposal["sha256"],
            "sealed_true_future_sha256_postchoice": prior24.sha_target(true),
            "support_sizes": {kind: len(regions[kind]) for kind in ALL_KINDS},
            "blind_attack_positions": {
                f"{kind}|{radius}": sorted(attacks[(kind, radius)])
                for kind in ALL_KINDS for radius in RADIUS},
            "theoretical_blind_floors": floors,
            "valid_target_failures": valid_target_fails,
            "informed_attack_detected": informed_fails,
            "delta_invariance_failures": magnitude_disagreements,
            "observations": observations,
        })
    primary = []
    for rec in records:
        d = {(x["kind"], x["radius"], x["budget"], x["policy"]): x
             for x in rec["observations"]}
        weighted = {
            arm: sum(weight*d[(kind,4,16,arm)]["detected"]/TRIALS
                     for kind, weight in zip(KINDS, EVAL_WEIGHTS))
            for arm in ARMS
        }
        off = {arm: d[(prior24.STRESS,4,16,arm)]["detected"]/TRIALS
               for arm in ARMS}
        primary.append({
            "public_sha256": rec["public_sha256"],
            "degree": rec["observation_degree"], "family": rec["family"],
            "weighted_in_template_rates": weighted,
            "off_template_rates": off,
            "hybrid50_minus_uniform": weighted["hybrid_50"]-weighted["uniform"],
            "hybrid50_minus_static": weighted["hybrid_50"]-weighted["static_prior"],
            "hybrid50_off_minus_static": off["hybrid_50"]-off["static_prior"],
        })
    wins_uniform = sum(x["hybrid50_minus_uniform"] > 0 for x in primary)
    wins_off = sum(x["hybrid50_off_minus_static"] > 0 for x in primary)
    mean_gain = statistics.mean(x["hybrid50_minus_uniform"] for x in primary)
    mean_off = statistics.mean(x["off_template_rates"]["hybrid_50"] for x in primary)
    primary_pass = (wins_uniform >= 9 and mean_gain >= .10 and
                    wins_off >= 10 and mean_off >= .02)
    return {
        "schema": SCHEMA, "registration_precedes_all_nor25_data": True,
        "fixture_sha256": fixture_hash, "source_count": SOURCE_COUNT,
        "challenge_trials_per_source_policy": TRIALS,
        "frozen_prior": PRIOR,
        "frozen_prior_numerators": list(PRIOR_NUMERATORS),
        "frozen_prior_denominator": PRIOR_DENOMINATOR,
        "blind_attacks_fixed_before_policy_challenges": True,
        "public_proposals_frozen_before_private_truth": True,
        "source_records": records, "primary_by_source": primary,
        "wins_hybrid50_vs_uniform": wins_uniform,
        "offtemplate_wins_hybrid50_vs_static": wins_off,
        "source_mean_weighted_gain_vs_uniform": mean_gain,
        "source_mean_offtemplate_hybrid50": mean_off,
        "registered_primary_pass": primary_pass,
        "scope": ("Synthetic fixed blind attacks with exact exogenous target witnesses; "
                  "seeded experimental reproducibility is not production unpredictability; "
                  "fully informed adversaries evade any partial inspection."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-fixture", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:
            parser.error("source generation and private assessment are separate stages")
        raw = prior24.canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True, exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR25 FROZEN SOURCE-ONLY FIXTURE", len(raw), sha256(raw).hexdigest())
        return
    if args.fixture is None:
        parser.error("--fixture required")
    outcome = evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(prior24.canonical(outcome))
    print("NOR25 IN-TEMPLATE BY SOURCE", [
        (x["degree"], x["family"], round(x["hybrid50_minus_uniform"], 6),
         round(x["hybrid50_minus_static"], 6),
         round(x["hybrid50_off_minus_static"], 6))
        for x in outcome["primary_by_source"]])
    print("NOR25 PRIMARY", outcome["registered_primary_pass"],
          outcome["wins_hybrid50_vs_uniform"],
          outcome["offtemplate_wins_hybrid50_vs_static"],
          round(outcome["source_mean_weighted_gain_vs_uniform"],6),
          round(outcome["source_mean_offtemplate_hybrid50"],6))


if __name__ == "__main__":
    main()
