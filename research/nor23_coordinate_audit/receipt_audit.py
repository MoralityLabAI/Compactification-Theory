"""Independent NOR-23 source, hidden truth and sampling replay auditor.

This script does not import NOR-23's experiment, its oracle, its PRNG
helpers or its grading logic. It uses the independent original RRC25
source-generator recipe from NOR-21's permanent receipt auditor and
recreates the full blind/informed coordinate selection from the
preregistered constants. The private generator is reconstructed only
after the stored public decision is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
from math import comb, sqrt
from pathlib import Path
import random
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import receipt_audit as original

OFF = 94323000
BLIND_SEED = 202611230071
SAMPLE_SEED = 202611230072
DEGREES = (512, 1024)
FAMILIES = ("wide_dense_small", "wide_sparse_small", "wide_rational_short")
KS = (0, 1, 4, 16, 64)
RS = (1, 4, 16)
DELTAS = (1, 4294049777)
N = 4096


def digest_target(a):
    return sha256(json.dumps(a, separators=(",", ":")).encode()).hexdigest()


def independent_coordinates(seed, public_id, size, count, ordinal=0):
    # Independent implementation of exactly the preregistered byte recipe.
    b = ("%d|%s|%d|%d|%d" %
         (seed, public_id, size, count, ordinal)).encode("ascii")
    random_seed = int.from_bytes(sha256(b).digest(), byteorder="big")
    return set(random.Random(random_seed).sample(range(size), count))


def replay_source(row):
    rep, d, fam = row["replicate"], row["observation_degree"], row["family"]
    assert rep in (0, 1) and d in DEGREES and fam in FAMILIES
    seed = (corpus.SEED + OFF + 1000003 * rep +
            100009 * DEGREES.index(d) + 10007 * FAMILIES.index(fam))
    assert row["source_seed"] == seed
    rng = random.Random(seed)
    q = 3 * d // 2
    assert row["query_degree"] == q
    assert 0 <= row["selected_attempt"] < 100
    prog = demos = query = None
    for _ in range(row["selected_attempt"] + 1):
        prog = (rng.choice(corpus.OPS), rng.choice(corpus.DELAYS),
                rng.choice(corpus.SPACINGS), rng.choice(corpus.AMPLITUDES),
                rng.choice(corpus.OPS), rng.choice(corpus.OPS))
        demos = tuple((src := original.independent_source(rng, fam, q),
                       corpus.execute(src, prog)) for _ in range(2))
        query = original.independent_source(rng, fam, q)
    assert prog is not None
    assert row["demos"] == [[list(src[:d+1]), list(y[:d+1])]
                            for src, y in demos]
    assert row["public_query_input"] == list(query)
    h = row["public_header"]
    header = corpus.Header(h["base"], h["delay"], h["spacing"], h["amplitude"])
    assert prog[:4] == (header.base, header.delay, header.spacing, header.amplitude)
    public = corpus.Case("nor23-independent", fam,
                         tuple((tuple(src), tuple(y)) for src, y in row["demos"]),
                         tuple(row["public_query_input"]), (), None)
    assert public.program is None and public.future == ()
    assert original.public_source_hash(public, d) == row["full_public_input_sha256"]
    mask = original.independent_native_mask(public, header, d)
    assert mask == row["original_public_accept_mask"] and mask != 0
    chosen = next(j for j in range(16) if mask & (1 << j))
    predicted = corpus.execute(query, original.candidate_program(header, chosen))
    for j in range(16):
        if mask & (1 << j):
            assert corpus.execute(query, original.candidate_program(header, j)) == predicted
    original_truth = corpus.execute(query, prog)
    assert predicted == original_truth
    return mask, chosen, original_truth


def verify(fixture_bytes, result_bytes):
    fixture = json.loads(fixture_bytes)
    results = json.loads(result_bytes)
    assert fixture["schema"] == "nor23.fresh_public_source_only.v1"
    assert results["schema"] == "nor23.randomized_exact_coordinate_audit.v1"
    assert fixture["registered_source_offset"] == OFF
    assert fixture["public_only"] and fixture["hidden_true_future_and_program_absent"]
    assert fixture["trusted_witness_values_absent"]
    assert fixture["registered_budgets"] == list(KS)
    assert fixture["registered_radii"] == list(RS)
    assert fixture["registered_magnitudes"] == list(DELTAS)
    assert fixture["registered_trials"] == N
    assert results["fixture_sha256"] == sha256(fixture_bytes).hexdigest()
    assert results["distinct_public_source_count"] == 12
    assert results["source_selection_freezes_before_true_future_access"]
    assert results["trusted_witness_supplies_exogenous_true_target_coefficients"]
    assert results["blind_modification_positions_fixed_before_challenge"]
    assert results["adaptive_modification_positions_chosen_after_challenge"]
    assert results["attack_seed"] == BLIND_SEED and results["challenge_seed"] == SAMPLE_SEED
    assert results["sampled_challenges_per_source_budget"] == N
    assert results["budgets"] == list(KS) and results["radii"] == list(RS)
    assert results["magnitudes"] == list(DELTAS)
    assert len(fixture["cases"]) == len(results["source_records"]) == 12
    assert len({x["full_public_input_sha256"] for x in fixture["cases"]}) == 12
    assert Counter(x["observation_degree"] for x in fixture["cases"]) == {512: 6, 1024: 6}
    totals = defaultdict(lambda: Counter())
    for row, rec in zip(fixture["cases"], results["source_records"], strict=True):
        d = row["observation_degree"]
        assert rec["degree"] == d and rec["query_degree"] == 3*d//2
        assert rec["family"] == row["family"]
        assert rec["replicate"] == row["replicate"]
        pub = row["full_public_input_sha256"]
        assert rec["public_sha256"] == pub
        # This function reconstructs the private program only after viewing
        # the frozen public proposal recorded in the full result.
        assert rec["frozen_chosen_class"] is not None
        mask, chosen, truth = replay_source(row)
        assert rec["frozen_public_mask"] == mask
        assert rec["frozen_chosen_class"] == chosen
        assert rec["frozen_predicted_future_sha256"] == digest_target(truth)
        assert rec["valid_true_future_sha256_postchoice"] == digest_target(truth)
        m = len(truth)
        attacks = {
            r: independent_coordinates(BLIND_SEED, pub, m, r) for r in RS
        }
        assert rec["blind_attack_positions"] == {
            str(r): sorted(attacks[r]) for r in RS
        }
        assert len(rec["audits"]) == len(KS)
        for budget_row, k in zip(rec["audits"], KS, strict=True):
            assert budget_row["budget"] == k
            assert budget_row["valid_witness_pass"] == N
            assert budget_row["valid_witness_rejected"] == 0
            assert budget_row["challenge_sample_ns"] > 0
            assert budget_row["all_12_audit_comparisons_ns"] > 0
            counts = {r: 0 for r in RS}
            for t in range(N):
                sample = independent_coordinates(SAMPLE_SEED, pub, m, k, t)
                assert len(sample) == k
                for r in RS:
                    if sample.intersection(attacks[r]):
                        counts[r] += 1
                    assert r <= m - k  # informed attacker can always evade
            observations = {(o["radius"], o["magnitude"]): o
                            for o in budget_row["outcomes"]}
            assert set(observations) == set(product(RS, DELTAS))
            for r, delta in product(RS, DELTAS):
                o = observations[(r, delta)]
                assert o["blind_detected"] == counts[r]
                assert o["blind_false_accept"] == N - counts[r]
                assert o["adaptive_detected"] == 0
                assert o["adaptive_false_accept"] == N
                totals[(d, k, r, delta)]["detected"] += counts[r]
                totals[(d, k, r, delta)]["trials"] += N
    assert len(results["degree_cells"]) == len(totals) == 60
    for cell in results["degree_cells"]:
        d, k, r, delta = (cell["degree"], cell["budget"],
                          cell["altered_coordinates"], cell["nonzero_delta"])
        assert d in DEGREES and k in KS and r in RS and delta in DELTAS
        n = totals[(d, k, r, delta)]["trials"]
        count = totals[(d, k, r, delta)]["detected"]
        m = 3*d//2 + 1
        assert cell["source_count"] == 6
        assert cell["trials"] == n == 6 * N
        assert cell["output_coordinates"] == m
        p = 1 - comb(m-r, k) / comb(m, k)
        tol = 5*sqrt(p*(1-p)/n) + 0.015
        assert abs(cell["exact_uniform_detection_probability"] - p) < 1e-13
        assert abs(cell["observed_blind_detection_probability"] - count/n) < 1e-13
        assert abs(cell["blind_detection_tolerance"] - tol) < 1e-13
        assert cell["blind_detection"] == count
        assert cell["blind_false_accept"] == n - count
        assert cell["adaptive_false_accept"] == n
        assert cell["valid_accept"] == n and cell["valid_reject"] == 0
        assert cell["blind_monte_carlo_within_registered_tolerance"] == (
            abs(count/n-p) <= tol)
    assert results["registered_criteria_pass"] == all(
        c["blind_monte_carlo_within_registered_tolerance"]
        for c in results["degree_cells"])
    return {
        "sources_replayed": 12,
        "degree_cells_independently_recomputed": 60,
        "distinct_challenge_sets_replayed": 12 * len(KS) * N,
        "paired_sparse_world_audit_decisions_replayed":
            12 * len(KS) * N * len(RS) * len(DELTAS) * 2,
        "registered_criteria_pass": results["registered_criteria_pass"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path)
    p.add_argument("--full", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.archive:
        with ZipFile(args.archive) as z:
            assert sorted(z.namelist()) == ["nor23_fixture.json", "nor23_full.json"]
            assert z.testzip() is None
            fixture = z.read("nor23_fixture.json")
            full = z.read("nor23_full.json")
    elif args.fixture and args.full:
        fixture = args.fixture.read_bytes()
        full = args.full.read_bytes()
    else:
        p.error("supply --archive or both --fixture and --full")
    out = verify(fixture, full)
    print("NOR23 INDEPENDENT FIRST-SOURCE REPLAY", out)
    if args.check:
        assert out["registered_criteria_pass"]


if __name__ == "__main__":
    main()
