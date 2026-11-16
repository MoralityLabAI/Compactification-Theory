"""NOR-24: preregistered train-only Bayesian sequential coordinate-audit policy.

Exact candidate futures use NOR-21's unmodified sixteen-program engine.
The training model only estimates weights for four declared public
attack-support templates; all trusted true-future coefficients remain
sealed until the public choices and policies have frozen. See registration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21

SCHEMA = "nor24.train_only_sequential_structured_coordinate_audit.v1"
FIXTURE_SCHEMA = "nor24.public_source_only_24_disjoint_train_test.v1"
OFFSET = 94324000
TRAIN_STREAM = 202611240071
ATTACK_STREAM = 202611240072
CHALLENGE_STREAM = 202611240073
DEGREES = (512, 1024)
FAMILIES = nor21.FAMILIES
TRAIN_REPS = (0, 1)
TEST_REPS = (2, 3)
KINDS = ("tail", "magnitude", "lag", "uniform")
STRESS = "off_template"
ALL_KINDS = KINDS + (STRESS,)
ARMS = ("uniform", "static_prior", "posterior", "shuffled_prior", "oracle_support")
PRIOR_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
SUPPORT_SIZE = 96
TRAIN_EPISODES = 64
TRAIN_RADIUS = 4
CALIBRATION_RADIUS = 4
RADII = (1, 4, 16)
DELTAS = (1, 4294049777)
BUDGETS = (4, 16, 64)
TRIALS = 1024
MAX_ATTEMPTS = 100
SOURCE_COUNT = 24
TEST_COUNT = 12


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()


def stream_seed(*fields):
    raw = "|".join(str(x) for x in fields).encode("ascii")
    return int.from_bytes(sha256(raw).digest(), "big")


def source_seed(rep, degree, family):
    if rep not in TRAIN_REPS + TEST_REPS or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("unregistered NOR24 source")
    return (corpus.SEED + OFFSET + 1000003 * rep +
            100009 * DEGREES.index(degree) + 10007 * FAMILIES.index(family))


def generate_source(rep, degree, family, known):
    q = 3 * degree // 2
    rng = random.Random(source_seed(rep, degree, family))
    for attempt in range(MAX_ATTEMPTS):
        planted = nor21.new_header_program(rng)
        demos = tuple((f := nor21.public_source(rng, family, q),
                       corpus.execute(f, planted)) for _ in range(2))
        query = nor21.public_source(rng, family, q)
        generating = corpus.Case("nor24-source", family, demos, query, (), None)
        headers = corpus.possible_headers(generating)
        if len(headers) != 1:
            continue
        header = headers[0]
        if (header.base, header.delay, header.spacing, header.amplitude) != planted[:4]:
            continue
        digest = nor21.public_digest(generating, degree)
        if digest in known:
            continue
        mask = nor21.native_mask(generating, header, degree)
        if mask == 0:
            continue
        hypothetical = nor21.candidate_predictions(query, header, q)
        if len({hypothetical[j] for j in range(16) if mask & (1 << j)}) != 1:
            continue
        planted_j = corpus.OPS.index(planted[4]) * 4 + corpus.OPS.index(planted[5])
        assert mask & (1 << planted_j)
        known.add(digest)
        return {
            "split": "TRAIN" if rep in TRAIN_REPS else "TEST",
            "replicate": rep, "observation_degree": degree,
            "query_degree": q, "family": family,
            "source_seed": source_seed(rep, degree, family),
            "selected_attempt": attempt,
            "full_public_input_sha256": digest,
            "public_header": {
                "base": header.base, "delay": header.delay,
                "spacing": header.spacing, "amplitude": header.amplitude},
            "demos": [[list(f[:degree+1]), list(y[:degree+1])]
                      for f, y in demos],
            "public_query_input": list(query),
            "original_public_accept_mask": mask,
        }
    raise RuntimeError("NOR24 exhausted source header and unanimity selection")


def make_fixture():
    seen = set()
    cases = [generate_source(rep, degree, fam, seen)
             for rep in TRAIN_REPS + TEST_REPS
             for degree in DEGREES for fam in FAMILIES]
    assert len(seen) == len(cases) == SOURCE_COUNT
    assert Counter(c["split"] for c in cases) == {"TRAIN": 12, "TEST": 12}
    return {
        "schema": FIXTURE_SCHEMA, "new_source_offset": OFFSET,
        "public_only": True, "private_program_and_target_absent": True,
        "heldout_attack_locations_absent": True,
        "train_replicates": list(TRAIN_REPS),
        "test_replicates": list(TEST_REPS),
        "training_episodes_per_source": TRAIN_EPISODES,
        "test_trials_per_source_policy": TRIALS,
        "cases": cases,
    }


def load_fixture(path):
    raw = Path(path).read_bytes()
    fixture = json.loads(raw)
    assert fixture["schema"] == FIXTURE_SCHEMA
    assert fixture["new_source_offset"] == OFFSET
    assert fixture["public_only"] and fixture["private_program_and_target_absent"]
    assert fixture["heldout_attack_locations_absent"]
    assert fixture["train_replicates"] == list(TRAIN_REPS)
    assert fixture["test_replicates"] == list(TEST_REPS)
    assert fixture["training_episodes_per_source"] == TRAIN_EPISODES
    assert fixture["test_trials_per_source_policy"] == TRIALS
    assert len(fixture["cases"]) == SOURCE_COUNT
    assert len({x["full_public_input_sha256"] for x in fixture["cases"]}) == SOURCE_COUNT
    assert Counter(x["split"] for x in fixture["cases"]) == {"TRAIN": 12, "TEST": 12}
    return fixture, sha256(raw).hexdigest()


def public_only_case(row):
    d = row["observation_degree"]
    assert d in DEGREES and row["query_degree"] == d * 3 // 2
    h = row["public_header"]
    header = corpus.Header(h["base"], h["delay"], h["spacing"], h["amplitude"])
    demos = tuple((tuple(f), tuple(y)) for f, y in row["demos"])
    assert len(demos) == 2 and all(len(f) == len(y) == d + 1 for f, y in demos)
    query = tuple(row["public_query_input"])
    assert len(query) == row["query_degree"] + 1
    c = corpus.Case("nor24-public-only", row["family"], demos, query, (), None)
    assert c.program is None and c.future == ()
    assert nor21.public_digest(c, d) == row["full_public_input_sha256"]
    return c, header


def public_proposal(row):
    c, h = public_only_case(row)
    d, q = row["observation_degree"], row["query_degree"]
    proposed = nor21.certify(c, h, d, q, "stream16")
    assert proposed["mask"] == row["original_public_accept_mask"]
    assert proposed["unanimous"] and proposed["chosen_class"] is not None
    chosen = proposed["chosen_class"]
    y = corpus.execute(c.query, nor21.program(h, chosen))
    return {"mask": proposed["mask"], "chosen_class": chosen,
            "predicted": y, "sha256": sha_target(y), "header": h}


def sha_target(a):
    return sha256(json.dumps(a, separators=(",", ":")).encode()).hexdigest()


def supports(predicted, header):
    m = len(predicted)
    assert m in (769, 1537)
    lag_start = header.delay + header.spacing
    assert lag_start >= 0 and lag_start + SUPPORT_SIZE <= m
    top = sorted(range(m), key=lambda i: (-abs(predicted[i]), i))[:SUPPORT_SIZE]
    regions = {
        "tail": frozenset(range(m-SUPPORT_SIZE, m)),
        "magnitude": frozenset(top),
        "lag": frozenset(range(lag_start, lag_start+SUPPORT_SIZE)),
        "uniform": frozenset(range(m)),
    }
    region_union = regions["tail"] | regions["magnitude"] | regions["lag"]
    regions[STRESS] = frozenset(range(m)) - region_union
    assert all(len(regions[f]) == SUPPORT_SIZE for f in KINDS[:3])
    assert len(regions["uniform"]) == m
    assert len(regions[STRESS]) >= max(RADII)
    return regions


def training_counts(rows, source_proposals):
    counts = Counter({f: 0 for f in KINDS})
    selected = []
    for row in rows:
        assert row["split"] == "TRAIN"
        digest = row["full_public_input_sha256"]
        proposal = source_proposals[digest]
        regs = supports(proposal["predicted"], proposal["header"])
        for episode in range(TRAIN_EPISODES):
            rng = random.Random(stream_seed(TRAIN_STREAM, digest, episode))
            draw = rng.random()
            acc = 0.0
            kind = KINDS[-1]
            for candidate_kind, weight in zip(KINDS, PRIOR_WEIGHTS):
                acc += weight
                if draw < acc:
                    kind = candidate_kind
                    break
            positions = tuple(sorted(rng.sample(sorted(regs[kind]), TRAIN_RADIUS)))
            counts[kind] += 1
            selected.append([digest, episode, kind, positions])
    assert sum(counts.values()) == 12 * TRAIN_EPISODES
    pi = {kind: (counts[kind] + 1) / (12 * TRAIN_EPISODES + len(KINDS))
          for kind in KINDS}
    assert abs(sum(pi.values()) - 1) < 1e-12
    receipt_sha = sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()
    return counts, pi, receipt_sha


def blind_attack(digest, kind, radius, reg):
    if kind not in ALL_KINDS or radius not in RADII:
        raise ValueError("attack outside frozen grid")
    rng = random.Random(stream_seed(ATTACK_STREAM, digest, kind, radius))
    return frozenset(rng.sample(sorted(reg[kind]), radius))


def conditional_nohit_count(s, already, radius=CALIBRATION_RADIUS):
    if s - already < radius:
        return 0
    return comb(s-already, radius)


def risk_scores(regions, prior, m):
    scores = [0.0] * m
    for kind in KINDS:
        part = prior[kind] / len(regions[kind])
        for j in regions[kind]:
            scores[j] += part
    return scores


def static_schedule(regions, prior, digest, trial):
    scores = risk_scores(regions, prior, len(regions["uniform"]))
    rng = random.Random(stream_seed(CHALLENGE_STREAM, digest, "static_prior", trial))
    return tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], rng.random()))[:64])


def sequential_nohit_schedule(regions, prior, digest, trial):
    """Exact sequential posterior update along the authentic all-matched path.

    Every altered-world run follows the same path UNTIL it hits a changed
    coefficient, where it rejects. Grouping coordinates by support-membership
    signature reduces complexity without changing the greedy policy.
    """
    rng = random.Random(stream_seed(CHALLENGE_STREAM, digest, "posterior", trial))
    groups = {}
    for i in range(len(regions["uniform"])):
        key = tuple(f for f in KINDS if i in regions[f])
        groups.setdefault(key, []).append(i)
    for pool in groups.values():
        rng.shuffle(pool)
    taken = {f: 0 for f in KINDS}
    denom = {f: comb(len(regions[f]), CALIBRATION_RADIUS) for f in KINDS}
    chosen = []
    for step in range(64):
        weights = {}
        for f in KINDS:
            s = len(regions[f])
            weights[f] = (prior[f] * conditional_nohit_count(s, taken[f]) /
                          denom[f])
        norm = sum(weights.values())
        assert norm > 0
        best = None
        best_score = -1.0
        ties = []
        for key, pool in groups.items():
            if not pool:
                continue
            val = sum(weights[f] * CALIBRATION_RADIUS /
                      (len(regions[f]) - taken[f])
                      for f in key) / norm
            if val > best_score + 1e-15:
                best_score = val
                ties = [key]
            elif abs(val - best_score) <= 1e-15:
                ties.append(key)
        assert ties
        best = rng.choice(ties)
        index = groups[best].pop()
        chosen.append(index)
        for f in best:
            taken[f] += 1
    assert len(chosen) == len(set(chosen)) == 64
    return tuple(chosen)


def policy_schedules(regions, prior, digest, trial):
    m = len(regions["uniform"])
    rng = random.Random(stream_seed(CHALLENGE_STREAM, digest, "uniform", trial))
    seq = list(range(m))
    rng.shuffle(seq)
    uniform = tuple(seq[:64])
    static = static_schedule(regions, prior, digest, trial)
    posterior = sequential_nohit_schedule(regions, prior, digest, trial)
    shuffle_rng = random.Random(
        stream_seed(CHALLENGE_STREAM, digest, "shuffled_prior", trial))
    permutation = list(range(m))
    shuffle_rng.shuffle(permutation)
    shuffled = tuple(permutation[i] for i in static)
    result = {"uniform": uniform, "static_prior": static,
              "posterior": posterior, "shuffled_prior": shuffled}
    for kind in ALL_KINDS:
        oracle_rng = random.Random(stream_seed(
            CHALLENGE_STREAM, digest, "oracle_support", kind, trial))
        pool = list(sorted(regions[kind]))
        oracle_rng.shuffle(pool)
        result[("oracle_support", kind)] = tuple(pool[:64])
    assert all(len(v) == len(set(v)) == 64 for v in result.values())
    return result


def exact_witness_accepts(proposed, true, checked, attacked, delta):
    if len(proposed) != len(true) or delta == 0:
        raise ValueError("misconfigured trusted witness")
    return all(proposed[i] == true[i] + (delta if i in attacked else 0)
               for i in checked)


def informed_attack(checked, support, radius):
    used = set(checked)
    unseen = [i for i in sorted(support) if i not in used][:radius]
    if len(unseen) != radius:
        raise ValueError("informed attack needs uninspected support")
    return frozenset(unseen)


def private_truth_only_after_proposal(row, predicted):
    rep, d, family = row["replicate"], row["observation_degree"], row["family"]
    rng = random.Random(source_seed(rep, d, family))
    prog = query = demos = None
    for _ in range(row["selected_attempt"] + 1):
        prog = nor21.new_header_program(rng)
        demos = tuple((src := nor21.public_source(rng, family, 3*d//2),
                       corpus.execute(src, prog)) for _ in range(2))
        query = nor21.public_source(rng, family, 3*d//2)
    assert row["source_seed"] == source_seed(rep, d, family)
    assert row["demos"] == [[list(s[:d+1]), list(y[:d+1])] for s, y in demos]
    assert row["public_query_input"] == list(query)
    actual = corpus.execute(query, prog)
    assert predicted == actual
    return actual


def evaluate(fixture_path):
    fixture, fixture_sha = load_fixture(fixture_path)
    train_rows = [r for r in fixture["cases"] if r["split"] == "TRAIN"]
    test_rows = [r for r in fixture["cases"] if r["split"] == "TEST"]
    proposals = {r["full_public_input_sha256"]: public_proposal(r)
                 for r in fixture["cases"]}
    counts, prior, train_sha = training_counts(train_rows, proposals)
    source_records = []
    for row in test_rows:
        digest = row["full_public_input_sha256"]
        proposal = proposals[digest]
        target = private_truth_only_after_proposal(row, proposal["predicted"])
        regions = supports(proposal["predicted"], proposal["header"])
        attacks = {(kind, r): blind_attack(digest, kind, r, regions)
                   for kind in ALL_KINDS for r in RADII}
        cells = {(kind, r, k, arm): {"detected": 0, "missed": 0,
                                    "informed_detected": 0,
                                    "informed_false_accept": 0}
                 for kind in ALL_KINDS for r in RADII
                 for k in BUDGETS for arm in ARMS}
        clock_ns = Counter({arm: 0 for arm in ARMS})
        for trial in range(TRIALS):
            start = time.perf_counter_ns()
            schedules = policy_schedules(regions, prior, digest, trial)
            total_ns = time.perf_counter_ns() - start
            # Timed separately in a subsequent pass; aggregate as proposal
            # scheduling CPU for all non-oracle policies together.
            clock_ns["all_schedule_generation"] += total_ns
            for kind in ALL_KINDS:
                for arm in ARMS:
                    plan = schedules[arm] if arm != "oracle_support" else schedules[(arm,kind)]
                    for k in BUDGETS:
                        checked = plan[:k]
                        assert len(checked) == len(set(checked)) == k
                        assert exact_witness_accepts(
                            proposal["predicted"], target, checked,
                            frozenset(), DELTAS[0])
                        for radius in RADII:
                            attack = attacks[(kind, radius)]
                            # A strict truthful witness returns exact target
                            # values; pair two nonzero magnitudes and ensure
                            # that their decisions coincide, without labels
                            # entering any public audit scheduling function.
                            baseline = exact_witness_accepts(
                                proposal["predicted"], target,
                                checked, attack, DELTAS[0])
                            larger = exact_witness_accepts(
                                proposal["predicted"], target,
                                checked, attack, DELTAS[1])
                            assert baseline == larger
                            cell = cells[(kind, radius, k, arm)]
                            cell["missed" if baseline else "detected"] += 1
                            changed_after = informed_attack(
                                checked, regions[kind], radius)
                            # The informed attacker must evade every
                            # inspected coordinate, even for oracle policy.
                            assert exact_witness_accepts(
                                proposal["predicted"], target, checked,
                                changed_after, DELTAS[0])
                            assert exact_witness_accepts(
                                proposal["predicted"], target, checked,
                                changed_after, DELTAS[1])
                            cell["informed_false_accept"] += 1
        for cell in cells.values():
            assert cell["detected"] + cell["missed"] == TRIALS
            assert cell["informed_detected"] == 0
            assert cell["informed_false_accept"] == TRIALS
        source_records.append({
            "split": "TEST", "replicate": row["replicate"],
            "observation_degree": row["observation_degree"],
            "family": row["family"], "public_sha256": digest,
            "chosen_class": proposal["chosen_class"],
            "public_mask": proposal["mask"],
            "public_prediction_sha256": proposal["sha256"],
            "sealed_true_future_sha256_postchoice": sha_target(target),
            "support_sizes": {k: len(regions[k]) for k in ALL_KINDS},
            "attack_positions": {
                f"{kind}|{r}": sorted(attacks[(kind, r)])
                for kind in ALL_KINDS for r in RADII},
            "schedule_generation_ns_total": clock_ns["all_schedule_generation"],
            "observations": [
                {"kind": kind, "radius": r, "budget": k, "policy": arm,
                 **cells[(kind, r, k, arm)]}
                for kind in ALL_KINDS for r in RADII
                for k in BUDGETS for arm in ARMS
            ],
        })
    primary_by_source = []
    for rec in source_records:
        rates = {}
        for arm in ARMS:
            cells = {(z["kind"], z["radius"], z["budget"], z["policy"]): z
                     for z in rec["observations"]}
            rates[arm] = sum(
                PRIOR_WEIGHTS[i] * cells[(kind, 4, 16, arm)]["detected"] / TRIALS
                for i, kind in enumerate(KINDS))
        primary_by_source.append({
            "public_sha256": rec["public_sha256"],
            "degree": rec["observation_degree"],
            "family": rec["family"], "rates": rates,
            "posterior_minus_uniform": rates["posterior"]-rates["uniform"],
            "posterior_minus_shuffled": rates["posterior"]-rates["shuffled_prior"],
            "posterior_minus_static": rates["posterior"]-rates["static_prior"],
        })
    wins_uniform = sum(v["posterior_minus_uniform"] > 0 for v in primary_by_source)
    wins_shuffle = sum(v["posterior_minus_shuffled"] > 0 for v in primary_by_source)
    mean_gain = statistics.mean(v["posterior_minus_uniform"] for v in primary_by_source)
    registered_pass = wins_uniform >= 9 and wins_shuffle >= 9 and mean_gain >= .10
    return {
        "schema": SCHEMA, "fixture_sha256": fixture_sha,
        "public_source_count": SOURCE_COUNT, "train_source_count": 12,
        "heldout_source_count": TEST_COUNT, "trial_count": TRIALS,
        "blind_attack_fixed_before_challenges": True,
        "all_public_proposals_frozen_before_private_target_access": True,
        "learned_prior_from_train_only": True,
        "trusted_witness_values_are_new_external_information": True,
        "family_support_templates_are_declared_model_prior": True,
        "train_family_counts": dict(counts),
        "fitted_prior": prior, "train_attack_position_receipt_sha256": train_sha,
        "weighted_evaluation_mixture": dict(zip(KINDS, PRIOR_WEIGHTS)),
        "source_records": source_records, "primary_by_source": primary_by_source,
        "primary_wins_vs_uniform": wins_uniform,
        "primary_wins_vs_shuffle": wins_shuffle,
        "primary_weighted_mean_gain_vs_uniform": mean_gain,
        "registered_primary_pass": registered_pass,
        "scope": ("Finite synthetic RRC25 and assumed authentic exact target witness. "
                  "Test attack support templates are disclosed to the policy, "
                  "only labeled prior probabilities fitted on training sources. "
                  "Blind and informed adversary protocols differ. "
                  "1024 challenge replicates do not create new source cases."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-fixture", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-parity", action="store_true")
    args = parser.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:
            parser.error("fixture registration and heldout evaluation are separate")
        raw = canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True, exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR24 PUBLIC 24-SOURCE FIXTURE", len(raw), sha256(raw).hexdigest())
        return
    if not args.fixture:
        parser.error("--fixture required")
    result = evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(canonical(result))
    print("NOR24 TRAIN FAMILY COUNTS", result["train_family_counts"])
    print("NOR24 FITTED TRAIN PRIOR", result["fitted_prior"])
    print("NOR24 TEST SOURCE PAIRED", [(z["degree"],
        round(z["posterior_minus_uniform"],4),
        round(z["posterior_minus_shuffled"],4),
        round(z["posterior_minus_static"],4))
        for z in result["primary_by_source"]])
    print("NOR24 PRIMARY", result["registered_primary_pass"],
          result["primary_wins_vs_uniform"],
          result["primary_wins_vs_shuffle"],
          round(result["primary_weighted_mean_gain_vs_uniform"],6))
    if args.check_parity:
        assert result["public_source_count"] == 24
        assert all(z["sealed_true_future_sha256_postchoice"] ==
                   z["public_prediction_sha256"] for z in result["source_records"])
        assert all(c["informed_detected"] == 0 and c["informed_false_accept"] == TRIALS
                   for z in result["source_records"] for c in z["observations"])


if __name__ == "__main__":
    main()
