"""ARG-02 split identities.

* test: the ARG-01 EvalMath roster (seven positive families, negatives,
  collision pairs), unchanged. Used once, for the final run.
* calibration: the ARG-01 36-case calibration roster (seen families).
* dev: an independent ARG-02 roster from the three seen generator families,
  seeded here, signature-disjoint from every EvalMath roster case.
* shots: one solved in-context example per topology, also disjoint.

Hidden fields (source graph, future coefficients) of dev and shot cases are
used only for few-shot answers, probe labels and grading, never for test
prompts.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
import random

from spiral_ln import algebraic_reasoning_oracle as oracle

DEV_SEED = 2026111102
SHOT_SEED = 2026111103
DEV_PER_FAMILY = 32
TEST_SPLITS = oracle.ALL_FAMILIES + ("negative", "collision")
POSITIVE_SPLITS = oracle.ALL_FAMILIES
SEEN_FAMILIES = oracle.CAL_FAMILIES
NOVEL_FAMILIES = oracle.OOD_FAMILIES


@lru_cache(maxsize=1)
def evalmath_roster():
    return oracle.roster()


def test_cases():
    roster = evalmath_roster()
    return [case for split in TEST_SPLITS for case in roster[split]]


def calibration_cases():
    return list(evalmath_roster()["calibration"])


def _forbidden_signatures():
    return {case.signature for cases in evalmath_roster().values() for case in cases}


@lru_cache(maxsize=1)
def dev_cases():
    forbidden = _forbidden_signatures()
    seen = set()
    out = []
    for i, family in enumerate(SEEN_FAMILIES):
        rng = random.Random(DEV_SEED + 7919 * i)
        trials = 0
        count = 0
        while count < DEV_PER_FAMILY:
            trials += 1
            if trials > DEV_PER_FAMILY * 500:
                raise RuntimeError("dev roster exhausted")
            topology = rng.choice(oracle.TOPOLOGIES)
            graph = rng.choice(oracle.PROGRAMS)
            case = oracle.make_case(rng, split="arg02_dev", family=family, topology=topology,
                                    graph=graph, identifier=f"arg02dev:{family}:{trials:05d}")
            if case.signature in forbidden or case.signature in seen:
                continue
            seen.add(case.signature)
            out.append(case)
            count += 1
    return tuple(out)


@lru_cache(maxsize=1)
def shot_cases():
    forbidden = _forbidden_signatures() | {c.signature for c in dev_cases()}
    rng = random.Random(SHOT_SEED)
    out = []
    for i, topology in enumerate(oracle.TOPOLOGIES):
        family = SEEN_FAMILIES[i % len(SEEN_FAMILIES)]
        for trial in range(1, 1000):
            graph = rng.choice(oracle.PROGRAMS)
            case = oracle.make_case(rng, split="arg02_shot", family=family, topology=topology,
                                    graph=graph, identifier=f"arg02shot:{topology}:{trial:04d}")
            if case.signature in forbidden:
                continue
            forbidden.add(case.signature)
            out.append(case)
            break
        else:
            raise RuntimeError("shot roster exhausted")
    return tuple(out)


def development_cases():
    """Selection data: EvalMath calibration plus the ARG-02 dev roster."""
    return calibration_cases() + list(dev_cases())


def public_record(case):
    payload = case.public
    payload["public_sha256"] = case.signature
    return payload


def shot_record(case):
    """A solved example. Only dev-side shot cases may carry their program."""
    if case.split != "arg02_shot":
        raise ValueError("only ARG-02 shot cases carry a visible program")
    row = public_record(case)
    row["program"] = list(case.source_graph)
    return row


def write_public(path, cases):
    rows = [public_record(c) for c in cases]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))
    return len(rows)


def roster_digest(cases):
    return oracle.hash_roster(list(cases))


def audit():
    """Contamination audit: every non-test roster is signature-disjoint from test."""
    test_sig = [c.signature for c in test_cases()]
    test_set = set(test_sig)
    dup_test = len(test_sig) - len(test_set)
    cal = calibration_cases()
    dev = dev_cases()
    shots = shot_cases()
    overlaps = {
        "calibration_vs_test": len({c.signature for c in cal} & test_set),
        "dev_vs_test": len({c.signature for c in dev} & test_set),
        "shots_vs_test": len({c.signature for c in shots} & test_set),
        "dev_vs_calibration": len({c.signature for c in dev} & {c.signature for c in cal}),
        "shots_vs_dev": len({c.signature for c in shots} & {c.signature for c in dev}),
    }
    return {
        "schema": "algernon.arg02.splits.v1",
        "dev_seed": DEV_SEED, "shot_seed": SHOT_SEED, "evalmath_seed": oracle.SEED,
        "counts": {"test": len(test_sig), "test_positive": sum(c.split in POSITIVE_SPLITS for c in test_cases()),
                   "calibration": len(cal), "dev": len(dev), "shots": len(shots)},
        "test_duplicate_public_signatures": dup_test,
        "signature_overlaps": overlaps,
        "roster_sha256": {"test": roster_digest(test_cases()), "calibration": roster_digest(cal),
                          "dev": roster_digest(dev), "shots": roster_digest(shots)},
        "dev_families": sorted({c.family for c in dev}),
        "shot_programs": {c.topology: list(c.source_graph) for c in shots},
    }


def audit_digest():
    return sha256(json.dumps(audit(), sort_keys=True).encode()).hexdigest()
