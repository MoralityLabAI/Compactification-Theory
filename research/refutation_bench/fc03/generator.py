"""FC-03 preregistered test-only unseen-family Horn curriculum.

Registration, frozen BEFORE this implementation and all CI metrics:
research/refutation_bench/fc03/PREREGISTRATION.md at 650bd2cc5f0fd628784f672248c3a3e362e9dcdf

Training samples come exclusively from FC-02. This module adds two test sets
at matched and shifted depth. It rejects WL structural fingerprint collisions
against ALL FC-02 families. Independent exhaustive Boolean checking audits all
test labels and certificates. It does not train or select a model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from ..formal_curriculum import generator as fc02
from ..certificates import find_countermodel, verify_countermodel

ROOT_SEED = 20261113
FAMILIES = ("fork_join", "anchored_cycle", "alternate_routes", "guarded_ladder")
TEST_SPLITS = {
    "test_same": (192, 2, 4, 41031),
    "test_deep": (192, 6, 8, 51047),
}


def _draw(rng: random.Random, family: str, depth: int, good: bool) -> dict:
    """Generate non-lexical graph families with both positive and invalid tasks.

    All source and goal indices are permuted after a graph is fully built;
    no raw-label, proof trace, split, or test-family features enter the model.
    """
    n = depth + 4
    h1, h2, guard = depth + 1, depth + 2, depth + 3
    facts = [0]
    rules = [{"premises": [i-1], "head": i} for i in range(1, depth+1)]
    last = depth - 1

    if family == "fork_join":
        a = rng.randint(0, max(0, depth-2))
        b = rng.randint(0, max(0, depth-2))
        # Two independent branches must close before the final join.
        rules.append({"premises": [a], "head": h1})
        rules.append({"premises": [b if good else guard], "head": h2})
        rules[last]["premises"] = sorted(set([depth-1, h1, h2]))

    elif family == "anchored_cycle":
        # Cyclic implications only propagate if an external fact anchors them.
        rules.append({"premises": [0 if good else guard], "head": h1})
        rules.append({"premises": [h1], "head": h2})
        rules.append({"premises": [h2], "head": h1})
        rules[last]["premises"] = sorted(set([depth-1, h2]))

    elif family == "alternate_routes":
        # Two separate derivations of the conclusion, both interrupted in
        # the invalid pair. Valid records can use either route.
        branch_point = rng.randint(0, max(0, depth-2))
        rules.append({"premises": [branch_point], "head": h1})
        rules.append({"premises": [h1], "head": h2})
        rules.append({"premises": [h2] if good else [h2, guard],
                      "head": depth})
        if not good:
            rules[last]["premises"] = sorted(set([depth-1, guard]))
        elif rng.choice([True, False]):
            # One route is blocked but the second stays usable.
            rules[last]["premises"] = sorted(set([depth-1, guard]))

    elif family == "guarded_ladder":
        rules.append({"premises": [0], "head": h1})
        rules.append({"premises": [max(0, depth-2)], "head": h2})
        for step in range(2, depth+1):
            if step % 2 == 0:
                rules[step-1]["premises"] = [step-1, h1]
        if not good:
            step = rng.randint(1, depth)
            rules[step-1]["premises"] = sorted(set(
                rules[step-1]["premises"] + [guard]))

    else:
        raise ValueError(f"unknown family {family}")

    # Guard is always unreachable; self-loop is a false source of support.
    rules.append({"premises": [guard], "head": guard})
    # Non-goal decoys alter topology and defeat fixed-order, fixed-size cues.
    for _ in range(rng.randint(2, 5)):
        head = rng.choice((h1,h2))
        choices = [guard, h1, h2, rng.randrange(depth)]
        prem = [rng.choice(choices)]
        if rng.random() < .35:
            prem.append(rng.choice(choices))
        rules.append({"premises": sorted(set(prem)), "head": head})

    assert len(rules) <= 20 and n <= 12 and n >= 6
    names = [f"P{i}" for i in range(n)]
    permutation = list(range(n))
    rng.shuffle(permutation)
    alias = {i: names[permutation[i]] for i in range(n)}
    result = {
        "atoms": sorted(names),
        "facts": [alias[i] for i in facts],
        "goal": alias[depth],
        "rules": [{"premises": sorted(alias[x] for x in r["premises"]),
                   "head": alias[r["head"]]} for r in rules],
    }
    rng.shuffle(result["rules"])
    return result


def audited_task(seed: int, split: str, family: str, depth: int,
                 positive: bool, identifier: str) -> dict:
    rng = random.Random(seed)
    task = _draw(rng,family,depth,positive)
    problem = fc02.formal_problem(task)
    reachable = fc02.least_closure(task["facts"],task["rules"])
    if (task["goal"] in reachable) != positive:
        raise AssertionError(f"construction error in {split} {family}")
    witness = find_countermodel(problem["premises"],problem["conclusion"])
    if (witness is None) != positive:
        raise AssertionError("truth-table / closure mismatch")
    assignment = {a:a in reachable for a in task["atoms"]}
    certificate = None
    if not positive:
        certificate = {"kind":"countermodel","assignment":assignment}
        if not verify_countermodel(problem["premises"],problem["conclusion"],
                                   assignment)["verified"]:
            raise AssertionError("generated countermodel failed exact verifier")
    else:
        fc02.proof_trace(task)
    referenced = set(task["facts"]) | {task["goal"]}
    for rule in task["rules"]:
        referenced |= set(rule["premises"])
        referenced.add(rule["head"])
    if referenced != set(task["atoms"]):
        raise AssertionError("unused atom in generator")
    return {
        **task,
        "case_id": identifier,
        "schema_version": 1,
        "split": split,
        "family_id": family,
        "depth": depth,
        "formal": problem,
        "label": "established" if positive else "refuted",
        "reachable": sorted(reachable),
        "countermodel": certificate,
        "fingerprint": fc02._wl_fingerprint(task),
    }


def generate_tests() -> list[dict]:
    """Generate tests with enforced non-overlap vs prior FC-02 dataset."""
    old = fc02.generate()
    seen = {task["fingerprint"] for task in old}
    if len(seen) != len(old):
        raise AssertionError("upstream structural duplicates")
    result: list[dict] = []
    for split,(count,dmin,dmax,offset) in TEST_SPLITS.items():
        accepted = Counter()
        attempts = 0
        while sum(accepted.values()) < count:
            attempts += 1
            if attempts > count*500:
                raise RuntimeError(f"too few distinct graphs in {split}")
            fam = FAMILIES[sum(accepted.values()) % len(FAMILIES)]
            k = accepted[fam]
            good = (k%2==0)
            depth = dmin + ((k//2) % (dmax-dmin+1))
            seed = ROOT_SEED*100_000 + offset*10_000 + attempts
            t = audited_task(seed,split,fam,depth,good,
                f"FC03_{split}_{sum(accepted.values()):04d}")
            if t["fingerprint"] in seen:
                continue
            seen.add(t["fingerprint"])
            accepted[fam] += 1
            result.append(t)
        if any(accepted[fam] != count//len(FAMILIES) for fam in FAMILIES):
            raise AssertionError("family imbalance")
    audit_tests(result,old)
    return result


def audit_tests(rows: list[dict], old: list[dict] | None = None) -> dict:
    if old is None:
        old = fc02.generate()
    source_fps = {t["fingerprint"] for t in old}
    seen = set()
    splits = Counter()
    labels = Counter()
    groups = Counter()
    for task in rows:
        split,family = task["split"],task["family_id"]
        if split not in TEST_SPLITS or family not in FAMILIES:
            raise AssertionError("wrong test split or family")
        splits[split] += 1
        labels[(split,task["label"])] += 1
        groups[(split,family,task["label"])] += 1
        fp = fc02._wl_fingerprint(task)
        if fp != task["fingerprint"] or fp in source_fps or fp in seen:
            raise AssertionError("split leakage or structural repetition")
        seen.add(fp)
        low,high=TEST_SPLITS[split][1:3]
        if not low<=task["depth"]<=high:
            raise AssertionError("depth outside preregistered interval")
        known = fc02.least_closure(task["facts"],task["rules"])
        if sorted(known)!=task["reachable"]:
            raise AssertionError("incorrect closure state")
        p=fc02.formal_problem(task)
        if p!=task["formal"]:
            raise AssertionError("formal problem changed")
        exhaustive=find_countermodel(p["premises"],p["conclusion"])
        if (exhaustive is None) != (task["label"]=="established"):
            raise AssertionError("truth-table disagreement")
        if task["label"]=="refuted":
            if task["countermodel"] is None:
                raise AssertionError("missing invalid certificate")
            if not verify_countermodel(p["premises"],p["conclusion"],
                  task["countermodel"]["assignment"])["verified"]:
                raise AssertionError("unverified counterexample")
        else:
            if task["countermodel"] is not None:
                raise AssertionError("positive has a counterexample")
            fc02.proof_trace(task)

    if dict(splits)!={key:data[0] for key,data in TEST_SPLITS.items()}:
        raise AssertionError("split size mismatch")
    for split in TEST_SPLITS:
        for label in ("established","refuted"):
            if labels[split,label]!=TEST_SPLITS[split][0]//2:
                raise AssertionError("label imbalance")
            for fam in FAMILIES:
                if groups[split,fam,label] != TEST_SPLITS[split][0]//(2*len(FAMILIES)):
                    raise AssertionError("family label imbalance")
    return {
        "source":"FC03 public preregistered grammar",
        "schema_version":1,
        "root_seed": ROOT_SEED,
        "splits":dict(splits),
        "labels":{f"{a}/{b}":v for (a,b),v in sorted(labels.items())},
        "families":list(FAMILIES),
        "distinct_fingerprints":len(seen),
        "fc02_fingerprint_overlap":0,
        "positive_lean_theorems":sum(x["label"]=="established" for x in rows),
        "exact_countermodels":sum(x["label"]=="refuted" for x in rows),
        "test_families_are_out_of_train_dev": True,
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    rows=generate_tests()
    audit=audit_tests(rows)
    fc02.write_jsonl(args.out,rows)
    audit["cases_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.lean:
        args.lean.parent.mkdir(parents=True,exist_ok=True)
        args.lean.write_text(fc02.lean_corpus(rows),encoding="utf8")
        audit["lean_source_sha256"]=hashlib.sha256(args.lean.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(audit,sort_keys=True,indent=2)+"\n")
    print(json.dumps(audit,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
