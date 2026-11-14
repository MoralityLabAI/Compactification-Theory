"""Deterministic, exact Horn entailment/countermodel curriculum.

The generator constructs short argument graphs and then *independently* audits
every label using RefutationBench's bounded exhaustive propositional checker.
Valid arguments receive explicit Lean proof terms; invalid arguments receive
a truth assignment checked against every asserted premise. The model sees
only the facts, rules and goal, never certificates, proof traces or split tags.

This is a controlled formal fragment, not natural-language theorem refutation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from ..certificates import find_countermodel, verify_countermodel

SCHEMA = 1
ROOT_SEED = 20261112
SPLITS = {
    "train": (1152, 2, 4, ("linear", "sidechain", "fanin"), 1103),
    "dev": (192, 5, 6, ("ladder", "diamond"), 2207),
    "test": (192, 7, 8, ("deepjoin", "cycle"), 3301),
}
KERNEL_VERSION = "4.19.0"


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _and(premises: list[str]) -> Any:
    if len(premises) == 1:
        return premises[0]
    if len(premises) > 1:
        return {"and": list(premises)}
    return {"and": []}


def formal_problem(task: dict) -> dict:
    premises: list[Any] = list(task["facts"])
    for rule in task["rules"]:
        premises.append({"implies": [_and(rule["premises"]), rule["head"]]})
    return {"kind": "propositional", "premises": premises, "conclusion": task["goal"]}


def least_closure(facts: list[str], rules: list[dict]) -> set[str]:
    """A separate operational Horn engine, independent of truth-table evaluation."""
    known = set(facts)
    for _ in range(len(rules) + 1):
        before = len(known)
        for rule in rules:
            if set(rule["premises"]) <= known:
                known.add(rule["head"])
        if len(known) == before:
            return known
    raise AssertionError("finite Horn closure failed to stabilize")


def proof_trace(task: dict) -> list[tuple[int, str]]:
    known = set(task["facts"])
    trace = []
    for _ in range(len(task["atoms"]) + 1):
        progressed = False
        for i, rule in enumerate(task["rules"]):
            if rule["head"] not in known and set(rule["premises"]) <= known:
                known.add(rule["head"])
                trace.append((i, rule["head"]))
                progressed = True
        if task["goal"] in known:
            return trace
        if not progressed:
            break
    raise AssertionError("claimed valid sample has no Horn proof")


def _wl_fingerprint(task: dict) -> str:
    """Name-invariant colour refinement of the typed atom/rule incidence DAG.

    Conservative WL collisions are treated as duplicates. This is a grouping
    guard and is not a graph-isomorphism theorem.
    """
    names = task["atoms"]
    n = len(names)
    idx = {x: i for i, x in enumerate(names)}
    rules = task["rules"]
    colours = [
        f"A:{int(x in task['facts'])}:{int(x == task['goal'])}"
        for x in names
    ] + [f"R:{len(r['premises'])}" for r in rules]
    for _ in range(7):
        next_colours = []
        for i, name in enumerate(names):
            inc = sorted(colours[n+j] for j, r in enumerate(rules) if name in r["premises"])
            out = sorted(colours[n+j] for j, r in enumerate(rules) if name == r["head"])
            payload = [colours[i], inc, out]
            next_colours.append(hashlib.sha256(canonical_bytes(payload)).hexdigest())
        for j, rule in enumerate(rules):
            payload = [colours[n+j],
                       sorted(colours[idx[x]] for x in rule["premises"]),
                       colours[idx[rule["head"]]]]
            next_colours.append(hashlib.sha256(canonical_bytes(payload)).hexdigest())
        colours = next_colours
    return hashlib.sha256(canonical_bytes(
        [n, len(rules), sorted(colours[:n]), sorted(colours[n:])]
    )).hexdigest()


def _draw(rng: random.Random, split: str, family: str, depth: int,
          good: bool) -> dict:
    """Construct a dependency spine, helper/decoy rules, and one controlled edit."""
    n = depth + 4
    assert 6 <= n <= 12
    h1, h2, guard = depth + 1, depth + 2, depth + 3
    rules: list[dict] = []
    main_indices = []
    for i in range(1, depth + 1):
        premises = [i - 1]
        if i > 1 and family in ("fanin", "ladder", "diamond", "deepjoin"):
            if family == "fanin" or i % 2 == 0:
                premises.append(rng.randrange(0, i - 1))
        main_indices.append(len(rules))
        rules.append({"premises": sorted(set(premises)), "head": i})

    # Helper branches in held-out families, with a required conjunction at
    # the far end. Helpers have an independent derivation from the spine.
    if family in ("diamond", "deepjoin"):
        rules.append({"premises": [max(0, depth - 3)], "head": h1})
        rules[main_indices[-1]]["premises"].append(h1)
        if family == "deepjoin":
            rules.append({"premises": [max(0, depth - 4)], "head": h2})
            rules[main_indices[-1]]["premises"].append(h2)

    # The adversarial guard has no generating rule except guard -> guard.
    # Other decoys may be reachable or cyclic; no decoy derives a spine atom.
    if family == "cycle":
        rules.extend([
            {"premises": [h1], "head": h2},
            {"premises": [h2], "head": h1},
        ])
    else:
        for extra in (h1, h2):
            if family in ("diamond", "deepjoin") and extra == h1:
                continue
            if family == "deepjoin" and extra == h2:
                continue
            p = rng.choice([guard, rng.randrange(depth), rng.randrange(depth + 1)])
            rules.append({"premises": [p], "head": extra})

    for _ in range(rng.randrange(2, 6)):
        head = rng.choice([h1, h2])
        source = rng.choice([guard, h1, h2, rng.randrange(depth + 1)])
        rules.append({"premises": [source], "head": head})
    rules.append({"premises": [guard], "head": guard})

    facts = [0]
    if not good:
        fault = rng.choice(["reverse", "guard_premise", "remove_fact"])
        broken = rng.randrange(1, depth + 1)
        original = rules[main_indices[broken - 1]]
        if fault == "reverse":
            # Old successor edge is replaced with a converse dependency.
            original["premises"], original["head"] = [broken], broken - 1
        elif fault == "guard_premise":
            original["premises"] = sorted(set(original["premises"] + [guard]))
        else:
            facts = [guard]

    # Rename every proposition and randomly reorder rules, eliminating
    # fixed atom indices and inference-order cues from the model's input.
    perm = list(range(n))
    rng.shuffle(perm)
    names = ["P" + str(i) for i in range(n)]
    mapping = {i: names[perm[i]] for i in range(n)}
    output = {
        "atoms": sorted(names),
        "facts": sorted(mapping[f] for f in facts),
        "goal": mapping[depth],
        "rules": [
            {"premises": sorted(mapping[x] for x in r["premises"]),
             "head": mapping[r["head"]]}
            for r in rules
        ],
    }
    rng.shuffle(output["rules"])
    return output


def audited_task(seed: int, split: str, family: str, depth: int, good: bool,
                 tag: str) -> dict:
    rng = random.Random(seed)
    task = _draw(rng, split, family, depth, good)
    problem = formal_problem(task)
    known = least_closure(task["facts"], task["rules"])
    reachable = task["goal"] in known
    if reachable != good:
        raise AssertionError("construction does not realize desired label")
    countermodel = find_countermodel(problem["premises"], problem["conclusion"])
    if (countermodel is None) != good:
        raise AssertionError("truth-table solver disagrees with Horn closure")
    # Least-Horn-model truth values produce an independently checkable
    # counterexample and expose all atom targets for the student.
    closure_assignment = {atom: atom in known for atom in task["atoms"]}
    all_atoms = set(closure_assignment)
    referenced = set()
    for rule in task["rules"]:
        referenced.update(rule["premises"])
        referenced.add(rule["head"])
    referenced.update(task["facts"])
    referenced.add(task["goal"])
    if all_atoms != referenced:
        raise AssertionError("not every declared atom occurs in premises")
    cert = None
    if not good:
        cert = {"kind": "countermodel", "assignment": closure_assignment}
        verified = verify_countermodel(problem["premises"], problem["conclusion"],
                                       closure_assignment)
        if not verified["verified"]:
            raise AssertionError("least Horn assignment failed certificate check")
    else:
        proof_trace(task)
    return {
        "case_id": tag, "schema_version": SCHEMA, "split": split,
        "family_id": family, "depth": depth,
        "atoms": task["atoms"], "facts": task["facts"],
        "rules": task["rules"], "goal": task["goal"],
        "formal": problem,
        "label": "established" if good else "refuted",
        "reachable": sorted(known),
        "countermodel": cert,
        "fingerprint": _wl_fingerprint(task),
    }


def generate() -> list[dict]:
    result = []
    observed: dict[str, str] = {}
    for split, (count, dmin, dmax, families, offset) in SPLITS.items():
        accepted = 0
        attempts = 0
        per_family = Counter()
        per_label = Counter()
        while accepted < count:
            attempts += 1
            if attempts > count * 200:
                raise RuntimeError(f"insufficient unique {split} examples")
            family = families[accepted % len(families)]
            good = (per_family[family] % 2 == 0)
            depth = dmin + ((per_family[family] // 2) % (dmax - dmin + 1))
            # Seed depends on attempt so rejected WL collisions retry.
            seed = ROOT_SEED * 100_000 + offset * 10_000 + attempts
            ident = "FC_" + split + "_" + f"{accepted:05d}"
            task = audited_task(seed, split, family, depth, good, ident)
            fp = task["fingerprint"]
            if fp in observed:
                continue
            observed[fp] = split
            result.append(task)
            per_family[family] += 1
            per_label[task["label"]] += 1
            accepted += 1
        assert abs(per_label["established"] - per_label["refuted"]) <= 2
    return result


def _conjunction_term(terms: list[str]) -> str:
    if len(terms) == 1:
        return terms[0]
    return "(And.intro " + terms[0] + " " + _conjunction_term(terms[1:]) + ")"


def lean_theorem(task: dict) -> str:
    if task["label"] != "established":
        raise ValueError("no Lean proof for a countermodel-labelled task")
    names = task["atoms"]
    facts = task["facts"]
    rules = task["rules"]
    theorem_name = task["case_id"].lower()
    params = ["(" + " ".join(names) + " : Prop)"]
    params += ["(h" + str(j) + " : " + f + ")" for j, f in enumerate(facts)]
    for j, r in enumerate(rules):
        premise_type = " ∧ ".join(r["premises"])
        params.append("(r" + str(j) + " : (" + premise_type + ") → " + r["head"] + ")")
    lines = ["theorem " + theorem_name + " " + " ".join(params) +
             " : " + task["goal"] + " := by"]
    known = {fact: "h" + str(j) for j, fact in enumerate(facts)}
    for j, head in proof_trace(task):
        if head in known:
            continue
        rule = rules[j]
        args = [known[x] for x in rule["premises"]]
        term = "r" + str(j) + " " + _conjunction_term(args)
        label = "d" + str(len(known))
        lines.append("  have " + label + " : " + head + " := " + term)
        known[head] = label
    lines.append("  exact " + known[task["goal"]])
    return "\n".join(lines)


def lean_corpus(records: list[dict]) -> str:
    good = [t for t in records if t["label"] == "established"]
    lines = [
        "-- GENERATED: RefutationBench exact formal curriculum.",
        "-- Every lemma is parameterized by its premises; no external axioms.",
        "-- Lean 4." + KERNEL_VERSION.split(".")[1] + " / version " + KERNEL_VERSION,
        "set_option maxRecDepth 4096",
        "",
    ]
    lines += [lean_theorem(task) + "\n" for task in good]
    if good:
        lines.append("#print axioms " + good[0]["case_id"].lower())
        lines.append("#print axioms " + good[-1]["case_id"].lower())
    return "\n".join(lines) + "\n"


def audit(records: list[dict]) -> dict:
    by_split = Counter(t["split"] for t in records)
    by_label = Counter(t["label"] for t in records)
    families: dict[str, str] = {}
    fingerprints: set[str] = set()
    for task in records:
        fam, split = task["family_id"], task["split"]
        if fam in families and families[fam] != split:
            raise AssertionError("family leaks across splits")
        families[fam] = split
        if task["fingerprint"] in fingerprints:
            raise AssertionError("duplicate WL fingerprint")
        fingerprints.add(task["fingerprint"])
        original = {"atoms": task["atoms"], "facts": task["facts"],
                    "rules": task["rules"], "goal": task["goal"]}
        if _wl_fingerprint(original) != task["fingerprint"]:
            raise AssertionError("fingerprint mismatch")
        known = least_closure(task["facts"], task["rules"])
        if sorted(known) != task["reachable"]:
            raise AssertionError("closure mismatch")
        if (task["goal"] in known) != (task["label"] == "established"):
            raise AssertionError("label mismatch")
        problem = formal_problem(task)
        if problem != task["formal"]:
            raise AssertionError("formal JSON changed")
        counter = find_countermodel(problem["premises"], problem["conclusion"])
        if (counter is None) != (task["label"] == "established"):
            raise AssertionError("exhaustive checker disagrees")
        if task["label"] == "refuted":
            cert = task["countermodel"]
            if cert is None or not verify_countermodel(
                problem["premises"], problem["conclusion"],
                cert["assignment"]
            )["verified"]:
                raise AssertionError("countermodel mismatch")
        elif task["countermodel"] is not None:
            raise AssertionError("valid theorem has a countermodel")
    expected = {k: v[0] for k, v in SPLITS.items()}
    if dict(by_split) != expected:
        raise AssertionError("counts changed")
    return {
        "schema_version": SCHEMA, "seed": ROOT_SEED,
        "count": len(records),
        "splits": dict(sorted(by_split.items())),
        "labels": dict(sorted(by_label.items())),
        "families": {k: families[k] for k in sorted(families)},
        "fingerprint_overlap": False,
        "independent_certificate_checks": len(records),
        "lean_theorem_count": by_label["established"],
        "declared_semantics": "bounded propositional Horn clauses only",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_bytes(x) + b"\n" for x in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lean", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    rows = generate()
    report = audit(rows)
    write_jsonl(args.out, rows)
    report["cases_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.lean:
        args.lean.parent.mkdir(parents=True, exist_ok=True)
        args.lean.write_text(lean_corpus(rows), encoding="utf8")
        report["lean_source_sha256"] = hashlib.sha256(args.lean.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
