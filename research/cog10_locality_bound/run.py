"""COG-10: exact local-receptive-field collisions in the COG-02 proof corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path

DEPTHS = tuple(range(17))
SOURCE_SHA256 = "eb6bf1605a293c6b8871e0f14c7eb6fd4af99609d97516ef29d21b712eb001dc"


def topology(world):
    rules = world["rules"]
    consumers = [[] for _ in rules]
    producers = [[] for _ in rules]
    by_atom = defaultdict(list)
    for j, rule in enumerate(rules):
        for p in rule["premises"]:
            by_atom[p].append(j)
    for i, rule in enumerate(rules):
        for j in by_atom[rule["conclusion"]]:
            consumers[i].append(j)
            producers[j].append(i)
    return consumers, producers


def initial_colors(world, known, history, step, budget, topo):
    """Exact rational analogue of the ten COG-03 local token features."""
    consumers, producers = topo
    rules = world["rules"]
    failed = set()
    for event in reversed(history):
        if event.get("new_fact") is not None:
            break
        if event.get("accepted") is False:
            failed.add(event["action"])
    out = []
    for i, rule in enumerate(rules):
        n = len(rule["premises"])
        count = sum(p in known for p in rule["premises"])
        out.append((count == n, F(count, n), rule["conclusion"] in known,
                    rule["conclusion"] == world["goal"], F(n, 4),
                    F(len(consumers[i]), len(rules)), F(len(producers[i]), len(rules)),
                    F(budget-step, budget), i in failed,
                    bool(history) and history[-1]["action"] == i))
    return out


def initial_goal_bits(world):
    return [rule["conclusion"] == world["goal"] for rule in world["rules"]]


def neighbors(topo):
    return [sorted({i, *cs}) for i, cs in enumerate(topo[0])]


def advance_color(colors, adjacency, intern):
    """Exact equality of rooted multiset signatures; no hashing approximation."""
    new = []
    for i, nbrs in enumerate(adjacency):
        key = (colors[i], tuple(sorted(colors[j] for j in nbrs)))
        if key not in intern:
            intern[key] = len(intern)
        new.append(intern[key])
    return new


def initial_intern(values, intern):
    out = []
    for key in values:
        if key not in intern:
            intern[key] = len(intern)
        out.append(intern[key])
    return out


def propagate(bits, adjacency):
    return [any(bits[j] for j in nbrs) for nbrs in adjacency]


def ancestor_bits(world, adjacency):
    bits = initial_goal_bits(world)
    for _ in range(len(bits)):
        nxt = propagate(bits, adjacency)
        if nxt == bits:
            return nxt
        bits = nxt
    if propagate(bits, adjacency) != bits:
        raise ValueError("goal closure did not converge")
    return bits


def pair(d):
    """Two equal menus, goal switches at directed distance d+1."""
    if type(d) is not int or d < 0:
        raise ValueError("nonnegative depth required")
    length = d + 2
    rules = []
    for branch in range(2):
        prev = 0
        for j in range(length):
            conclusion = 1 + branch*length + j
            rules.append({"premises": [prev], "conclusion": conclusion})
            prev = conclusion
    shared = {"atoms": list(range(2*length + 1)), "facts": [0], "rules": rules}
    return [{**shared, "goal": length}, {**shared, "goal": 2*length}]


def witness(d):
    worlds = pair(d)
    top = [topology(w) for w in worlds]
    adj = [neighbors(t) for t in top]
    intern = {}
    colors = [initial_intern(initial_colors(w, {0}, [], 0, 24, t), intern)
              for w, t in zip(worlds, top)]
    labels = [ancestor_bits(w, a)[0] for w, a in zip(worlds, adj)]
    match = []
    for k in range(d+2):
        match.append(colors[0][0] == colors[1][0])
        if k != d+1:
            intern = {}
            colors = [advance_color(c, a, intern) for c, a in zip(colors, adj)]
    if labels != [True, False] or not all(match[:d+1]) or match[d+1]:
        raise AssertionError("locality witness failed")
    return {"depth": d, "rules_per_world": 2*(d+2),
            "identical_through_depth": d, "distinguished_at": d+1,
            "labels": labels, "min_pair_errors_at_depth": 1}


def exact_oracle_error(counts):
    return sum(min(v) for v in counts.values())


def metric(counts, train):
    total = sum(sum(v) for v in counts.values())
    positives = sum(v[1] for v in counts.values())
    errors = unknown = covered_errors = 0
    for key, (neg, pos) in counts.items():
        if key not in train:
            unknown += neg + pos
            errors += pos
        else:
            a, b = train[key]
            wrong = neg if b > a else pos
            errors += wrong
            covered_errors += wrong
    return {
        "candidates": total, "positives": positives, "signatures": len(counts),
        "mixed_signatures": sum(a > 0 and b > 0 for a, b in counts.values()),
        "ambiguous_examples": sum(a+b for a, b in counts.values() if a and b),
        "exact_minimum_errors_for_signature_only_decoder": exact_oracle_error(counts),
        "train_majority_errors_default_negative": errors,
        "train_unseen_signature_examples": unknown,
        "train_covered_errors": covered_errors,
    }


def corpus_states(generator):
    tasks = generator.generate_tasks(64, 20261126)
    states = []
    initial = {}
    checks = 0
    for task in tasks:
        world = task["world"]
        topo = topology(world)
        adj = neighbors(topo)
        ancestry = ancestor_bits(world, adj)
        trace = generator.collect_trace(task)
        generator.replay_trace(task, trace)
        seen = set()
        for step in trace["steps"]:
            obs = step["observation"]
            known = set(obs["known"])
            key = tuple(sorted(known))
            if key in seen:
                continue
            seen.add(key)
            # Fixed budget 64 matches the collection policy, not COG-03's 24-call rollout.
            features = initial_colors(world, known, obs["history"], step["visit"], 64, topo)
            colors = initial_intern(features, initial)
            ready = [set(r["premises"]) <= known and r["conclusion"] not in known
                     for r in world["rules"]]
            labels = [int(r and a) for r, a in zip(ready, ancestry)]
            # Check the labels against the original corpus ranking function.
            row = generator.ranking_row(task, obs["known"])
            if labels != [int(x) for x in row["labels"]["useful"]]:
                raise AssertionError("candidate labels differ from source checker")
            checks += row["collection_cost"]["verifier_calls"]
            states.append({"root_id": task["root_id"], "split": task["split"], "adj": adj,
                           "colors": colors, "ready": ready, "labels": labels,
                           "goal_bits": initial_goal_bits(world)})
    return states, checks


def corpus_metrics(states):
    reports = []
    previous = {}
    for depth in DEPTHS:
        split_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        false_negatives = Counter()
        roots_with_false_negatives = defaultdict(set)
        for state in states:
            split = state["split"]
            for color, label, ready, goal in zip(state["colors"], state["labels"],
                                                 state["ready"], state["goal_bits"]):
                split_counts[split][color][label] += 1
                pred = bool(ready and goal)
                if pred and not label:
                    raise AssertionError("bounded goal reachability produced a false positive")
                if label and not pred:
                    false_negatives[split] += 1
                    roots_with_false_negatives[split].add(state["root_id"])
        train = split_counts["train"]
        summaries = {}
        for split, counts in sorted(split_counts.items()):
            m = metric(counts, train)
            m["bounded_ancestry_false_negatives"] = false_negatives[split]
            m["roots_with_bounded_ancestry_false_negatives"] = len(roots_with_false_negatives[split])
            if split in previous and m["exact_minimum_errors_for_signature_only_decoder"] > previous[split]:
                raise AssertionError("refinement increased minimum possible empirical errors")
            previous[split] = m["exact_minimum_errors_for_signature_only_decoder"]
            summaries[split] = m
        reports.append({"depth": depth, "splits": summaries})
        if depth < DEPTHS[-1]:
            intern = {}
            for state in states:
                state["colors"] = advance_color(state["colors"], state["adj"], intern)
                state["goal_bits"] = propagate(state["goal_bits"], state["adj"])
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if hashlib.sha256(args.generator.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("source generator SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("cog02_generator", args.generator)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    args.output.mkdir(parents=True, exist_ok=False)
    witnesses = [witness(d) for d in range(65)]
    states, checks = corpus_states(generator)
    metrics = corpus_metrics(states)
    summary = {"scope": "exact finite feature-signature analysis on public synthetic proof roots",
               "witness_pairs": len(witnesses), "witnesses": witnesses,
               "source_generator_sha256": SOURCE_SHA256,
               "roots": len({s["root_id"] for s in states}), "states": len(states),
               "source_candidate_checks": checks, "depth_metrics": metrics,
               "confirmation": "none; exact finite corpus results"}
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    # Compact log for remote review, complete depth and split table in artifact.
    print("COG10_SUMMARY " + json.dumps({
        "roots": summary["roots"], "states": len(states), "candidate_checks": checks,
        "witness_pairs": len(witnesses),
        "selected_depths": [x for x in metrics if x["depth"] in (0, 1, 2, 4, 8, 16)],
        "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
