"""Independently authenticate COG-07 examples to complete COG-02 executions.

This auditor imports neither the producer's Horn checker nor the compression
loader. It reconstructs every teacher state from declared facts and rules,
checks complete ranking coverage, and can independently enumerate the finite
certificate represented by selected.json and its adjacent compressed.bin.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "cogentic.ledger.v1"
SPLITS = {"train", "calibration", "ordinary_test", "ood_test"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows


def require_equal(actual: Any, expected: Any, message: str) -> None:
    if canonical(actual) != canonical(expected):
        raise ValueError(message)


def validate_world(world: dict[str, Any]) -> None:
    if not isinstance(world, dict) or set(world) != {"atoms", "facts", "rules", "goal"}:
        raise ValueError("world field mismatch")
    atoms, facts, rules = world["atoms"], world["facts"], world["rules"]
    if (not isinstance(atoms, list) or not 2 <= len(atoms) <= 128
            or any(type(x) is not int for x in atoms) or atoms != list(range(len(atoms)))):
        raise ValueError("invalid atom domain")
    if (not isinstance(facts, list) or not facts or any(type(x) is not int for x in facts)
            or facts != sorted(set(facts)) or not set(facts) <= set(atoms)):
        raise ValueError("invalid initial facts")
    if type(world["goal"]) is not int or world["goal"] not in atoms or world["goal"] in facts:
        raise ValueError("invalid goal")
    if not isinstance(rules, list) or not 1 <= len(rules) <= 128:
        raise ValueError("invalid rule count")
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"premises", "conclusion"}:
            raise ValueError("invalid rule record")
        premises = rule["premises"]
        if (not isinstance(premises, list) or not premises
                or any(type(x) is not int for x in premises)
                or premises != sorted(set(premises)) or not set(premises) <= set(atoms)
                or type(rule["conclusion"]) is not int or rule["conclusion"] not in atoms):
            raise ValueError("invalid implication")


def checked_step(world: dict[str, Any], known: set[int], action: Any) -> dict[str, Any]:
    """An independent interpreter of the public finite implication language."""
    after = set(known)
    accepted, code, atom = False, "invalid_action", None
    if type(action) is int and 0 <= action < len(world["rules"]):
        rule = world["rules"][action]
        if all(p in known for p in rule["premises"]):
            accepted = True
            conclusion = rule["conclusion"]
            code = "duplicate_lemma" if conclusion in known else "proved_lemma"
            if conclusion not in known:
                atom = conclusion
            after.add(conclusion)
        else:
            code = "missing_premise"
    return {"accepted": accepted, "code": code, "next_known": sorted(after),
            "new_fact": atom, "solved": world["goal"] in after, "verifier_calls": 1}


def goal_ancestors(world: dict[str, Any]) -> set[int]:
    parents: dict[int, set[int]] = defaultdict(set)
    for rule in world["rules"]:
        parents[rule["conclusion"]].update(rule["premises"])
    pending, visited = [world["goal"]], set()
    while pending:
        atom = pending.pop()
        if atom not in visited:
            visited.add(atom)
            pending.extend(parents[atom] - visited)
    return visited


def candidate_features(world: dict[str, Any], known: set[int], action: int) -> list[float]:
    rule = world["rules"][action]
    premises, conclusion = set(rule["premises"]), rule["conclusion"]
    uses = Counter(p for item in world["rules"] for p in item["premises"])
    goal_parents = {p for item in world["rules"] if item["conclusion"] == world["goal"]
                    for p in item["premises"]}
    width, rule_count = len(premises), len(world["rules"])
    return [1.0, width / 3, len(premises & known) / width, len(premises - known) / width,
            float(conclusion in known), float(conclusion == world["goal"]),
            len(goal_parents & known) / max(1, len(goal_parents)), uses[conclusion] / rule_count,
            sum(uses[p] for p in premises) / (width * rule_count),
            len(premises & set(world["facts"])) / width,
            float(conclusion in goal_parents), len(known) / len(world["atoms"])]


def feature_bits(world: dict[str, Any], known: set[int], action: int, ancestors: set[int]) -> int:
    rule = world["rules"][action]
    premises, conclusion = set(rule["premises"]), rule["conclusion"]
    values = (premises <= known, conclusion not in known, conclusion in ancestors,
              conclusion == world["goal"], len(premises) == 1,
              premises <= set(world["facts"]),
              any(conclusion in other["premises"] for other in world["rules"]),
              2 * len(known) >= len(world["atoms"]))
    return sum(int(value) << i for i, value in enumerate(values))


def _registry(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not tasks:
        raise ValueError("empty task registry")
    registry, presentations, worlds = {}, set(), set()
    for task in tasks:
        root = task["root_id"]
        if not isinstance(root, str) or re.fullmatch("[0-9a-f]{64}", root) is None:
            raise ValueError("invalid root digest")
        if root in registry:
            raise ValueError("duplicate root or split custody violation")
        if task.get("schema_version") != SCHEMA or task.get("split") not in SPLITS:
            raise ValueError("task schema or split mismatch")
        validate_world(task["world"])
        world_sha = digest(task["world"])
        if world_sha in worlds or task["task_id"] in presentations:
            raise ValueError("aliased world or presentation")
        require_equal(task["task_id"], world_sha, "presentation digest mismatch")
        require_equal(task["task_digest"], digest({k: v for k, v in task.items() if k != "task_digest"}),
                      "task content digest mismatch")
        require_equal(task["provenance"]["source_world_sha256"], root, "root provenance mismatch")
        if task["provenance"].get("kind") != "synthetic_horn_proof_dag":
            raise ValueError("unsupported source provenance")
        registry[root] = task
        worlds.add(world_sha)
        presentations.add(task["task_id"])
    return registry


def _teacher_states(
    registry: dict[str, dict[str, Any]], traces: list[dict[str, Any]],
) -> tuple[dict[str, set[tuple[int, ...]]], Counter[str]]:
    states: dict[str, set[tuple[int, ...]]] = {}
    counts: Counter[str] = Counter()
    for trace in traces:
        root = trace["root_id"]
        if root not in registry or root in states:
            raise ValueError("unregistered or duplicate teacher root")
        task = registry[root]
        require_equal(trace["schema_version"], SCHEMA, "trace schema mismatch")
        for key in ("root_id", "task_id", "task_digest", "split", "family", "provenance"):
            require_equal(trace[key], task[key], "trace task binding mismatch: " + key)
        require_equal(trace["trace_digest"], digest({k: v for k, v in trace.items() if k != "trace_digest"}),
                      "trace digest mismatch")
        world, history = task["world"], []
        known = set(world["facts"])
        previous = digest({"task_digest": task["task_digest"], "policy": trace["policy"]})
        states[root] = set()
        for visit, event in enumerate(trace["steps"]):
            require_equal(event["visit"], visit, "trace order mismatch")
            require_equal(event["previous_digest"], previous, "trace chain mismatch")
            require_equal(event["step_digest"], digest({k: v for k, v in event.items() if k != "step_digest"}),
                          "step digest mismatch")
            observation = {"world": world, "known": sorted(known), "history": history}
            require_equal(event["observation"], observation, "forged public state or history")
            receipt = checked_step(world, known, event["action"])
            require_equal(event["verification"], receipt, "producer checker mismatch")
            require_equal(event["cost"], {"verifier_calls": 1, "proposal_calls": 1}, "erased verifier cost")
            states[root].add(tuple(sorted(known)))
            counts[receipt["code"]] += 1
            counts["verifier_calls"] += 1
            known = set(receipt["next_known"])
            history.append({"action": event["action"], **{k: receipt[k] for k in
                            ("accepted", "code", "new_fact")}})
            previous = event["step_digest"]
        require_equal(trace["solved"], world["goal"] in known, "terminal verdict mismatch")
        require_equal(trace["verifier_calls"], len(trace["steps"]), "episode call total mismatch")
        require_equal(trace["final_digest"], previous, "terminal chain mismatch")
        if world["goal"] not in known:
            raise ValueError("teacher trace lacks a completed proof")
        counts["completed_teacher_roots"] += 1
    if set(states) != set(registry):
        raise ValueError("missing registered teacher trace")
    return states, counts


def _ranking_examples(
    registry: dict[str, dict[str, Any]], states: dict[str, set[tuple[int, ...]]],
    ranking: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    seen, examples, counts = set(), [], Counter()
    for row in ranking:
        root = row["root_id"]
        if root not in registry:
            raise ValueError("unregistered ranking root")
        task = registry[root]
        for key in ("root_id", "task_id", "split", "family", "schema_version"):
            require_equal(row[key], task[key], "ranking task binding mismatch: " + key)
        observation = row["public"]
        if set(observation) != {"world", "known", "history"}:
            raise ValueError("ranking public field mismatch")
        require_equal(observation["world"], task["world"], "ranking world mismatch")
        require_equal(observation["history"], [], "ranking projection has unexpected history")
        known = observation["known"]
        if (not isinstance(known, list) or any(type(x) is not int for x in known)
                or known != sorted(set(known)) or tuple(known) not in states[root]):
            raise ValueError("ranking state never reached by authenticated teacher")
        state_key = (root, tuple(known))
        if state_key in seen:
            raise ValueError("duplicate ranking state")
        seen.add(state_key)
        world, known_set = task["world"], set(known)
        candidates = row["candidates"]
        require_equal([c["action"] for c in candidates], list(range(len(world["rules"]))),
                      "ranking candidate coverage mismatch")
        for field in ("accepted", "useful"):
            if len(row["labels"][field]) != len(candidates):
                raise ValueError("ranking label coverage mismatch")
        require_equal(row["collection_cost"], {"verifier_calls": len(candidates)},
                      "ranking collection cost mismatch")
        ancestors = goal_ancestors(world)
        for index, candidate in enumerate(candidates):
            receipt = checked_step(world, known_set, candidate["action"])
            label = receipt["new_fact"] is not None and receipt["new_fact"] in ancestors
            require_equal(row["labels"]["accepted"][index], receipt["accepted"], "accepted label mismatch")
            require_equal(row["labels"]["useful"][index], label, "useful label mismatch")
            require_equal(candidate["features"], candidate_features(world, known_set, index),
                          "public candidate feature mismatch")
            bits = feature_bits(world, known_set, index, ancestors)
            if bool(bits & 7 == 7) != label:
                raise ValueError("independent feature and target mismatch")
            if task["split"] == "train":
                examples.append({"bits": bits, "label": int(label), "root_id": root,
                                 "split": "train", "action": index, "state_sha256": digest(observation)})
            counts["candidate_labels_and_features_checked"] += 1
        counts["ranking_states_authenticated"] += 1
    expected = {(root, state) for root, values in states.items() for state in values}
    if seen != expected:
        raise ValueError("ranking corpus omits a registered reached state")
    return examples, counts


def independent_survivors(pairs: set[tuple[int, int]]) -> list[int]:
    survivors = []
    for index, literals in enumerate(itertools.product((0, 1, -1), repeat=8)):
        if all(
            all(lit == 0 or ((bits >> j) & 1) == (1 if lit == 1 else 0)
                for j, lit in enumerate(literals)) == bool(label)
            for bits, label in pairs
        ):
            survivors.append(index)
    if all(label == 0 for _, label in pairs):
        survivors.append(3 ** 8)
    return survivors


def _selected_audit(path: Path, training: list[dict[str, Any]]) -> dict[str, Any]:
    sidecar = json.loads(path.read_text())
    indices = sidecar["selected_training_indices"]
    if (any(type(i) is not int or not 0 <= i < len(training) for i in indices)
            or len(indices) != len(set(indices))):
        raise ValueError("invalid selected source indices")
    selected = [training[i] for i in indices]
    require_equal(sidecar["selected_examples"], selected, "selected provenance mismatch")
    packet_path = path.with_name("compressed.bin")
    packet = packet_path.read_bytes()
    require_equal(sidecar["packet_sha256"], file_digest(packet_path), "packet digest mismatch")
    if len(packet) < 7 or packet[:4] != b"COG7" or packet[4] != 8:
        raise ValueError("unsupported finite packet")
    count = int.from_bytes(packet[5:7], "big")
    bits = 9 * count
    if len(packet) != 7 + (bits + 7) // 8 or count != len(selected):
        raise ValueError("selected packet length mismatch")
    stream = "".join(f"{byte:08b}" for byte in packet[7:])
    if any(bit != "0" for bit in stream[bits:]):
        raise ValueError("noncanonical packet padding")
    decoded = [(int(stream[i*9:i*9+8], 2), int(stream[i*9+8])) for i in range(count)]
    require_equal(decoded, [(x["bits"], x["label"]) for x in selected], "selected packet values mismatch")
    pairs = {(x["bits"], x["label"]) for x in training}
    full_space, small_space = independent_survivors(pairs), independent_survivors(set(decoded))
    require_equal(small_space, full_space, "independent version-space certificate failed")
    return {"source_sha256": file_digest(path), "packet_sha256": file_digest(packet_path),
            "selected_examples": count, "selected_roots": len({x["root_id"] for x in selected}),
            "hypotheses_enumerated": 3 ** 8 + 1, "full_surviving_ids": full_space,
            "compressed_surviving_ids": small_space, "version_space_equal": True,
            "packet_bytes": len(packet), "selected_source_indices_verified": True}


def audit(
    tasks_path: Path, traces_path: Path, ranking_path: Path, selected_path: Path | None = None,
) -> dict[str, Any]:
    tasks, traces, ranking = (read_rows(path) for path in (tasks_path, traces_path, ranking_path))
    registry = _registry(tasks)
    states, trace_counts = _teacher_states(registry, traces)
    training, ranking_counts = _ranking_examples(registry, states, ranking)
    unique_pairs = {(x["bits"], x["label"]) for x in training}
    registry_values = [{"root_id": root, "task_id": task["task_id"], "split": task["split"],
                        "world_sha256": digest(task["world"])} for root, task in sorted(registry.items())]
    result = {
        "schema": "COG07/independent-provenance-audit/v1", "passed": True,
        "source_sha256": file_digest(Path(__file__)),
        "inputs": {name: {"sha256": file_digest(path), "bytes": path.stat().st_size}
                   for name, path in (("tasks", tasks_path), ("traces", traces_path), ("ranking", ranking_path))},
        "registered_roots": len(registry), "registered_root_registry_sha256": digest(registry_values),
        "roots_per_split": dict(Counter(task["split"] for task in tasks)),
        "duplicate_root_or_world_aliases": 0, "missing_teacher_traces": 0, "missing_ranking_states": 0,
        "teacher": dict(trace_counts), "ranking": dict(ranking_counts),
        "training_examples": len(training), "unique_training_feature_label_pairs": len(unique_pairs),
        "deduplicated_feature_packet_bytes": 7 + (len(unique_pairs) * 9 + 7) // 8,
        "scope": "Exact source-file binding and independent finite Horn replay. Native Lean has a separate receipt. "
                 "The curriculum target equals the conjunction of the first three public symbolic features.",
    }
    if selected_path is not None:
        result["selected_certificate"] = _selected_audit(selected_path, training)
    result["audit_sha256"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.tasks, args.traces, args.ranking, args.selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in
                     ("passed", "registered_roots", "teacher", "ranking", "training_examples")}, indent=2))


if __name__ == "__main__":
    main()
