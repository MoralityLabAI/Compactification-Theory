"""Omniscient minimum-read verdict oracle for SA-02's fixed certificate grammar."""
import argparse
import json
from collections import deque
from pathlib import Path
from gym import BUDGETS, SPLITS, Session, check_certificate, evidence, families, refs, validate

def closure(graph, roots, children):
    seen, todo = set(), list(roots)
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        todo.extend(children(graph[n]))
    return seen

def shortest_forbidden(graph, root):
    todo = deque([(root, (root,))])
    seen = set()
    while todo:
        node, path = todo.popleft()
        if node in seen:
            continue
        seen.add(node)
        if graph[node]["kind"] == "axiom":
            return path
        todo.extend((x, path + (x,)) for x in graph[node].get("deps", []))
    return None

def minimum_nodes(graph, arm):
    root, target = arm["root"], arm["target"]
    path = shortest_forbidden(graph, root)
    target_nodes = closure(graph, [target], refs)
    candidates = []
    if path is not None:
        candidates.append(set(path))
    if arm["gold"]["semantic"] == "false":
        candidates.append(target_nodes)
    if candidates:
        return min(candidates, key=lambda x: (len(x), sorted(x)))
    proof = closure(graph, [root], lambda node: node.get("deps", []))
    source = closure(graph, [graph[root]["type"]], refs)
    return proof | source | target_nodes

def replay_oracle(graph, arm):
    selected = minimum_nodes(graph, arm)
    session = Session(graph, arm["root"], arm["target"], len(selected))
    while selected - set(session.observed):
        frontier = (selected & set(session.discovered)) - set(session.observed)
        assert frontier, "unreachable minimum certificate"
        session.read(min(frontier))
    result, _, _ = evidence(session.root, session.target, session.observed)
    assert result["verdict"] == arm["gold"]["verdict"]
    check_certificate(graph, arm, session.observed, result)
    return {"id": arm["id"], "minimum_reads": len(selected),
            "verdict": result["verdict"], "schedule": list(session.observed)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fs = list(families())
    validate(fs)
    summary = []
    for split in SPLITS:
        records = [replay_oracle(f["graph"], arm) for f in fs if f["split"] == split for arm in f["arms"]]
        (args.out / (split + "-oracle.json")).write_text(json.dumps(records, indent=2))
        for budget in BUDGETS:
            summary.append({"split": split, "budget": budget, "n": len(records),
                            "oracle_feasible": sum(x["minimum_reads"] <= budget for x in records),
                            "mean_minimum_reads": sum(x["minimum_reads"] for x in records) / len(records)})
    (args.out / "oracle-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
