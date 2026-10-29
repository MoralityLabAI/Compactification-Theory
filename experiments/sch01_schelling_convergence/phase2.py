"""SCH-01 Phase 2: auditable transmission graph and pairing-blinded coding packets.

Human source-language review and genuinely independent annotation remain external gates.
The script uses only the Python standard library, no network and no model inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
import random
from typing import Any

ROOT = Path(__file__).resolve().parent
AXES = ("stage", "time", "agency", "identity", "ultimate", "destination", "authority")
UNKNOWN = {"", "unknown", "disputed", "multi"}
EDGE_CLASSES = {"transmission", "scholarly_analysis", "comparison"}
EDGE_STATUSES = {"documented", "interpretive", "unverified"}
FIELDS = ("passage_key", "source_url", "source_locator", "language", "edition", "synthetic_text") + AXES + ("confidence", "notes")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_graph(data: dict[str, Any]) -> None:
    nodes = data["nodes"]
    edges = data["edges"]
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate graph node")
    by_id = {n["id"]: n for n in nodes}
    for node in nodes:
        if not isinstance(node["min_year"], int) or not isinstance(node["max_year"], int):
            raise ValueError("noninteger year")
        if node["min_year"] > node["max_year"]:
            raise ValueError("invalid chronology interval")
        if not str(node.get("source_url", "")).startswith("https://"):
            raise ValueError("node without traceable source")
    pairs = set()
    for edge in edges:
        u, v = edge["from"], edge["to"]
        if u not in by_id or v not in by_id:
            raise ValueError("dangling graph edge")
        if u == v or (u, v) in pairs:
            raise ValueError("duplicate or self-directed edge")
        pairs.add((u, v))
        if edge["edge_class"] not in EDGE_CLASSES or edge["status"] not in EDGE_STATUSES:
            raise ValueError("invalid evidence status or relation")
        if not edge["evidence_url"].startswith("https://"):
            raise ValueError("missing edge citation")
        if edge["status"] == "documented" and not edge.get("evidence_scope"):
            raise ValueError("documented edge requires source scope")
        if by_id[u]["min_year"] > by_id[v]["max_year"]:
            raise ValueError("chronologically impossible influence or analysis")
    adjacency = defaultdict(list)
    for e in edges:
        if e["edge_class"] == "transmission" and e["status"] == "documented":
            adjacency[e["from"]].append(e["to"])
    visited, visiting = set(), set()

    def walk(node: str) -> None:
        if node in visiting:
            raise ValueError("cycle in documented transmission graph")
        if node in visited:
            return
        visiting.add(node)
        for nxt in adjacency[node]:
            walk(nxt)
        visiting.remove(node)
        visited.add(node)

    for node in ids:
        walk(node)


def transmission_path(graph: dict[str, Any], start: str, target: str) -> list[str] | None:
    """Documented, causal/transmission edges only. Analysis links cannot imply influence."""
    verify_graph(graph)
    nodes = {n["id"] for n in graph["nodes"]}
    if start not in nodes or target not in nodes:
        raise ValueError("unknown path endpoint")
    routes = defaultdict(list)
    for e in graph["edges"]:
        if e["status"] == "documented" and e["edge_class"] == "transmission":
            routes[e["from"]].append(e["to"])
    queue = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == target:
            return path
        for nxt in sorted(routes[path[-1]]):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(path + [nxt])
    return None


def graph_report(graph: dict[str, Any]) -> dict[str, Any]:
    verify_graph(graph)
    counts = Counter((e["edge_class"], e["status"]) for e in graph["edges"])
    queries = {"plotinus_to_falsafa": ("PLOTINUS", "FALSAFA"),
               "bible_to_witnesses": ("APOCALYPTIC_BIBLE", "JW_PRESENT"),
               "watchtower_to_kurzweil": ("BIBLE_STUDENTS", "KURZWEIL_2005"),
               "advaita_to_kurzweil": ("SHANKARA", "KURZWEIL_2005"),
               "dao_to_kurzweil": ("DAO_DE_JING", "KURZWEIL_2005")}
    return {"evidence_grade": "dated_public_source_ledger_not_independence_proof",
            "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
            "edge_counts": {f"{a}:{b}": v for (a, b), v in sorted(counts.items())},
            "documented_transmission_paths": {name: transmission_path(graph, *pair) for name, pair in queries.items()},
            "unverified_queries": graph["unverified_queries"],
            "warning": "No path is an evidence-absence statement, not proof of independent origin."}


def verify_passages(data: dict[str, Any]) -> None:
    cases = data["passages"]
    if len(cases) < 2 or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("insufficient or duplicate passages")
    for c in cases:
        if c["provenance"] not in {"primary", "secondary", "synthetic"}:
            raise ValueError("invalid passage provenance")
        if not all(isinstance(c[k], str) and c[k].strip() for k in ("id", "locator", "language", "edition")):
            raise ValueError("incomplete passage locator")
        if c["provenance"] == "synthetic":
            if c["source_url"] is not None or not c.get("synthetic_text"):
                raise ValueError("synthetic case improperly sourced")
        elif not (isinstance(c["source_url"], str) and c["source_url"].startswith("https://")):
            raise ValueError("historical case missing source URL")


def make_packets(data: dict[str, Any], output: Path) -> dict[str, Any]:
    """Coders see source and locator, with proposed pairings and seed labels withheld.

    Published web sources reveal the tradition; hence pairing-blind, not source-blind.
    """
    verify_passages(data)
    output.mkdir(parents=True, exist_ok=True)
    cards = data["passages"]
    # IDs remain stable across coders for mechanical agreement, original IDs hidden.
    aliases = {p["id"]: f"P{i+1:03d}" for i, p in enumerate(cards)}
    for side, seed in (("A", 17), ("B", 71)):
        shuffled = list(cards)
        random.Random(seed).shuffle(shuffled)
        with (output / f"coder_{side}.csv").open("w", encoding="utf-8", newline="") as fd:
            writer = csv.DictWriter(fd, fieldnames=FIELDS)
            writer.writeheader()
            for p in shuffled:
                writer.writerow({"passage_key": aliases[p["id"]], "source_url": p["source_url"] or "",
                    "source_locator": p["locator"], "language": p["language"],
                    "edition": p["edition"], "synthetic_text": p.get("synthetic_text", "")})
    # This manifest contains the alias key and should be kept with study management,
    # away from reviewers, until their independent coding is frozen.
    manifest = {"phase": "SCH-01B", "type": "researcher_mapping_keep_from_coders",
                "alias_to_source_id": {v: k for k, v in aliases.items()},
                "coder_ids": ["A", "B"], "coded_rows": 0,
                "independent_coder_observations": False}
    (output / "researcher_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"prepared": len(cards), "coders": 2, "coded_rows": 0,
            "status": "uncoded_templates_only", "source_blinded": False, "pairing_blinded": True}


def read_codes(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as fd:
        rd = csv.DictReader(fd)
        if not rd.fieldnames or set(FIELDS) - set(rd.fieldnames):
            raise ValueError("missing coding columns")
        result = {}
        for row in rd:
            key = row["passage_key"]
            if not key or key in result:
                raise ValueError("invalid or duplicate passage key")
            result[key] = row
        return result


def score_code_files(a_file: Path, b_file: Path) -> dict[str, Any]:
    if a_file.resolve() == b_file.resolve():
        raise ValueError("coders require distinct files")
    a, b = read_codes(a_file), read_codes(b_file)
    if not a or set(a) != set(b):
        raise ValueError("different passage sets")
    # This is an audit of supplied annotation files. Human independence is not
    # technically verifiable and must be recorded separately by investigators.
    per_axis = {}
    all_complete = True
    for axis in AXES:
        pairs = [(a[k][axis].strip().lower(), b[k][axis].strip().lower()) for k in sorted(a)]
        eligible = [(v, w) for v, w in pairs if v not in UNKNOWN and w not in UNKNOWN
                    and not v.startswith("multi:") and not w.startswith("multi:")]
        complete = sum(bool(v) and bool(w) for v, w in pairs)
        all_complete &= complete == len(pairs)
        if not eligible:
            per_axis[axis] = {"eligible": 0, "raw_agreement": None, "cohen_kappa": None}
            continue
        agreement = sum(v == w for v, w in eligible) / len(eligible)
        pa, pb = Counter(v for v, _ in eligible), Counter(w for _, w in eligible)
        expected = sum(pa[k] * pb[k] for k in set(pa) | set(pb)) / len(eligible) ** 2
        kappa = None if expected >= 1.0 else (agreement - expected) / (1 - expected)
        per_axis[axis] = {"eligible": len(eligible), "raw_agreement": agreement, "cohen_kappa": kappa}
    return {"cases": len(a), "coding_status": "supplied_labels_unverified_independence" if all_complete else "incomplete_annotations",
            "human_independence_verified": False, "axes": per_axis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("graph", "prepare", "score"))
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "phase2_packets")
    parser.add_argument("--coder-a", type=Path)
    parser.add_argument("--coder-b", type=Path)
    args = parser.parse_args()
    if args.action == "graph":
        result = graph_report(load(ROOT / "influence_graph.json"))
    elif args.action == "prepare":
        result = make_packets(load(ROOT / "passages.json"), args.out)
    else:
        if args.coder_a is None or args.coder_b is None:
            parser.error("score requires --coder-a and --coder-b")
        result = score_code_files(args.coder_a, args.coder_b)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
