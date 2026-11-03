"""Check Algernon's experiment registration without downloading models or using a GPU.

Run: python -m research.algernon.validate_register
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REGISTER = HERE / "experiment_register.json"
HEX_SHA = re.compile(r"^[0-9a-f]{40}$")


def validate(data: dict[str, Any]) -> list[str]:
    """Return every registration problem. Incomplete source pins are allowed for planning."""
    problems: list[str] = []
    if data.get("schema_version") != "0.1.0":
        problems.append("unsupported schema_version")
    if data.get("project") != "Algernon-01":
        problems.append("wrong project")
    if data.get("status") != "registered_unrun":
        problems.append("initial scaffold must be registered_unrun")
    if data.get("run_count") != 0 or data.get("observed_results") != []:
        problems.append("initial scaffold cannot claim executed experiments")
    if data.get("dataset_splits_frozen") is not False:
        problems.append("dataset_splits_frozen must be false until split hashes exist")
    if data.get("source_shas_frozen") is not False:
        problems.append("source_shas_frozen must be false until all source hashes exist")

    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        problems.append("sources must contain a nonempty list")
        sources = []
    names: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            problems.append("source must be an object")
            continue
        name = source.get("name")
        if not isinstance(name, str) or not name:
            problems.append("source name is empty")
        elif name in names:
            problems.append(f"duplicate source: {name}")
        else:
            names.add(name)
        revision = source.get("revision")
        if revision is not None and (
            not isinstance(revision, str) or HEX_SHA.fullmatch(revision) is None
        ):
            problems.append(f"invalid source SHA for {name}")
        if not str(source.get("repo", "")).startswith("https://"):
            problems.append(f"missing HTTPS source URL for {name}")

    hypotheses = data.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 4:
        problems.append("expected four registered hypotheses")
        hypotheses = []
    ids = []
    for h in hypotheses:
        if not isinstance(h, dict):
            problems.append("hypothesis must be an object")
            continue
        ids.append(h.get("id"))
        if h.get("status") != "unrun":
            problems.append(f"hypothesis {h.get('id')} must remain unrun")
        controls = h.get("required_controls")
        if not isinstance(controls, list) or len(controls) < 2 or len(set(map(str, controls))) != len(controls):
            problems.append(f"hypothesis {h.get('id')} needs distinct controls")
        for key in ("name", "primary_metric", "decision_gate"):
            if not isinstance(h.get(key), str) or not h.get(key).strip():
                problems.append(f"hypothesis {h.get('id')} lacks {key}")
    if set(ids) != {"H-C", "H-R", "H-W", "H-J"} or len(ids) != len(set(ids)):
        problems.append("hypothesis IDs must be H-C/H-R/H-W/H-J once each")

    sites = data.get("site_candidates_08b")
    steps = data.get("recurrence_depths")
    if sites != [8, 12, 15, 19, 21]:
        problems.append("the registered site sweep changed")
    if steps != [0, 1, 2, 4, 8]:
        problems.append("the registered recurrence schedule changed")

    equipment = data.get("equipment_constraints", {})
    if equipment.get("gpu_vram_gb") != 16:
        problems.append("document the VRAM assumption before changing it")
    if equipment.get("qwenputer_4b_rv32_bf16_reported_checkpoint_gb", 0) <= equipment.get("gpu_vram_gb", 0):
        problems.append("4B checkpoint VRAM warning lost")

    for required_file in ("README.md", "ARCHITECTURE.md", "PROTOCOL.md", "AGENT_WORKORDER.md"):
        if not (HERE / required_file).is_file():
            problems.append(f"missing adjacent document: {required_file}")
    return problems


def main() -> int:
    data = json.loads(REGISTER.read_text(encoding="utf-8"))
    problems = validate(data)
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    print("PASS: Algernon-01 scaffold registered; no experiment reported executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
