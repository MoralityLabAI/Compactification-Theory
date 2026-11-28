"""Independently replay behavioral and optional activation metrics from a receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import aggregate_episode, behavioral_metrics, trace_metrics
from .protocol import CONDITIONS, PROMPTS, TURNS, protocol_digest, sha256_text
from .run import file_sha256, stable_seed


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert manifest["protocol_sha256"] == protocol_digest() == receipt["protocol_sha256"]
    assert receipt["manifest_sha256"] == file_sha256(output / "manifest.json")
    assert receipt["episodes_sha256"] == file_sha256(output / "episodes.jsonl")
    assert manifest["evidence_type"] == receipt["evidence_type"]
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {(rep, cond) for rep in range(manifest["replicates"]) for cond in CONDITIONS}
    actual = {(row["replicate"], row["condition"]) for row in rows}
    assert actual == expected and len(rows) == len(expected) == receipt["episodes"]
    for row in rows:
        assert row["episode_seed"] == stable_seed(manifest["root_seed"], row["replicate"])
        assert len(row["steps"]) == TURNS
        for turn, step in enumerate(row["steps"]):
            assert step["turn"] == turn
            assert step["prompt_sha256"] == sha256_text(PROMPTS[row["condition"]][turn])
            assert step["response_sha256"] == sha256_text(step["response"])
            assert all(step["metrics"][key] == value for key, value
                       in behavioral_metrics(step["response"]).items())
            if "trace" in step:
                assert all(step["metrics"][key] == value for key, value
                           in trace_metrics(step["trace"]).items())
            else:
                # Without traces, activation metrics are recorded but not independently verifiable.
                assert step["metrics"]["trace_tokens"] <= 64
        assert row["aggregate"] == aggregate_episode([step["metrics"] for step in row["steps"]])
    return {"status": "hash_and_structure_verified", "episodes": len(rows),
            "activation_metrics_recomputed": manifest["save_traces"],
            "independent_model_replay": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
