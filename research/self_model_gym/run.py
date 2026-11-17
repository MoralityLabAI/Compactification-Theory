"""SELF-01 experiment runner. Default runs are synthetic smoke tests only."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .backends import HuggingFaceBackend, ScriptedBackend
from .env import SelfModelEnv
from .metrics import aggregate_episode
from .protocol import (CONDITIONS, PRIMARY_COMPARATOR, PROTOCOL_ID, SYSTEM_MESSAGE,
                       TURNS, canonical_json, protocol_digest, sha256_text)


def stable_seed(root: int, replicate: int, turn: int = 0) -> int:
    return int(sha256_text(f"{root}|{replicate}|{turn}")[:15], 16)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(backend, *, replicates: int, root_seed: int, output: Path,
             save_traces: bool = False) -> dict:
    if not (1 <= replicates <= 1000) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("replicates must be [1,1000] and seed must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("output directory must be empty to preserve run evidence")
    manifest = {"protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_digest(),
                "backend": backend.name, "evidence_type": ("synthetic_smoke" if backend.name == "scripted"
                                                          else "real_model_inference"),
                "replicates": replicates, "root_seed": root_seed,
                "conditions": list(CONDITIONS), "turns": TURNS,
                "save_traces": save_traces, "trace_kind": ("synthetic" if backend.name == "scripted"
                                                         else "teacher_forced_generated_token_layer"),
                "model_id": getattr(backend, "model_id", None),
                "revision": getattr(backend, "revision", None),
                "layer": getattr(backend, "actual_layer", None),
                "quantized_4bit": getattr(backend, "load_in_4bit", None),
                "max_new_tokens": getattr(backend, "max_new_tokens", None),
                "generation": {"do_sample": True, "temperature": 0.7,
                               "top_p": 0.95, "thinking": False} if backend.name == "hf" else None,
                "library_versions": getattr(backend, "library_versions", None)}
    (output / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    env = SelfModelEnv()
    pairs = [(rep, cond) for rep in range(replicates) for cond in CONDITIONS]
    random.Random(root_seed).shuffle(pairs)
    episode_rows = []
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as file:
        for rep, cond in pairs:
            episode_seed = stable_seed(root_seed, rep)
            observation, info = env.reset(seed=episode_seed, options={"condition": cond})
            steps = []
            terminated = False
            while not terminated:
                turn = observation["turn"]
                messages = [{"role": "system", "content": SYSTEM_MESSAGE}, *observation["history"],
                            {"role": "user", "content": observation["prompt"]}]
                sample = backend.generate(messages, seed=stable_seed(root_seed, rep, turn+1),
                                          condition=cond, turn=turn)
                prompt = observation["prompt"]
                observation, reward, terminated, truncated, info = env.step(
                    {"text": sample.text, "trace": sample.trace, "trace_source": sample.trace_source})
                step = {"turn": turn, "prompt_sha256": sha256_text(prompt),
                        "response": sample.text, "response_sha256": sha256_text(sample.text),
                        "generation_tokens": sample.generation_tokens,
                        "trace_source": sample.trace_source, "metrics": info["metrics"]}
                if save_traces:
                    step["trace"] = sample.trace
                steps.append(step)
            row = {"replicate": rep, "condition": cond, "episode_seed": episode_seed,
                   "steps": steps, "aggregate": aggregate_episode([x["metrics"] for x in steps])}
            file.write(canonical_json(row) + "\n")
            episode_rows.append(row)
    grouped = {cond: [r for r in episode_rows if r["condition"] == cond] for cond in CONDITIONS}
    per_condition = {}
    for cond, rows in grouped.items():
        values = [r["aggregate"]["primary_mean_recurrence"] for r in rows
                  if r["aggregate"]["primary_mean_recurrence"] is not None]
        per_condition[cond] = {"episodes": len(rows), "valid_primary": len(values),
                               "mean_primary": sum(values) / len(values) if values else None}
    by_pair = {(r["replicate"], r["condition"]): r for r in episode_rows}
    differences = []
    for rep in range(replicates):
        a = by_pair[(rep, "self_reference")]["aggregate"]["primary_mean_recurrence"]
        b = by_pair[(rep, PRIMARY_COMPARATOR)]["aggregate"]["primary_mean_recurrence"]
        if a is not None and b is not None:
            differences.append(a - b)
    receipt = {"status": "complete", "evidence_type": manifest["evidence_type"],
               "protocol_sha256": protocol_digest(), "manifest_sha256": file_sha256(output / "manifest.json"),
               "episodes_sha256": file_sha256(output / "episodes.jsonl"),
               "episodes": len(episode_rows), "expected_episodes": replicates * len(CONDITIONS),
               "per_condition": per_condition, "paired_n": len(differences),
               "primary_paired_difference": (sum(differences)/len(differences) if differences else None),
               "source_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    (output / "receipt.json").write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["scripted", "hf"], default="scripted")
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--output", type=Path, default=Path("runs/self-01-smoke"))
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--trace-layer", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--confirm-real-inference", action="store_true")
    args = parser.parse_args()
    if args.backend == "hf":
        if not args.confirm_real_inference or not args.model_id or not args.revision:
            parser.error("hf backend needs --confirm-real-inference, --model-id, and pinned --revision")
        backend = HuggingFaceBackend(args.model_id, args.revision, trace_layer=args.trace_layer,
                                    max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)
    else:
        backend = ScriptedBackend()
    result = evaluate(backend, replicates=args.replicates, root_seed=args.seed,
                      output=args.output, save_traces=args.save_traces)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
