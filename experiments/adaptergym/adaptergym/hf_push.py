"""Upload trained bonsai-gym adapters to the Hugging Face Hub (private repos by default) with a model card built
from each run's result.json and, where the run has one, its eval-split score summary.

  python -m adaptergym.hf_push adapters.json [--public] [--dry-run]

adapters.json: [{"repo": "AlephFunk/<name>", "run": "<run dir with adapter.gguf + result.json>",
                 "base": "<base model: hub id or file name>", "base_gguf": "<local base GGUF, hashed into the card>",
                 "about": "<one paragraph: what the adapter is trained on>", "scores": ["<scores/<arm>/summary.json>", ...]}]

Uses the login saved by `hf auth login`; reads no token itself. --dry-run writes the cards next to each run
(hf_card.md) and uploads nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = ("adapter.gguf", "adapter.safetensors", "result.json", "train_log.jsonl")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def card(a: dict) -> str:
    run = Path(a["run"])
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    cfg_name = res.get("canary", run.name)
    base_ev, fin_ev = res.get("base_eval", {}), res.get("final_eval", {})
    lines = [
        "---", "library_name: llama.cpp", "tags: [lora, gguf, bonsai, ternary, bitcoin, bitagent]",
        f"base_model: {a['base']}" if "/" in a["base"] else "", "---", "",
        f"# {a['repo'].split('/')[-1]}", "", a["about"], "",
        "## Files", "",
        f"- `adapter.gguf`: llama.cpp LoRA (`llama-server -m <base> --lora adapter.gguf`), sha256 `{sha256(run / 'adapter.gguf')}`",
        "- `adapter.safetensors`: the same adapter in PEFT-style tensors", "- `result.json`, `train_log.jsonl`: the run record",
        "", "## Base", "", f"- {a['base']}",
    ]
    if a.get("base_gguf"):
        lines.append(f"- base GGUF `{Path(a['base_gguf']).name}`, sha256 `{sha256(Path(a['base_gguf']))}`")
    lines += ["", "## Training", "",
              f"- config `{cfg_name}`; {res.get('completed_steps')} optimizer steps; {res.get('lora_params'):,} LoRA parameters",
              f"- held-out loss {base_ev.get('loss', float('nan')):.4f} (base) -> {fin_ev.get('loss', float('nan')):.4f} (adapter), "
              f"teacher-forced greedy exact match {base_ev.get('tf_greedy_em', 0):.3f} -> {fin_ev.get('tf_greedy_em', 0):.3f}",
              f"- trained at commit `{res.get('run', {}).get('git_commit', '?')}` of MoralityLabAI/Compactification-Theory "
              f"(experiments/bonsai-gym)"]
    for sp in a.get("scores", []):
        p = Path(sp)
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            arm = p.parent.name
            fams = ", ".join(f"{f} {v['mean']:.3f} (n={v['n']})" for f, v in s.items())
            lines += [f"- eval split, arm `{arm}`: {fams}"]
    return "\n".join(l for l in lines if l is not None) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    items = json.loads(Path(a.manifest).read_text(encoding="utf-8-sig"))
    api = None
    if not a.dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
        print("hub user:", api.whoami()["name"], flush=True)
    for it in items:
        run = Path(it["run"])
        if not (run / "adapter.gguf").exists():
            print("skip (not trained yet)", it["repo"], flush=True)
            continue
        text = card(it)
        (run / "hf_card.md").write_text(text, encoding="utf-8")
        if a.dry_run:
            print("card", run / "hf_card.md", flush=True)
            continue
        api.create_repo(it["repo"], private=not a.public, exist_ok=True)
        for f in FILES:
            if (run / f).exists():
                api.upload_file(path_or_fileobj=str(run / f), path_in_repo=f, repo_id=it["repo"])
        api.upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo="README.md", repo_id=it["repo"])
        local = sha256(run / "adapter.gguf")
        remote = api.get_paths_info(it["repo"], ["adapter.gguf"])[0].lfs.sha256
        print("pushed", it["repo"], "adapter.gguf sha256 match" if local == remote else f"SHA MISMATCH {local} {remote}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
