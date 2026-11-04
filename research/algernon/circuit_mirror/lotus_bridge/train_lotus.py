"""H-A2c trainer: upstream LOTUS objects in a single-GPU loop that survives power-cap pauses.

`--phase cot` is upstream's stage-1 objective on a plain causal LM: question,
steps and answer, with the question masked. `--phase lotus` wraps the stage-1
weights in upstream `Lotus` and runs the curriculum:
stage = min(epoch // epochs_per_stage, max_latent_stage). Stage s replaces the
first s CoT steps with s blocks of c_thought latents and runs s loops.

The model class, `get_dataset`, `get_cot_latent_dataset`, `MyCollator`, the
latent-token initialisation and the single-GPU loss (`outputs.loss`) are
upstream's (`scripts/run.py` at the pinned commit). This file supplies the loop
around them: one process, no FSDP, and a checkpoint at most every --ckpt-secs
and at every epoch end. It also pauses (exit 75) when the card is power-capped,
and a restart resumes with the same epoch, the same batch order (the shuffle
seed is a function of the epoch) and the RNG states restored. Upstream maps
datasets with num_proc=32, which spawns 32 interpreters on Windows; the same
map functions run in one process here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import types
from pathlib import Path

from .capture import UPSTREAM_SHA, file_sha256, import_lotus_module, verify_upstream
from .gpu_run import PAUSE_EXIT, Paused, PowerWatch, log, upstream_extract_answer, write_atomic

LATENT_TOKENS = ("<|start-latent|>", "<|end-latent|>", "<|latent|>")  # run.py adds them in this order
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def import_upstream_dataset(upstream: Path):
    spec = importlib.util.spec_from_file_location("algernon_lotus_dataset_pinned", upstream / "scripts" / "dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def single_process_map():
    import datasets
    if getattr(datasets.Dataset.map, "_algernon_single_process", False):
        return
    orig = datasets.Dataset.map

    def map_(self, *a, **k):
        k.pop("num_proc", None)
        return orig(self, *a, **k)
    map_._algernon_single_process = True
    datasets.Dataset.map = map_


def tokenizer_and_base(base_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.pad_token = tok.eos_token
    for t in LATENT_TOKENS:
        tok.add_tokens(t)
    return tok, AutoModelForCausalLM.from_pretrained(base_dir)


def latent_ids(tok):
    return {t: tok.convert_tokens_to_ids(t) for t in LATENT_TOKENS}


def init_latent_rows(base, tok):
    """run.py: grow the vocabulary and start the three new rows at the embedding of "<<"."""
    import torch
    base.resize_token_embeddings(len(tok))
    emb = base.get_input_embeddings()
    src = tok.convert_tokens_to_ids("<<")
    with torch.no_grad():
        for tid in latent_ids(tok).values():
            emb.weight.data[tid] = emb.weight.data[src]
            base.lm_head.weight.data[tid] = base.lm_head.weight.data[src]


def wrap_lotus(lm, tok, base, *, c_thought, after_loop, inter_weight):
    ids = latent_ids(tok)
    return lm.Lotus(base, latent_token_id=ids["<|latent|>"], start_latent_id=ids["<|start-latent|>"],
                    end_latent_id=ids["<|end-latent|>"], eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id, c_thought=c_thought, use_kv_cache=True,
                    intermediate_loss_weight=inter_weight, intermediate_loss_after_loop=after_loop,
                    flat_intermediate_supervision=False, latent_injection_mode="add")


def load_trained(upstream: Path, base_dir, checkpoint: Path, *, device):
    """(model, lotus module, tokenizer) for a final.pt this trainer wrote; eval mode."""
    import torch
    lm = import_lotus_module(upstream)
    tok, base = tokenizer_and_base(base_dir)
    meta = json.loads(checkpoint.with_name("train_manifest.json").read_text(encoding="utf-8"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if meta["phase"] == "cot":
        base.load_state_dict(state)
        model = base
    else:
        base.resize_token_embeddings(len(tok))
        model = wrap_lotus(lm, tok, base, c_thought=meta["c_thought"], after_loop=meta["after_loop"],
                           inter_weight=meta["intermediate_loss_weight"])
        model.load_state_dict(state)
    return model.to(device).eval(), lm, tok


def question_ids(tok, row):
    return tok.encode(row["question"] + "\n", add_special_tokens=True)


def answers(model, tok, rows, *, kind, blocks=0, loops=0, c_thought=0, max_new_tokens=16, extract):
    """Greedy answers. kind "cot": the causal LM decodes steps and answer itself.
    kind "lotus": question, start, blocks*c_thought latents, end, then upstream Lotus.generate."""
    import torch
    device = next(model.parameters()).device
    out = []
    with torch.inference_mode():
        for row in rows:
            q = question_ids(tok, row)
            if kind == "cot":
                x = torch.tensor([q], device=device)
                gen = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False,
                                     max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.eos_token_id)
            else:
                ids = latent_ids(tok)
                x = torch.tensor([q + [ids["<|start-latent|>"]] + [ids["<|latent|>"]] * (blocks * c_thought)
                                  + [ids["<|end-latent|>"]]], device=device)
                gen = model.generate(x, torch.ones_like(x), max_new_tokens=max_new_tokens, n_looped_iters=loops)
            text = tok.decode(gen[0, x.shape[1]:].tolist(), skip_special_tokens=True)
            out.append({"answer_text": text, "answer": extract(text), "truth": row["answer"]})
    return out


def accuracy(rows):
    return sum(r["answer"] == r["truth"] for r in rows) / max(len(rows), 1)


def configs_for(args, *, cot):
    return types.SimpleNamespace(pad_latent_to_max=True, max_latent_stage=0 if cot else args.max_latent_stage,
                                 c_thought=0 if cot else args.c_thought, uniform_prob=0.0, no_cot=False,
                                 fixed_latent_tokens=0)


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    import numpy as np
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path: Path, model, opt, epoch, step, last_epoch):
    import torch
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "step": step,
                "last_epoch": last_epoch, "rng": rng_state()}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--base", required=True, help="GPT-2 snapshot directory")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", choices=("cot", "lotus"), required=True)
    ap.add_argument("--init", type=Path, help="lotus phase: stage-1 final.pt")
    ap.add_argument("--after-loop", choices=("true", "false"), help="lotus phase: intermediate_loss_after_loop")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--extend-to", type=int, default=0, help="cot: keep training to this epoch while val < --extend-below")
    ap.add_argument("--extend-below", type=float, default=0.95)
    ap.add_argument("--c-thought", type=int, default=6)
    ap.add_argument("--max-latent-stage", type=int, default=8)
    ap.add_argument("--epochs-per-stage", type=int, default=1)
    ap.add_argument("--intermediate-loss-weight", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=128, help="rows per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=16, help="rows per forward/backward (gradient accumulation)")
    ap.add_argument("--tf32", choices=("on", "off"), default="on", help="TF32 matmuls on CUDA")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every-epoch", type=int, default=64, help="validation cases scored at each epoch end")
    ap.add_argument("--ckpt-secs", type=float, default=120)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-power-w", type=float, default=50)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (tests)")
    args = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader

    cot = args.phase == "cot"
    if not cot and (args.init is None or args.after_loop is None):
        ap.error("--phase lotus needs --init and --after-loop")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path, ckpt_path, log_path = (args.out / n for n in ("train_manifest.json", "ckpt.pt", "train_log.jsonl"))
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if old.get("status") == "complete":
        log("already complete")
        return
    src_hash = verify_upstream(args.upstream)
    data_meta = json.loads((args.data_dir / "pointer_data.json").read_text(encoding="utf-8"))
    manifest = {"schema_version": "algernon-02c-train-v1", "phase": args.phase, "status": "running",
                "argv": [Path(a).name if ("/" in a or "\\" in a) else a for a in sys.argv],
                "upstream": {"commit": UPSTREAM_SHA, "lotus_py_sha256": src_hash,
                             "dataset_py_sha256": file_sha256(args.upstream / "scripts" / "dataset.py")},
                "base": {"model_id": "openai-community/gpt2", "revision": GPT2_REVISION},
                "data": {k: data_meta[k] for k in ("train_sha256", "val_sha256", "parameters")},
                "init_sha256": file_sha256(args.init) if args.init else None,
                "c_thought": 0 if cot else args.c_thought, "after_loop": None if cot else args.after_loop == "true",
                "intermediate_loss_weight": 0.0 if cot else args.intermediate_loss_weight,
                "max_latent_stage": 0 if cot else args.max_latent_stage,
                "hyper": {k: getattr(args, k) for k in ("epochs", "extend_to", "extend_below", "epochs_per_stage",
                                                        "lr", "weight_decay", "batch", "micro_batch", "tf32",
                                                        "clip", "seed")},
                "epochs_done": old.get("epochs_done", []), "attempts": old.get("attempts", 0) + 1,
                "pauses": old.get("pauses", [])}

    def write(status=None):
        if status:
            manifest["status"] = status
        write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    single_process_map()
    ds = import_upstream_dataset(args.upstream)
    lm = import_lotus_module(args.upstream)
    extract = upstream_extract_answer(args.upstream)
    tok, base = tokenizer_and_base(args.base)
    ids = latent_ids(tok)
    if cot:
        model = base
    else:
        base.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
        init_latent_rows(base, tok)
        model = wrap_lotus(lm, tok, base, c_thought=args.c_thought, after_loop=args.after_loop == "true",
                           inter_weight=args.intermediate_loss_weight)
    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, start_step, last_epoch = 0, 0, args.epochs - 1
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch, start_step, last_epoch = ck["epoch"], ck["step"], ck["last_epoch"]
        set_rng_state(ck["rng"])
        log(f"resumed at epoch {start_epoch} step {start_step}")
    val_rows = json.loads((args.data_dir / "pointer_val.json").read_text(encoding="utf-8"))
    base_train = ds.get_dataset(str(args.data_dir / "pointer_train.json"), tok)
    cfg = configs_for(args, cot=cot)
    collate = ds.MyCollator(tok, latent_id=ids["<|latent|>"], label_pad_token_id=-100)
    watch = PowerWatch(args.min_power_w, active=str(args.device).startswith("cuda"))
    keys = ("input_ids", "attention_mask", "labels", "position_ids") if cot else None
    last_ckpt, total_steps = time.monotonic(), 0
    epoch, cur_step = start_epoch, start_step
    try:
        watch.check("start")
        while epoch <= last_epoch:
            stage = 0 if cot else min(epoch // args.epochs_per_stage, args.max_latent_stage)
            data = ds.get_cot_latent_dataset(stage, base_train, cfg, ids["<|start-latent|>"], ids["<|latent|>"],
                                             ids["<|end-latent|>"], no_special_marker=cot, shuffle=True,
                                             shuffle_seed=args.seed + 100003 * epoch)
            # Its own generator: iterating a DataLoader draws from the global torch RNG otherwise, which
            # would shift every later dropout mask after a mid-epoch resume.
            loader = DataLoader(data, batch_size=args.batch, shuffle=False, collate_fn=collate, drop_last=True,
                                generator=torch.Generator().manual_seed(args.seed + epoch))
            model.train()
            t_epoch, losses = time.perf_counter(), []
            for step, batch in enumerate(loader):
                if epoch == start_epoch and step < start_step:
                    continue
                cur_step = step
                watch.check(f"epoch {epoch} step {step}")
                batch = {k: v.to(args.device) for k, v in batch.items()
                         if k not in ("idx", "answer_labels") and (keys is None or k in keys)}
                # One optimizer step per --batch rows, accumulated over --micro-batch slices of the collated
                # batch, each slice's loss weighted by its share of the rows.
                n = batch["input_ids"].shape[0]
                mb = args.micro_batch if args.micro_batch > 0 else n
                stat = {"loss": 0.0, "main_loss": 0.0, "intermediate_loss": 0.0}
                for i in range(0, n, mb):
                    sub = {k: v[i:i + mb] for k, v in batch.items()}
                    if not cot:
                        sub["n_looped_iters"] = stage
                    out = model(**sub)
                    w = sub["input_ids"].shape[0] / n
                    (out.loss * w).backward()
                    stat["loss"] += float(out.loss) * w
                    if not cot:
                        stat["main_loss"] += float(out.main_loss) * w
                        stat["intermediate_loss"] += float(out.intermediate_loss) * w
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float("inf"))
                opt.step()
                opt.zero_grad()
                total_steps += 1
                losses.append(stat["loss"])
                if step % 25 == 0:
                    rec = {"epoch": epoch, "stage": stage, "step": step, "loss": stat["loss"], "grad_norm": float(gn),
                           "seconds": round(time.perf_counter() - t_epoch, 1)}
                    if not cot:
                        rec |= {k: stat[k] for k in ("main_loss", "intermediate_loss")}
                    if str(args.device).startswith("cuda"):
                        rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                cur_step = step + 1
                if time.monotonic() - last_ckpt > args.ckpt_secs:
                    save_checkpoint(ckpt_path, model, opt, epoch, cur_step, last_epoch)
                    last_ckpt = time.monotonic()
                if args.max_steps and total_steps >= args.max_steps:
                    save_checkpoint(ckpt_path, model, opt, epoch, cur_step, last_epoch)
                    write()
                    log(f"stopped after {total_steps} steps (--max-steps)")
                    return
            model.eval()
            n_val = len(val_rows) if epoch >= args.epochs - 1 else args.val_every_epoch
            res = answers(model, tok, val_rows[:n_val], kind="cot" if cot else "lotus", blocks=stage, loops=stage,
                          c_thought=args.c_thought, max_new_tokens=8 * 8 + 16, extract=extract)
            ep = {"epoch": epoch, "stage": stage, "train_loss_mean": sum(losses) / max(len(losses), 1),
                  "seconds": round(time.perf_counter() - t_epoch, 1), "val_n": n_val, "val_accuracy": accuracy(res),
                  "val_accuracy_by_depth": {d: accuracy([r for r, v in zip(res, val_rows) if v["depth"] == d])
                                            for d in range(1, 9)}}
            manifest["epochs_done"] = [e for e in manifest["epochs_done"] if e["epoch"] != epoch] + [ep]
            log(f"epoch {epoch} stage {stage}: loss {ep['train_loss_mean']:.4f}, val {ep['val_accuracy']:.3f} on {n_val}")
            epoch, cur_step = epoch + 1, 0
            if cot and epoch > last_epoch and epoch < args.extend_to and ep["val_accuracy"] < args.extend_below:
                last_epoch = epoch
            save_checkpoint(ckpt_path, model, opt, epoch, 0, last_epoch)
            last_ckpt = time.monotonic()
            write()
        final = args.out / "final.pt"
        torch.save(model.state_dict(), final)
        manifest["final_sha256"] = file_sha256(final)
        manifest["epochs_trained"] = epoch
        write("complete")
        log(f"done: {epoch} epochs, final {manifest['final_sha256'][:12]}")
    except Paused as e:
        save_checkpoint(ckpt_path, model, opt, epoch, cur_step, last_epoch)
        manifest["pauses"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reason": str(e)})
        write("paused_power_cap")
        log(f"PAUSE: {e}; exit {PAUSE_EXIT}")
        sys.exit(PAUSE_EXIT)


if __name__ == "__main__":
    main()
