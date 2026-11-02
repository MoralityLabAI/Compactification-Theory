"""Training canary: LoRA on the frozen, packed-resident Bonsai 2 27B, on the RTX 5080.

Steps: load the packed base; attach LoRA; evaluate the frozen base (B = 0, so
the adapted model equals the base exactly); train with AdamW + grad accumulation
+ per-layer activation checkpointing; evaluate on held-out val rows; save the
adapter, zero it, reload it from disk and check the reloaded eval matches;
export a GGUF LoRA for the PrismML runtime. Resumes from its own checkpoint, so
`vramorama run --restart N` can restart it after a GPU reset.

  python scripts/canary.py --config configs/canary_reverse_text.json
"""
import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bonsai_gym.adapter_io import load_adapter, save_adapter, write_gguf_lora  # noqa: E402
from bonsai_gym.data import Sampler, collate, load_examples  # noqa: E402
from bonsai_gym.model import attach_lora, load_packed_qwen35, lora_state  # noqa: E402
from bonsai_gym.runinfo import run_info  # noqa: E402


class GpuSampler(threading.Thread):
    """nvidia-smi every `period` s: utilization, memory used, board power."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop = period, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,enforced.power.limit",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                u, m, p, lim = (float(x) for x in out.strip().split(","))
                self.samples.append((time.time(), u, m, p, lim))
            except Exception:
                pass
            self.stop.wait(self.period)

    def summary(self, since=0.0):
        s = [x for x in self.samples if x[0] >= since]
        if not s:
            return {}
        return {"util_mean": sum(x[1] for x in s) / len(s), "mem_used_max_mib": max(x[2] for x in s),
                "power_mean_w": sum(x[3] for x in s) / len(s), "power_limit_min_w": min(x[4] for x in s), "n": len(s)}


def masked_loss(model, ids, labels):
    """Mean NLL over target positions; logits only at those positions."""
    h, _ = model.hidden(ids)
    sel = labels != -100
    logits = model.logits(h[sel])
    tgt = labels[sel]
    return F.cross_entropy(logits, tgt, reduction="sum"), logits, tgt


@torch.no_grad()
def evaluate(model, examples, pad_id, bs):
    nll, n_tok, n_correct, n_em = 0.0, 0, 0, 0
    per = []
    for i in range(0, len(examples), bs):
        batch = examples[i:i + bs]
        ids, lab = collate(batch, pad_id, "cuda")
        h, _ = model.hidden(ids)
        for b, e in enumerate(batch):
            sel = lab[b] != -100
            lg = model.logits(h[b][sel])
            tgt = lab[b][sel]
            l = float(F.cross_entropy(lg, tgt, reduction="sum"))
            ok = (lg.argmax(-1) == tgt)
            nll += l
            n_tok += int(tgt.numel())
            n_correct += int(ok.sum())
            n_em += int(bool(ok.all()))
            per.append({"id": e.id, "nll": l, "tokens": int(tgt.numel()), "em": bool(ok.all())})
    return {"loss": nll / n_tok, "token_acc": n_correct / n_tok, "tf_greedy_em": n_em / len(examples),
            "tokens": n_tok, "items": len(examples)}, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def log(rec):
        rec["t"] = time.time()
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps(rec), flush=True)

    torch.manual_seed(cfg["seed"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(cfg["tokenizer"])
    pad_id = tok.token_to_id("<|endoftext|>")
    train, sk_tr = load_examples(ROOT / cfg["train"], tok, cfg["max_len"])
    val, sk_va = load_examples(ROOT / cfg["val"], tok, cfg["max_len"])
    val = val[: cfg.get("n_val", len(val))]
    # confirmatory split (held-out task families): scored at step 0 and once at the end, never during training
    confirm = load_examples(ROOT / cfg["confirm"], tok, cfg["max_len"])[0] if cfg.get("confirm") else None
    gpu = GpuSampler()
    gpu.start()

    t0 = time.time()
    model = load_packed_qwen35(cfg["gguf"], chunk_elems=cfg.get("chunk_elems", 1 << 26), log=lambda m: log({"event": "load", "msg": m}))
    load_s = time.time() - t0
    base_mib = torch.cuda.memory_allocated() / 2**20
    n_lora = attach_lora(model, cfg["rank"], cfg["alpha"], cfg["targets"], cfg["seed"], cfg.get("adapter_quant", "none"))
    params = model.lora_parameters()
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg.get("weight_decay", 0.0))
    total = cfg["steps"]
    warm = cfg.get("warmup_steps", 0)

    def lr_at(step):
        if step < warm:
            return cfg["lr"] * (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return cfg["lr"] * (cfg.get("min_lr_frac", 0.1) + (1 - cfg.get("min_lr_frac", 0.1)) * 0.5 * (1 + math.cos(math.pi * prog)))

    sampler = Sampler(len(train), cfg["seed"])
    ckpt = out / "checkpoint.pt"
    start, history = 0, {}
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cuda", weights_only=False)
        for p_, v in zip(params, st["params"]):
            p_.data.copy_(v)
        opt.load_state_dict(st["opt"])
        sampler.load(st["sampler"])
        start, history = st["step"], st["history"]
        log({"event": "resumed", "step": start})
    log({"event": "setup", "load_s": load_s, "base_allocated_mib": base_mib, "lora_params": n_lora,
         "train_items": len(train), "val_items": len(val), "skipped_over_max_len": [sk_tr, sk_va],
         "allocated_after_lora_mib": torch.cuda.memory_allocated() / 2**20})

    if "base_eval" not in history:
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        ev, per = evaluate(model, val, pad_id, cfg.get("eval_batch", 4))
        ev["s"] = time.time() - t1
        ev["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        history["base_eval"] = ev
        (out / "eval_base_per_item.json").write_text(json.dumps(per), encoding="utf-8")
        log({"event": "eval", "step": 0, "frozen_base": True, **ev})
        if confirm:
            cev, cper = evaluate(model, confirm, pad_id, cfg.get("eval_batch", 4))
            history["base_confirm"], history["base_confirm_per"] = cev, cper
            log({"event": "confirm", "step": 0, "frozen_base": True, **cev})

    mb, accum = cfg["microbatch"], cfg["grad_accum"]
    torch.cuda.reset_peak_memory_stats()
    train_t0, train_tokens, seq_tokens = time.time(), 0, 0
    for step in range(start, total):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        batch_idx = sampler.next(mb * accum)
        n_tgt = 0
        losses = []
        for a in range(accum):
            batch = [train[i] for i in batch_idx[a * mb:(a + 1) * mb]]
            ids, lab = collate(batch, pad_id, "cuda")
            n_tgt += int((lab != -100).sum())
        loss_sum = 0.0
        for a in range(accum):
            batch = [train[i] for i in batch_idx[a * mb:(a + 1) * mb]]
            ids, lab = collate(batch, pad_id, "cuda")
            seq_tokens += int(ids.numel())
            lsum, _, _ = masked_loss(model, ids, lab)
            (lsum / n_tgt).backward()
            loss_sum += float(lsum)
        grads_finite = all(bool(torch.isfinite(p_.grad).all()) for p_ in params if p_.grad is not None)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 1.0)))
        if not grads_finite or not math.isfinite(loss_sum):
            log({"event": "nonfinite", "step": step, "loss_sum": loss_sum, "grads_finite": grads_finite})
            raise SystemExit(3)
        opt.step()
        torch.cuda.synchronize()
        train_tokens += n_tgt
        dt = time.time() - ts
        rec = {"event": "step", "step": step + 1, "loss": loss_sum / n_tgt, "lr": lr_at(step), "grad_norm": gnorm,
               "target_tokens": n_tgt, "step_s": dt, "seq_tok_s": seq_tokens / (time.time() - train_t0),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        log(rec)
        history.setdefault("steps", []).append(rec)
        if (step + 1) % cfg.get("eval_every", 10) == 0 or step + 1 == total:
            ev, _ = evaluate(model, val, pad_id, cfg.get("eval_batch", 4))
            history.setdefault("evals", []).append({"step": step + 1, **ev})
            log({"event": "eval", "step": step + 1, **ev})
        if (step + 1) % cfg.get("ckpt_every", 10) == 0 and step + 1 < total:
            tmp = out / "checkpoint.tmp"
            torch.save({"params": [p_.detach() for p_ in params], "opt": opt.state_dict(),
                        "sampler": sampler.state(), "step": step + 1, "history": history}, tmp)
            tmp.replace(ckpt)
    train_s = time.time() - train_t0
    train_gpu = gpu.summary(since=train_t0)

    final = history["evals"][-1]
    meta = {"config": cfg, "step": total, "final_eval": final}
    saved = save_adapter(model, out / "adapter.safetensors", meta)
    # reload check: wipe the adapter in place, load it back from disk, re-evaluate
    for p_ in params:
        p_.data.zero_()
    zeroed, _ = evaluate(model, val[:8], pad_id, cfg.get("eval_batch", 4))
    n_loaded = load_adapter(model, out / "adapter.safetensors")
    reloaded, per = evaluate(model, val, pad_id, cfg.get("eval_batch", 4))
    (out / "eval_final_per_item.json").write_text(json.dumps(per), encoding="utf-8")
    gg = write_gguf_lora(lora_state(model), out / "adapter.gguf", cfg["alpha"])
    confirm_res = None
    if confirm:
        from bonsai_gym.gym import paired_gain
        cev, cper = evaluate(model, confirm, pad_id, cfg.get("eval_batch", 4))
        b = {r["id"]: r["nll"] / r["tokens"] for r in history["base_confirm_per"]}
        a = {r["id"]: r["nll"] / r["tokens"] for r in cper}
        ids = sorted(set(a) & set(b))
        # gain = drop in per-item mean NLL (base - adapted), paired over items
        confirm_res = {"base": history["base_confirm"], "final": cev,
                       "nll_drop": paired_gain([a[i] for i in ids], [b[i] for i in ids])}
        log({"event": "confirm", "step": total, **cev})
    gpu.stop.set()
    steps = history.get("steps", [])
    result = {
        "canary": cfg["name"], "completed_steps": total, "optimizer_updates": len(steps),
        "lora_params": n_lora, "base_packed_allocated_mib": base_mib,
        "base_eval": history["base_eval"], "final_eval": final, "evals": history.get("evals", []),
        "reload": {"modules_loaded": n_loaded, "zeroed_eval_8": zeroed, "reloaded_eval": reloaded,
                   "matches_final_loss": abs(reloaded["loss"] - final["loss"]) < 1e-6,
                   "matches_final_em": reloaded["tf_greedy_em"] == final["tf_greedy_em"]},
        "adapter_safetensors": saved, "adapter_gguf": gg, "confirm": confirm_res,
        "train": {"wall_s": train_s, "target_tokens": train_tokens, "seq_tokens": seq_tokens,
                  "seq_tok_s": seq_tokens / train_s if train_s else None,
                  "target_tok_s": train_tokens / train_s if train_s else None,
                  "first_loss": steps[0]["loss"] if steps else None, "last_loss": steps[-1]["loss"] if steps else None,
                  "peak_allocated_mib": max((s["peak_allocated_mib"] for s in steps), default=None),
                  "peak_reserved_mib": max((s["peak_reserved_mib"] for s in steps), default=None),
                  "max_grad_norm": max((s["grad_norm"] for s in steps), default=None), "gpu": train_gpu},
        "run": run_info(args, gguf=cfg["gguf"]),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log({"event": "done", "reload_matches": result["reload"]["matches_final_loss"]})
    if ckpt.exists():
        ckpt.unlink()


if __name__ == "__main__":
    main()
