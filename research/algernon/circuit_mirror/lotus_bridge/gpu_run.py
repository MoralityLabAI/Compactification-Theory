"""Algernon-02B GPU job: reproduce LOTUS, check the capture hook, capture the roster.

One process, run in order, each stage gated on the previous one:

1. Environment, upstream source pin, checkpoint file hashes, GPU power state.
2. Unmodified upstream scripts/eval.py on a fixed seeded GSM8K test sample.
3. Load the checkpoint, fingerprint its weights.
4. Hook parity on the first roster case at full depth: upstream outputs with the
   hook installed, with logits suppressed and after removal must be bitwise
   identical to the unhooked run; readout ids must equal upstream's
   intermediate-logit argmax; lm_head on captured vectors must match upstream's
   intermediate logits at those positions.
5. Two-case smoke (forward path), then the frozen calibration subset, then the
   full roster and the relabelled counterfactual roster (generate path, so the
   captured states are the ones the model's own answer is decoded from).
6. Behaviour: answers at the trained 6 loops and with no latent tokens.
7. Drift: recapture case 0 and compare bitwise; refingerprint the weights.

Stops (status "aborted" in the manifest) on parity failure, nonfinite states,
allocated memory growing past --growth-mib over the first case, peak memory
over --vram-budget-gib, drift, or a weight change. Never downloads: needs the
pinned snapshot in the local Hugging Face cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from .capture import (UPSTREAM_SHA, DEFAULT_MODEL, capture_case, file_sha256, import_lotus_module,
                      intercept_loop_states, latent_input, load_model, verify_upstream)
from .counterfactual import validate_counterfactual
from .roster import prompt_for, validate_roster

PAPER_GSM8K = 0.7005  # upstream README, yingfanbot/gsm-lotus-llama3b on GSM8k-Aug test
GSM8K_FLOOR = 0.55    # 200 items: the 95% interval around 0.70 is about +-0.065
SUBSET_PER_FAMILY = 4


class Abort(RuntimeError):
    pass


def scrub(arg) -> str:
    """Command-line element without local directories, for committed receipts."""
    arg = str(arg)
    return Path(arg).name if ("/" in arg or "\\" in arg) else arg


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tensor_sha(t):
    import torch
    t = t.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def weights_fingerprint(base):
    h, seen = hashlib.sha256(), set()
    for name, t in sorted(base.state_dict().items()):
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        h.update(name.encode())
        h.update(tensor_sha(t).encode())
    return h.hexdigest()


def unique_params(base):
    seen, n = set(), 0
    for p in base.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            n += p.numel()
    return n


def nvidia_smi():
    q = "name,driver_version,memory.total,memory.used,enforced.power.limit,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    return dict(zip(q.split(","), [x.strip() for x in out.split(",")]))


def environment():
    import numpy, torch, transformers
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "transformers": transformers.__version__, "numpy": numpy.__version__,
           "nvidia_smi": nvidia_smi()}
    root = Path(__file__).resolve().parents[4]
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()
    env["code"] = {"repo": "MoralityLabAI/Compactification-Theory", "commit": git("rev-parse", "HEAD"),
                   "circuit_mirror_dirty": bool(git("status", "--porcelain", "--", "research/algernon/circuit_mirror"))}
    return env


def parity_suite(model, module, tok, question, *, loops, c_thought):
    """Bitwise checks that capturing changes nothing upstream computes."""
    import torch
    ids, first, last, _ = latent_input(model, tok, question, loops=loops, c_thought=c_thought)
    device = next(model.base_causallm.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    args = (x, torch.ones_like(x), torch.full_like(x, -100), torch.arange(len(ids), device=device)[None])
    original = module.efficient_forward

    def plain():
        with torch.inference_mode():
            return model.forward(*args, n_looped_iters=loops)

    def digests(out):
        return {"final": tensor_sha(out.logits), "inter": [tensor_sha(t) for t in out.intermediate_logits]}

    a = plain()
    ref = digests(a)
    n_inter = len(a.intermediate_logits)
    ref_argmax = [t[0].argmax(-1).to(torch.int32).cpu() for t in a.intermediate_logits]
    last_pos = [(b + 1) * c_thought for b in range(loops)]  # loop-region index of each block's last latent
    ref_rows = [t[0, last_pos, :].float().cpu() for t in a.intermediate_logits]
    del a
    kw = dict(first_latent=first, last_latent=last, c_thought=c_thought, record_blocks=True)
    with torch.inference_mode(), intercept_loop_states(module, suppress_logits=False, **kw) as full:
        b = model.forward(*args, n_looped_iters=loops)
    hooked = digests(b)
    del b
    with torch.inference_mode(), intercept_loop_states(module, suppress_logits=True, **kw) as light:
        c = model.forward(*args, n_looped_iters=loops)
    light_final = tensor_sha(c.logits)
    del c
    restored = module.efficient_forward is original
    again = digests(plain())
    lm = model.base_causallm.lm_head
    max_diff, agree, total, readout_equal = 0.0, 0, 0, True
    for p in range(1, loops + 1):
        rows = full[p]["blocks"][:, 0, :].to(device=device, dtype=lm.weight.dtype)
        with torch.inference_mode():
            got = lm(rows).float().cpu()
        max_diff = max(max_diff, float((got - ref_rows[p-1]).abs().max()))
        agree += int((got.argmax(-1) == ref_rows[p-1].argmax(-1)).sum())
        total += got.shape[0]
        readout_equal &= bool(torch.equal(full[p]["readout"], ref_argmax[p-1]))
    feats_equal = all(torch.equal(f[k], g[k]) for f, g in zip(full, light)
                      for k in ("pools", "blocks", "static"))
    static_drift = max(float((e["static"] - full[0]["static"]).abs().max()) for e in full)
    checks = {
        "captured_passes_equal_loops_plus_one": len(full) == loops + 1 == len(light),
        "upstream_intermediate_logits_one_per_loop": n_inter == loops,
        "hook_installed_outputs_bitwise_identical": hooked == ref,
        "hook_removed_and_outputs_bitwise_identical": restored and again == ref,
        "logit_suppression_keeps_hidden_bitwise": feats_equal,
        "logit_suppression_keeps_final_logits_bitwise": light_final == ref["final"],
        "readout_ids_equal_upstream_intermediate_argmax": readout_equal,
        "lm_head_on_captured_vectors_argmax_agreement_ge_0.98": agree / total >= 0.98,
        "lm_head_on_captured_vectors_max_abs_diff_lt_1": max_diff < 1.0,
    }
    return {"loops": loops, "c_thought": c_thought, "sequence_tokens": len(ids), "checks": checks,
            "passed": all(checks.values()), "lm_head_max_abs_diff": max_diff,
            "lm_head_argmax_agreement": agree / total, "static_token_max_drift_across_passes": static_drift,
            "reference_digests": ref}


def gsm8k_repro(args, snapshot: Path, run_dir: Path):
    """Run upstream eval.py unmodified on a seeded sample; returns accuracy record."""
    source = args.upstream / "data" / "gsm_test.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    idx = sorted(random.Random(args.seed).sample(range(len(data)), args.gsm8k_n))
    sample = [{"idx": i, "question": data[i]["question"], "answer": data[i]["answer"]} for i in idx]
    sample_path = run_dir / "gsm8k_sample.json"
    sample_path.write_text(json.dumps(sample, indent=1) + "\n", encoding="utf-8")
    preds = run_dir / "gsm8k_preds.json"
    cmd = [sys.executable, str(args.upstream / "scripts" / "eval.py"), "--model_id", str(snapshot),
           "--datasets", str(sample_path), "--n_looped_iters", "6", "--c_thought", "25",
           "--bf16", "--device", args.device, "--save_preds", str(preds)]
    t0 = time.perf_counter()
    with (run_dir / "gsm8k_eval.log").open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=3600,
                              env={**__import__("os").environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"})
    seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        raise Abort(f"upstream eval.py exited {proc.returncode}; see gsm8k_eval.log")
    rows = next(iter(json.loads(preds.read_text(encoding="utf-8")).values()))
    n, k = len(sample), sum(r["correct"] for r in rows)
    from .probe import wilson
    lo, hi = wilson(k / n, n)
    text = (run_dir / "gsm8k_eval.log").read_text(encoding="utf-8", errors="replace")
    peak = [ln.strip() for ln in text.splitlines() if "peak GPU mem" in ln]
    return {"command": ["python"] + [scrub(c) for c in cmd[1:]],
            "source_file_sha256": file_sha256(source), "sample_sha256": file_sha256(sample_path),
            "sample_seed": args.seed, "n": n, "scored_rows": len(rows), "correct": k,
            "accuracy": k / n, "wilson_95": [lo, hi], "paper_accuracy": PAPER_GSM8K,
            "paper_inside_wilson_95": lo <= PAPER_GSM8K <= hi, "seconds": round(seconds, 2),
            "upstream_peak_memory_line": peak}


def upstream_extract_answer(upstream: Path):
    spec = importlib.util.spec_from_file_location("algernon_lotus_eval_pinned", upstream / "scripts" / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_answer


class Recorder:
    """Per-case CUDA memory and time gates."""

    def __init__(self, device, budget_gib, growth_mib):
        self.cuda = str(device).startswith("cuda")
        self.budget = budget_gib * 2**30
        self.growth = growth_mib * 2**20
        self.baseline = None

    def check(self, label):
        if not self.cuda:
            return {}
        import torch
        torch.cuda.synchronize()
        alloc, peak = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
        if self.baseline is None:
            self.baseline = alloc
        if alloc - self.baseline > self.growth:
            raise Abort(f"{label}: allocated memory grew {(alloc-self.baseline)/2**20:.0f} MiB")
        if peak > self.budget:
            raise Abort(f"{label}: peak allocated {peak/2**30:.2f} GiB over budget")
        return {"allocated_bytes": alloc, "peak_allocated_bytes": peak}


def run_cases(rows, label, model, module, tok, args, rec, extract, *, generate):
    import numpy as np
    keep = {"pools": [], "blocks": [], "static": [], "readout": []}
    per_case = []
    t_stage = time.perf_counter()
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        res, meta = capture_case(model, module, tok, prompt_for(row), loops=args.loops,
                                 c_thought=args.c_thought,
                                 max_new_tokens=args.max_new_tokens if generate else 0)
        mem = rec.check(f"{label}:{row['id']}")
        meta.update(mem, id=row["id"], split=row["split"], family=row["family"],
                    seconds=round(time.perf_counter() - t0, 4))
        if res["generated"] is not None:
            text = tok.decode(res["generated"], skip_special_tokens=True)
            meta.update(answer_text=text, answer=extract(text), generated_tokens=len(res["generated"]))
        meta["readout_text"] = {
            kind: [tok.decode(res["readout"][p, b*args.c_thought:(b+1)*args.c_thought].tolist(),
                              skip_special_tokens=False) for b, p in blocks]
            for kind, blocks in (("final_pass", [(b, args.loops) for b in range(args.loops)]),
                                 ("diagonal", [(b, b + 1) for b in range(args.loops)]),
                                 ("pass0", [(b, 0) for b in range(args.loops)]))}
        for k in keep:
            keep[k].append(res[k])
        per_case.append(meta)
        if (i + 1) % 8 == 0 or i + 1 == len(rows):
            log(f"{label}: {i+1}/{len(rows)} ({time.perf_counter()-t_stage:.0f}s)")
    arrays = {k: np.stack(v) for k, v in keep.items()}
    return arrays, per_case, time.perf_counter() - t_stage


def save_capture(path: Path, arrays, ids):
    import numpy as np
    big = float(max(np.abs(arrays["blocks"]).max(), np.abs(arrays["static"]).max()))
    if big >= 6.0e4:
        raise Abort(f"hidden magnitude {big} does not fit float16 storage")
    np.savez(path, blocks=arrays["blocks"].astype(np.float16), static=arrays["static"].astype(np.float16),
             readout=arrays["readout"].astype(np.int32), ids=np.array(ids))
    return {"file": path.name, "sha256": file_sha256(path), "max_abs_hidden": big,
            "blocks_shape": list(arrays["blocks"].shape), "storage": "float16 (lossless for bf16 within range)"}


def save_legacy(path: Path, pools, ids, roster, args, revision, src_hash, per_case, parity, complete):
    """features/ids npz plus manifest in exactly the layout probe.verify_capture reads."""
    import numpy as np
    np.savez_compressed(path, features=pools.astype(np.float32), ids=np.array(ids))
    record = {"schema_version": "algernon-lotus-hidden-v1", "status": "actual_lotus_forward_observations",
              "source": {"repo": "https://github.com/yingfan-bot/lotus", "commit": UPSTREAM_SHA,
                         "lotus_py_sha256": src_hash, "model_id": args.model_id, "model_revision": revision},
              "roster_sha256": roster["roster_sha256"], "roster_file_sha256": file_sha256(args.roster),
              "cases": len(ids), "complete_roster": complete, "feature_sha256": file_sha256(path),
              "device": args.device, "precision": args.precision, "loops": args.loops,
              "c_thought": args.c_thought, "feature_shape": list(pools.shape),
              "feature_pools": ["last_latent", "mean_last_block"], "logit_capture": "suppressed, no answer grading",
              "instrumentation_parity": {"full_depth_suite": parity["checks"], "passed": parity["passed"]},
              "per_case": [{k: m[k] for k in ("id", "split", "seconds", "input_sha256")} for m in per_case]}
    path.with_suffix(".manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n",
                                                  encoding="utf-8")


def behaviour(rows, model, tok, args, extract, *, loops):
    """Answers with `loops` latent loops via upstream Lotus.generate, or with loops=0
    greedy decoding of the bare causal LM on the question alone, as upstream
    eval.py --cot does. (Lotus.generate without latents cannot follow a latent
    run under transformers 4.46: the cached DynamicCache is an nn.Module child,
    and the no-latent branch then assigns it a tuple.)"""
    import torch
    device = next(model.base_causallm.parameters()).device
    out, t0 = [], time.perf_counter()
    for row in rows:
        if loops:
            ids = latent_input(model, tok, prompt_for(row), loops=loops, c_thought=args.c_thought)[0]
        else:
            ids = tok.encode(prompt_for(row), add_special_tokens=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            if loops:
                gen = model.generate(x, torch.ones_like(x), max_new_tokens=args.max_new_tokens, n_looped_iters=loops)
            else:
                gen = model.base_causallm.generate(x, attention_mask=torch.ones_like(x), do_sample=False,
                                                   max_new_tokens=args.explicit_tokens, eos_token_id=tok.eos_token_id,
                                                   pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, len(ids):].tolist(), skip_special_tokens=True)
        out.append({"id": row["id"], "loops": loops, "answer_text": text, "answer": extract(text),
                    "generated_tokens": int(gen.shape[1] - len(ids)), "sequence_tokens": len(ids)})
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--counterfactual", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--hf-revision", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--loops", type=int, default=16)
    ap.add_argument("--c-thought", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--explicit-tokens", type=int, default=128)
    ap.add_argument("--gsm8k-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--vram-budget-gib", type=float, default=11.0)
    ap.add_argument("--cuda-fraction", type=float, default=0.70,
                    help="torch per-process cap, so Windows cannot silently spill past the lease")
    ap.add_argument("--growth-mib", type=float, default=512)
    ap.add_argument("--min-power-w", type=float, default=50)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU plumbing check: tiny loops, parity + one case, no GSM8K, no generate")
    args = ap.parse_args()
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    if args.dry_run:
        args.loops, args.c_thought, args.gsm8k_n = 2, 2, 0
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "run_manifest.json"
    manifest = {"schema_version": "algernon-02b-gpu-run-v1", "status": "running",
                "argv": [scrub(a) for a in sys.argv],
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": {},
                "device": args.device, "precision": args.precision, "loops": args.loops,
                "c_thought": args.c_thought, "max_new_tokens": args.max_new_tokens,
                "explicit_tokens": args.explicit_tokens}

    def write(status=None):
        if status:
            manifest["status"] = status
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        manifest["environment"] = environment()
        roster = json.loads(args.roster.read_text(encoding="utf-8"))
        validate_roster(roster)
        cf = json.loads(args.counterfactual.read_text(encoding="utf-8"))
        validate_counterfactual(cf, roster)
        if args.loops != roster["parameters"]["steps"] and not args.dry_run:
            raise Abort("--loops must equal the roster's steps")
        src_hash = verify_upstream(args.upstream)
        snapshot = Path(snapshot_download(args.model_id, revision=args.hf_revision, local_files_only=True))
        manifest["source"] = {"repo": "https://github.com/yingfan-bot/lotus", "commit": UPSTREAM_SHA,
                              "lotus_py_sha256": src_hash, "model_id": args.model_id,
                              "model_revision": args.hf_revision,
                              "checkpoint_files_sha256": {p.name: file_sha256(p) for p in sorted(snapshot.iterdir())
                                                          if p.is_file()}}
        manifest["inputs"] = {"roster_sha256": roster["roster_sha256"], "roster_file_sha256": file_sha256(args.roster),
                              "counterfactual_sha256": cf["counterfactual_sha256"],
                              "counterfactual_file_sha256": file_sha256(args.counterfactual)}
        cuda = args.device.startswith("cuda")
        if cuda:
            power = manifest["environment"]["nvidia_smi"].get("enforced.power.limit", "0 W")
            if float(power.split()[0]) < args.min_power_w:
                raise Abort(f"GPU power-capped at {power}; refusing to start")
            torch.cuda.set_per_process_memory_fraction(args.cuda_fraction, 0)
        write()
        if cuda and args.gsm8k_n:
            log("stage gsm8k: upstream eval.py")
            manifest["stages"]["gsm8k"] = gsm8k_repro(args, snapshot, args.run_dir)
            log(f"gsm8k {manifest['stages']['gsm8k']['accuracy']:.4f} on {args.gsm8k_n}")
            write()
            if manifest["stages"]["gsm8k"]["accuracy"] < GSM8K_FLOOR:
                raise Abort(f"upstream GSM8K reproduction below {GSM8K_FLOOR}; environment suspect")
        t0 = time.perf_counter()
        model, module, tok = load_model(args.upstream, model_id=args.model_id, revision=args.hf_revision,
                                        device=args.device, dtype=args.precision, local_only=True,
                                        c_thought=args.c_thought)
        fp0 = weights_fingerprint(model.base_causallm)
        manifest["model"] = {"load_seconds": round(time.perf_counter() - t0, 2),
                             "unique_parameters": unique_params(model.base_causallm),
                             "weights_fingerprint_start": fp0,
                             "latent_token_ids": [model.start_latent_id, model.latent_token_id, model.end_latent_id],
                             "tokenizer_len": len(tok)}
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            manifest["model"]["allocated_after_load_bytes"] = torch.cuda.memory_allocated()
        write()
        log("stage parity")
        parity = parity_suite(model, module, tok, prompt_for(roster["cases"][0]), loops=args.loops,
                              c_thought=args.c_thought)
        manifest["stages"]["parity"] = parity
        (args.run_dir / "parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
        write()
        if not parity["passed"]:
            raise Abort(f"parity failed: {[k for k, v in parity['checks'].items() if not v]}")
        rec = Recorder(args.device, args.vram_budget_gib, args.growth_mib)
        extract = upstream_extract_answer(args.upstream)
        smoke_rows = roster["cases"][:1 if args.dry_run else 2]
        smoke, smoke_meta, sec = run_cases(smoke_rows, "smoke", model, module, tok, args, rec, extract,
                                           generate=False)
        save_legacy(args.run_dir / "smoke.npz", smoke["pools"], [r["id"] for r in smoke_rows], roster,
                    args, args.hf_revision, src_hash, smoke_meta, parity, False)
        manifest["stages"]["smoke"] = {"seconds": round(sec, 2), "per_case": smoke_meta,
                                       "capture": save_capture(args.run_dir / "smoke_blocks.npz", smoke,
                                                               [r["id"] for r in smoke_rows])}
        write()
        if args.dry_run:
            fp1 = weights_fingerprint(model.base_causallm)
            manifest["model"]["weights_fingerprint_end"] = fp1
            write("dry_run_passed" if fp1 == fp0 else "aborted")
            log(f"dry run done: {manifest['status']}")
            return
        families = {}
        for r in roster["cases"]:
            families.setdefault(r["family"], []).append(r)
        subset_rows = [r for fam in families.values() for r in fam[:SUBSET_PER_FAMILY]]
        sub, sub_meta, sec = run_cases(subset_rows, "calibration_subset", model, module, tok, args, rec,
                                       extract, generate=True)
        manifest["stages"]["calibration_subset"] = {"ids": [r["id"] for r in subset_rows],
                                                    "seconds": round(sec, 2), "per_case": sub_meta}
        write()
        full, full_meta, sec = run_cases(roster["cases"], "full", model, module, tok, args, rec, extract,
                                         generate=True)
        ids = [r["id"] for r in roster["cases"]]
        pos = {i: n for n, i in enumerate(ids)}
        same = all(np.array_equal(smoke["blocks"][j], full["blocks"][pos[r["id"]]]) for j, r in enumerate(smoke_rows))
        same &= all(np.array_equal(sub["blocks"][j], full["blocks"][pos[r["id"]]]) for j, r in enumerate(subset_rows))
        manifest["stages"]["full"] = {"seconds": round(sec, 2), "per_case": full_meta,
                                      "smoke_and_subset_bitwise_equal_full": bool(same),
                                      "capture": save_capture(args.run_dir / "full_blocks.npz", full, ids)}
        save_legacy(args.run_dir / "full.npz", full["pools"], ids, roster, args, args.hf_revision, src_hash,
                    full_meta, parity, True)
        write()
        if not same:
            raise Abort("repeated captures differ (smoke/subset vs full)")
        cfa, cf_meta, sec = run_cases(cf["cases"], "counterfactual", model, module, tok, args, rec, extract,
                                      generate=True)
        manifest["stages"]["counterfactual"] = {"seconds": round(sec, 2), "per_case": cf_meta,
                                                "capture": save_capture(args.run_dir / "cf_blocks.npz", cfa,
                                                                        [r["id"] for r in cf["cases"]])}
        write()
        b6, s6 = behaviour(roster["cases"], model, tok, args, extract, loops=6)
        b0, s0 = behaviour(roster["cases"], model, tok, args, extract, loops=0)
        rec.check("behaviour")
        with (args.run_dir / "behaviour.jsonl").open("w", encoding="utf-8") as fh:
            for row in b6 + b0:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        manifest["stages"]["behaviour"] = {"loops6_seconds": round(s6, 2), "explicit_seconds": round(s0, 2),
                                           "file_sha256": file_sha256(args.run_dir / "behaviour.jsonl")}
        again, _, _ = run_cases(roster["cases"][:1], "drift", model, module, tok, args, rec, extract,
                                generate=True)
        fp1 = weights_fingerprint(model.base_causallm)
        manifest["model"]["weights_fingerprint_end"] = fp1
        manifest["stages"]["drift"] = {"case0_bitwise_equal": bool(np.array_equal(again["blocks"][0], full["blocks"][0])),
                                       "weights_unchanged": fp1 == fp0}
        if cuda:
            manifest["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            manifest["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        ok = all(manifest["stages"]["drift"].values())
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write("complete" if ok else "aborted")
        log(f"done: {manifest['status']}")
    except Abort as e:
        manifest["abort_reason"] = str(e)
        write("aborted")
        log(f"ABORT: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
