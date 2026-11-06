"""H-A3 (Phase 4): constraint-guided correction of the 02D D1 slot state under noise.

F is D1 frozen; its six fixed slots are the recurrent state. pi is D1's own LM head on slot 2, which spells the
state after the step (`<<s->t>>`, t at slot 2), restricted to the digits 0-5 and normalised. After loop j the slots
get Gaussian noise of scale sigma times their RMS. After every loop k, a corrector arm adds G(features) to the
slots before the next loop reads them:

    e_k = p_{k-1} @ M      the map applied to the previous decoded state (p_0 is the question's start)
    C   G(e_k - p_k)       the soft transition residual of the trace constraint s_k = f(s_{k-1})
    I   G([e_k, p_k])      the same two vectors unsubtracted: at least the information C has
    N   G(p_k)             no map, no constraint
    P   no corrector

G is one linear map to 6 x 768, initialised at zero, so every arm starts as P. Training uses generated
`random_function` cases and a local loss through F: cross-entropy of pi on the corrected slots after loop j and
after loop j + 1, against the true states. Evaluation scores the answer and, separately, the whole decoded trace
under the exact BN254 R1CS checker in `spec.py`.

    calibrate: plain accuracy on generated cases for each sigma in the grid; sigma* is the one closest to 0.5
    scale:     (H-A3b) mean clean slot RMS r on the calibration cases; kappa = sigma* x r
    train:     one corrector per arm at sigma*
    eval:      every arm on the frozen roster's 72 calibration and held-out cases

With `--out-scale noise` (H-A3b) every corrector's output is multiplied by kappa, so a weight of 1 moves a slot
element by about one noise std. The default, `unit`, is H-A3 as registered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from ..spec import PointerTask, build, case, witness_for_trace
from .capture import file_sha256
from .gpu_run import PAUSE_EXIT, Parts, Paused, PowerWatch, log, upstream_extract_answer, write_atomic
from .pointer_loop import prompt_nocount
from .slot_patch import SLOTS, iterate

T_SLOT = 2
N_STATES = 6
ARMS = ("P", "C", "I", "N")
SIGMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def digit_ids(tok):
    ids = [tok.encode(str(d)) for d in range(N_STATES)]
    if any(len(i) != 1 for i in ids):
        raise ValueError("each state digit must be one token")
    return [i[0] for i in ids]


def corrector(arm, hidden):
    import torch
    if arm == "P":
        return None
    lin = torch.nn.Linear(2 * N_STATES if arm == "I" else N_STATES, SLOTS * hidden)
    torch.nn.init.zeros_(lin.weight)
    torch.nn.init.zeros_(lin.bias)
    return lin


def features(arm, e, p):
    import torch
    return {"C": lambda: e - p, "I": lambda: torch.cat([e, p]), "N": lambda: p}[arm]()


def generated(seed, count, steps):
    """`random_function` cases on 6 nodes from spec.case, excluding the frozen roster and counterfactual."""
    root = Path(__file__).parent
    banned = {(tuple(c["successors"]), c["initial"]) for f in ("frozen_roster_6x16.json", "frozen_counterfactual_6x16.json")
              for c in json.loads((root / f).read_text(encoding="utf-8"))["cases"]}
    out, idx = [], 0
    while len(out) < count:
        t = case("random_function", seed, idx, n=N_STATES, steps=steps)
        idx += 1
        if (t.successors, t.initial) not in banned:
            out.append({"id": f"gen{seed}/{idx - 1}", "successors": list(t.successors), "initial": t.initial,
                        "family": "random_function", "split": "generated"})
    return out


class Recurrence:
    """Noise and correction inside upstream's loop, through its efficient_forward. Call 0 is the initial pass and
    call k the pass of loop k. Records pi's distribution after each loop (after noise and correction)."""

    def __init__(self, model, lm, digits, successors, start, *, arm="P", G=None, noise=None, scale=1.0,
                 record_rms=False):
        import torch
        self.model, self.lm, self.arm, self.G, self.noise = model, lm, arm, G, noise
        self.scale, self.record_rms, self.rms = scale, record_rms, []
        dev = next(model.parameters()).device
        self.head = model.base_causallm.lm_head.weight[digits]  # [6, H]
        self.M = torch.zeros(N_STATES, N_STATES, device=dev)
        self.M[torch.arange(N_STATES), torch.tensor(successors)] = 1.0
        self.p0 = torch.nn.functional.one_hot(torch.tensor(start), N_STATES).float().to(dev)
        self.probs = []

    def pi(self, slots):
        import torch
        return torch.softmax(slots[T_SLOT].float() @ self.head.float().T, -1)

    @contextmanager
    def hooked(self, first, last):
        original, calls, p_prev = self.lm.efficient_forward, [0], [self.p0]

        def fwd(*args, **kwargs):
            out = original(*args, **kwargs)
            k = calls[0]
            calls[0] += 1
            if k == 0:
                return out
            h = out.last_hidden_state
            offset = first - 1 if h.shape[1] == last + 2 - first else 0
            slots = h[0, first - offset:last - offset + 1]
            if self.record_rms:
                self.rms.append(float(slots.float().pow(2).mean().sqrt()))
            if self.noise is not None and k == self.noise["j"]:
                rms = slots.float().pow(2).mean().sqrt()
                slots = slots + (self.noise["sigma"] * rms * self.noise["eps"].to(slots.device)).to(slots.dtype)
            if self.G is not None:
                p = self.pi(slots)
                delta = self.G(features(self.arm, p_prev[0] @ self.M, p)).view(SLOTS, -1)
                if self.scale != 1.0:
                    delta = delta * self.scale
                slots = slots + delta.to(slots.dtype)
            p = self.pi(slots)
            self.probs.append(p)
            p_prev[0] = p
            h = h.clone()
            h[0, first - offset:last - offset + 1] = slots
            out.last_hidden_state = h
            return out

        self.lm.efficient_forward = fwd
        try:
            yield calls
        finally:
            self.lm.efficient_forward = original


def layout(model, tok, question):
    q = tok.encode(question + "\n", add_special_tokens=True)
    return q + [model.start_latent_id] + [model.latent_token_id] * SLOTS + [model.end_latent_id], len(q) + 1, len(q) + SLOTS


def answer(model, lm, tok, digits, row, *, loops, extract, arm="P", G=None, noise=None, scale=1.0):
    """Generate the answer with noise and correction; returns the answer and the decoded trace."""
    import torch
    ids, first, last = layout(model, tok, prompt_nocount(row).rstrip("\n"))
    x = torch.tensor([ids], device=next(model.parameters()).device)
    rec = Recurrence(model, lm, digits, row["successors"], row["initial"], arm=arm, G=G, noise=noise, scale=scale)
    with torch.inference_mode(), rec.hooked(first, last) as calls:
        gen = model.generate(x, torch.ones_like(x), max_new_tokens=16, n_looped_iters=loops)
    if calls[0] != loops + 1:
        raise AssertionError(f"expected initial + {loops} loop passes, got {calls[0]}")
    text = tok.decode(gen[0, x.shape[1]:].tolist(), skip_special_tokens=True)
    return extract(text), text, [row["initial"]] + [int(p.argmax()) for p in rec.probs]


def slot_rms(model, lm, tok, digits, row, *, loops):
    """Slot RMS after each loop of a clean plain run (before any noise or correction)."""
    import torch
    ids, first, last = layout(model, tok, prompt_nocount(row).rstrip("\n"))
    x = torch.tensor([ids], device=next(model.parameters()).device)
    rec = Recurrence(model, lm, digits, row["successors"], row["initial"], record_rms=True)
    with torch.inference_mode(), rec.hooked(first, last):
        model.generate(x, torch.ones_like(x), max_new_tokens=1, n_looped_iters=loops)
    return rec.rms


_CIRCUITS, _VALID = {}, {}


def trace_valid(row, trace):
    """Exact finite-field check of a decoded trace against the public machine (spec.build). Memoised on
    (map, start, trace): the check is deterministic, and identical decodes recur across arms and conditions."""
    key = (tuple(row["successors"]), row["initial"], tuple(trace))
    if key in _VALID:
        return _VALID[key]
    steps = len(trace) - 1
    if steps not in _CIRCUITS:
        _CIRCUITS[steps] = build(PointerTask(tuple(range(N_STATES)), 0, steps))
    task = PointerTask(tuple(row["successors"]), row["initial"], steps)
    w, public = witness_for_trace(task, _CIRCUITS[steps], trace)
    _VALID[key] = bool(_CIRCUITS[steps].check(w, public)["valid"])
    return _VALID[key]


def eps_for(rid, hidden):
    import torch
    g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(rid.encode()).digest()[:8], "little") % 2**63)
    return torch.randn(SLOTS, hidden, generator=g)


def train_step_loss(model, lm, tok, digits, row, j, *, arm, G, noise, scale=1.0):
    """Local loss through F: pi on the corrected slots after loop j and after loop j + 1."""
    import torch
    ids, first, last = layout(model, tok, prompt_nocount(row).rstrip("\n"))
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    rec = Recurrence(model, lm, digits, row["successors"], row["initial"], arm=arm, G=G, noise=noise, scale=scale)
    with rec.hooked(first, last):
        model.forward(x, torch.ones_like(x), torch.full_like(x, -100), torch.arange(len(ids), device=dev)[None],
                      n_looped_iters=j + 1)
    targets = [iterate(row["successors"], row["initial"], k) for k in (j, j + 1)]
    return sum(-torch.log(rec.probs[k - 1][t] + 1e-9) for k, t in zip((j, j + 1), targets))


def load(args):
    import torch
    from .train_lotus import load_trained
    meta = json.loads(args.checkpoint.with_name("train_manifest.json").read_text(encoding="utf-8"))
    if meta.get("fixed_slots") != SLOTS:
        raise SystemExit(f"{args.checkpoint} is not a {SLOTS}-slot loop-phase checkpoint")
    model, lm, tok = load_trained(args.upstream, args.base, args.checkpoint, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.cuda_fraction, 0)
    return meta, model, lm, tok, digit_ids(tok)


def calibrate(args, model, lm, tok, digits, extract, parts, watch, hidden):
    rows, results = generated(args.calib_seed, args.calib_cases, args.depth), {}
    for sigma in SIGMA_GRID:
        hits = []
        for r in rows:
            for j in args.noise_loops:
                rid = f"{r['id']}:j{j}:s{sigma}"
                hit = parts.load("calibrate", rid)
                if hit is None:
                    watch.check(rid)
                    noise = {"j": j, "sigma": sigma, "eps": eps_for(f"{r['id']}:j{j}", hidden)}
                    ans, _, _ = answer(model, lm, tok, digits, r, loops=args.depth, extract=extract, noise=noise)
                    hit = ({"correct": ans == str(iterate(r["successors"], r["initial"], args.depth))},)
                    parts.save("calibrate", rid, hit[0])
                hits.append(hit[0]["correct"])
        results[sigma] = sum(hits) / len(hits)
        log(f"calibrate sigma {sigma}: plain accuracy {results[sigma]:.3f}")
    star = min(SIGMA_GRID, key=lambda s: (abs(results[s] - 0.5), s))
    return {"plain_accuracy_by_sigma": results, "sigma_star": star, "cases": len(rows), "seed": args.calib_seed}


def measure_scale(args, model, lm, tok, digits, extract, parts, watch, hidden, sigma):
    """H-A3b: r = mean clean slot RMS over the calibration cases and loops 1..depth; kappa = sigma* x r."""
    rows, per_case = generated(args.calib_seed, args.calib_cases, args.depth), {}
    for r in rows:
        watch.check(f"scale {r['id']}")
        per_case[r["id"]] = slot_rms(model, lm, tok, digits, r, loops=args.depth)
    vals = [v for rms in per_case.values() for v in rms]
    mean = sum(vals) / len(vals)
    log(f"scale: mean clean slot RMS {mean:.4f} over {len(vals)} loops; kappa {sigma * mean:.4f}")
    return {"slot_rms_mean": mean, "slot_rms_min": min(vals), "slot_rms_max": max(vals), "kappa": sigma * mean,
            "sigma_star": sigma, "cases": len(rows), "seed": args.calib_seed, "per_case": per_case}


def train(args, model, lm, tok, digits, extract, parts, watch, hidden, sigma):
    import torch
    out = {}
    for arm in ARMS[1:]:
        ck = args.out / f"G_{arm}.pt"
        G = corrector(arm, hidden).to(args.device)
        opt = torch.optim.Adam(G.parameters(), lr=args.lr)
        step, log_rows = 0, []
        state = args.out / f"G_{arm}.state.pt"
        if state.exists():
            s = torch.load(state, map_location="cpu", weights_only=False)
            G.load_state_dict(s["G"])
            opt.load_state_dict(s["opt"])
            step, log_rows = s["step"], s["log"]
        rows = generated(args.train_seed, args.steps * args.batch, args.max_train_loop + 1)
        while step < args.steps:
            watch.check(f"train {arm} step {step}")
            rng = random.Random(f"{args.train_seed}:{step}")  # the same cases, loops and noise for every arm
            opt.zero_grad()
            total = 0.0
            for i in range(args.batch):
                row = rows[step * args.batch + i]
                j = rng.randint(1, args.max_train_loop)
                noise = None if rng.random() < args.clean_share else {
                    "j": j, "sigma": sigma, "eps": eps_for(f"train:{row['id']}:{j}", hidden)}
                loss = train_step_loss(model, lm, tok, digits, row, j, arm=arm, G=G, noise=noise,
                                       scale=args.kappa) / args.batch
                loss.backward()
                total += float(loss)
            opt.step()
            step += 1
            log_rows.append({"step": step, "loss": total})
            if step % 25 == 0 or step == args.steps:
                torch.save({"G": G.state_dict(), "opt": opt.state_dict(), "step": step, "log": log_rows}, state)
                log(f"train {arm} step {step}/{args.steps}: loss {total:.4f}")
        torch.save(G.state_dict(), ck)
        out[arm] = {"steps": step, "final_loss_mean_last_25": sum(r["loss"] for r in log_rows[-25:]) / 25,
                    "parameters": sum(p.numel() for p in G.parameters()), "loss_log": log_rows}
    return out


def evaluate(args, model, lm, tok, digits, extract, parts, watch, hidden, sigma):
    import torch
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    rows = [c for c in roster["cases"] if c["split"] != "train"]
    Gs = {"P": None}
    for arm in ARMS[1:]:
        G = corrector(arm, hidden).to(args.device)
        G.load_state_dict(torch.load(args.out / f"G_{arm}.pt", map_location=args.device, weights_only=True))
        Gs[arm] = G.eval()
    conds = [("clean", None, None, 0)] + [("primary", j, sigma, s) for j in args.noise_loops for s in range(args.seeds)] + \
            [(tag, j, sigma * f, 0) for tag, f in (("half", 0.5), ("double", 2.0)) for j in args.noise_loops]
    records, t0 = [], time.perf_counter()
    for arm in ARMS:
        for r in rows:
            truth = str(r["trace"][args.depth]) if args.depth < len(r["trace"]) else str(iterate(r["successors"], r["initial"], args.depth))
            for tag, j, sg, seed in conds:
                rid = f"{arm}:{r['id']}:{tag}:j{j}:s{seed}"
                hit = parts.load("eval", rid)
                if hit is None:
                    watch.check(rid)
                    noise = None if j is None else {"j": j, "sigma": sg,
                                                    "eps": eps_for(f"{r['id']}:{tag}:j{j}:s{seed}", hidden)}
                    ans, text, trace = answer(model, lm, tok, digits, r, loops=args.depth, extract=extract,
                                              arm=arm, G=Gs[arm], noise=noise, scale=args.kappa)
                    rec = {"arm": arm, "case": r["id"], "family": r["family"], "split": r["split"], "condition": tag,
                           "j": j, "sigma": sg, "seed": seed, "answer": truth, "predicted": ans, "text": text,
                           "correct": ans == truth, "decoded_trace": trace, "trace_valid": trace_valid(r, trace)}
                    parts.save("eval", rid, rec)
                else:
                    rec = hit[0]
                records.append(rec)
        log(f"eval arm {arm} done ({time.perf_counter() - t0:.0f}s)")
    return {"records": records, "summary": summarize(records, args.noise_loops)}


def case_bootstrap(per_case, n_boot=2000, seed=0):
    import numpy as np
    x = np.asarray(per_case, dtype=float)
    boots = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(1)
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def summarize(records, noise_loops):
    from collections import defaultdict

    def per_case(arm, cond, key="correct"):
        by = defaultdict(list)
        for r in records:
            if r["arm"] == arm and r["condition"] == cond:
                by[r["case"]].append(r[key])
        return {c: sum(v) / len(v) for c, v in sorted(by.items())}

    out = {"accuracy": {}, "trace_valid": {}, "by_j": {}, "by_family": {}, "decisions": {}}
    for arm in ARMS:
        for cond in ("clean", "primary", "half", "double"):
            pc, tv = per_case(arm, cond), per_case(arm, cond, "trace_valid")
            out["accuracy"].setdefault(arm, {})[cond] = sum(pc.values()) / len(pc)
            out["trace_valid"].setdefault(arm, {})[cond] = sum(tv.values()) / len(tv)
        prim = [r for r in records if r["arm"] == arm and r["condition"] == "primary"]
        out["by_j"][arm] = {j: sum(r["correct"] for r in prim if r["j"] == j) / max(1, sum(r["j"] == j for r in prim))
                            for j in noise_loops}
        fams = sorted({r["family"] for r in prim})
        out["by_family"][arm] = {f: sum(r["correct"] for r in prim if r["family"] == f) / sum(r["family"] == f for r in prim)
                                 for f in fams}
    for name, a, b, margin in (("R1_C_minus_P", "C", "P", 0.10), ("R2_C_minus_I", "C", "I", 0.05),
                               ("R3_C_minus_N", "C", "N", 0.10)):
        pa, pb = per_case(a, "primary"), per_case(b, "primary")
        diff = [pa[c] - pb[c] for c in pa]
        ci = case_bootstrap(diff)
        out["decisions"][name] = {"diff": sum(diff) / len(diff), "ci95": ci, "margin": margin,
                                  "met": bool(sum(diff) / len(diff) >= margin and ci[0] > 0)}
    out["clean_damage"] = {arm: out["accuracy"][arm]["clean"] < 0.97 for arm in ARMS}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("calibrate", "scale", "train", "eval"))
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--noise-loops", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--calib-seed", type=int, default=20261108)
    ap.add_argument("--calib-cases", type=int, default=48)
    ap.add_argument("--train-seed", type=int, default=20261107)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-loop", type=int, default=7)
    ap.add_argument("--clean-share", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--min-power-w", type=float, default=50)
    ap.add_argument("--cuda-fraction", type=float, default=0.70)
    ap.add_argument("--out-scale", choices=("unit", "noise"), default="unit",
                    help="corrector output scale: unit (H-A3) or kappa = sigma* x clean slot RMS from scale.json (H-A3b)")
    args = ap.parse_args()
    meta, model, lm, tok, digits = load(args)
    hidden = model.base_causallm.config.hidden_size
    args.out.mkdir(parents=True, exist_ok=True)
    roster_sha = json.loads(args.roster.read_text(encoding="utf-8"))["roster_sha256"]
    sigma = None
    if args.cmd != "calibrate":
        sigma = json.loads((args.out / "calibrate.json").read_text(encoding="utf-8"))["result"]["sigma_star"]
    args.kappa = 1.0
    if args.out_scale == "noise" and args.cmd in ("train", "eval"):
        args.kappa = json.loads((args.out / "scale.json").read_text(encoding="utf-8"))["result"]["kappa"]
    key = {"checkpoint_sha256": meta["final_sha256"], "roster": roster_sha, "depth": args.depth, "sigma_star": sigma,
           "cmd": args.cmd, "G": [file_sha256(args.out / f"G_{a}.pt") for a in ARMS[1:]] if args.cmd == "eval" else None}
    if args.out_scale == "noise":
        key["kappa"] = args.kappa
    parts = Parts(args.out / "parts", key)
    watch = PowerWatch(args.min_power_w, active=str(args.device).startswith("cuda"))
    extract = upstream_extract_answer(args.upstream)
    fn = {"calibrate": calibrate, "scale": measure_scale, "train": train, "eval": evaluate}[args.cmd]
    try:
        res = fn(args, model, lm, tok, digits, extract, parts, watch, hidden,
                 *(() if args.cmd == "calibrate" else (sigma,)))
    except Paused as e:
        log(f"PAUSE: {e}; exit {PAUSE_EXIT}")
        sys.exit(PAUSE_EXIT)
    out = {"schema_version": "algernon-03-v1", "experiment": "H-A3b" if args.out_scale == "noise" else "H-A3",
           "cmd": args.cmd, "checkpoint_sha256": meta["final_sha256"], "roster_sha256": roster_sha, "sigma_star": sigma,
           "kappa": args.kappa,
           "args": {k: v for k, v in vars(args).items() if k not in ("upstream", "base", "checkpoint", "roster", "out")}
           | {"device": args.device}, "result": res}
    write_atomic(args.out / f"{args.cmd}.json", json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    if args.cmd == "calibrate":
        log(json.dumps(res))
    elif args.cmd == "scale":
        log(json.dumps({k: v for k, v in res.items() if k != "per_case"}))
    elif args.cmd == "eval":
        log(json.dumps({k: res["summary"][k] for k in ("accuracy", "decisions", "clean_damage")}))


if __name__ == "__main__":
    main()
