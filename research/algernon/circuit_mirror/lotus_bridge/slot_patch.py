"""H-A2e on the 02D arm D1: depth beyond 16, and causal patching of the fixed latent slots.

`depth`: the no-count question, one block of 6 slots and d loops, for every d in --depths, on the frozen roster's
calibration and held-out cases. The truth is the map iterated d times from the start, so d may exceed the
roster's 16 recorded steps. The slot states' largest absolute value after the last loop is recorded per prompt.

`patch`: the six slots are the only recurrent state; upstream re-injects their hidden states at every loop. For a
recipient case A and a donor case B, the hidden states of A's slots after loop k are replaced by B's after loop k,
and A runs on to d loops. If the slots carry the pointer state and each loop applies the map in A's prompt, the
answer is A's map iterated d - k times from B's state after k steps ("follow"). Each (A, k) gets the first donor,
in a seeded order, for which "follow", A's own answer and B's own answer are three different states. A
self-patch (B = A) must reproduce A's unpatched answer.

Answers are saved per prompt, so a power-cap pause (exit 75) resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .gpu_run import PAUSE_EXIT, Parts, Paused, PowerWatch, log, upstream_extract_answer, write_atomic
from .pointer_loop import prompt_nocount
from .probe import wilson

SLOTS = 6
RECIPIENT_FAMILIES = ("permutation", "long_cycle")


def iterate(successors, state, n):
    for _ in range(n):
        state = successors[state]
    return state


@contextmanager
def slot_hook(lotus_module, first, last, *, record=None, patch=None):
    """Wrap upstream efficient_forward for one generate call. Call 0 is the initial pass over the loop region and
    call j the pass of loop j. record: list that receives the slot hidden states [SLOTS, H] of every call.
    patch: (k, tensor) replaces the slot hidden states returned by call k, before the next loop injects them."""
    original, calls = lotus_module.efficient_forward, [0]

    def hooked(*args, **kwargs):
        out = original(*args, **kwargs)
        h = out.last_hidden_state
        offset = first - 1 if h.shape[1] == last + 2 - first else 0  # loop region [first-1, last], or prefix + loop
        if patch is not None and calls[0] == patch[0]:
            h[0, first - offset:last - offset + 1] = patch[1].to(h.dtype)
        if record is not None:
            record.append(h[0, first - offset:last - offset + 1].detach().float().cpu().clone())
        calls[0] += 1
        return out

    lotus_module.efficient_forward = hooked
    try:
        yield calls
    finally:
        lotus_module.efficient_forward = original


def run_case(model, lotus_module, tok, question, *, loops, extract, record=False, patch=None, max_new_tokens=16):
    """Greedy answer with one block of SLOTS slots and `loops` loops, laid out as train_lotus.answers lays it out."""
    import torch
    q = tok.encode(question + "\n", add_special_tokens=True)
    first, last = len(q) + 1, len(q) + SLOTS
    device = next(model.parameters()).device
    x = torch.tensor([q + [model.start_latent_id] + [model.latent_token_id] * SLOTS + [model.end_latent_id]],
                     device=device)
    states = [] if record else None
    with torch.inference_mode(), slot_hook(lotus_module, first, last, record=states, patch=patch) as calls:
        gen = model.generate(x, torch.ones_like(x), max_new_tokens=max_new_tokens, n_looped_iters=loops)
    if calls[0] != loops + 1:
        raise AssertionError(f"expected initial + {loops} loop passes, got {calls[0]}")
    text = tok.decode(gen[0, x.shape[1]:].tolist(), skip_special_tokens=True)
    return {"text": text, "answer": extract(text)}, states


def pool(roster):
    return [c for c in roster["cases"] if c["split"] != "train"]


def depth(args, model, lm, tok, extract, parts, watch):
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    rows = [(c, d) for c in pool(roster) for d in args.depths]
    records, t0 = [], time.perf_counter()
    for i, (c, d) in enumerate(rows):
        rid = f"{c['id']}@{d}"
        hit = parts.load("depth", rid)
        if hit is None:
            watch.check(rid)
            truth = str(iterate(c["successors"], c["initial"], d))
            res, states = run_case(model, lm, tok, prompt_nocount(c).rstrip("\n"), loops=d, extract=extract, record=True)
            rec = {"id": rid, "case": c["id"], "family": c["family"], "split": c["split"], "depth": d, "answer": truth,
                   "predicted": res["answer"], "text": res["text"], "correct": res["answer"] == truth,
                   "slot_max_abs_last_loop": float(states[-1].abs().max())}
            parts.save("depth", rid, rec)
        else:
            rec = hit[0]
        records.append(rec)
        if (i + 1) % 64 == 0:
            log(f"depth {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s)")
    table = {}
    for r in records:
        for key in (r["family"], "heldout" if r["split"] == "test" else None):
            if key:
                table.setdefault(key, {}).setdefault(r["depth"], []).append(r["correct"])
    summary = {"accuracy_by_depth": {f: {d: sum(v) / len(v) for d, v in sorted(t.items())} for f, t in table.items()},
               "slot_max_abs_by_depth": {d: max(r["slot_max_abs_last_loop"] for r in records if r["depth"] == d)
                                         for d in args.depths}}
    lc = summary["accuracy_by_depth"].get("long_cycle", {})
    summary["holds_to_depth"] = max([d for d in args.depths if lc.get(d, 0) >= 0.95 and
                                     all(lc.get(e, 0) >= 0.95 for e in args.depths if e <= d)], default=None)
    return {"roster_sha256": roster["roster_sha256"], "depths": list(args.depths), "summary": summary,
            "records": records}


def choose_donor(a, k, d, donors, rng):
    own = iterate(a["successors"], a["initial"], d)
    order = donors[:]
    rng.shuffle(order)
    for b in order:
        if b["id"] == a["id"]:
            continue
        follow = iterate(a["successors"], iterate(b["successors"], b["initial"], k), d - k)
        donor_own = iterate(b["successors"], b["initial"], d)
        if len({follow, own, donor_own}) == 3:
            return b, {"follow": follow, "own": own, "donor_own": donor_own}
    return None, None


def patch(args, model, lm, tok, extract, parts, watch):
    import torch
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    cases = pool(roster)
    recipients = [c for c in cases if c["family"] in RECIPIENT_FAMILIES]
    d, states_of, records, t0 = args.patch_depth, {}, [], time.perf_counter()

    def states(c):
        """Unpatched answer and slot states of every pass for case c at d loops (saved as a part)."""
        if c["id"] not in states_of:
            hit = parts.load("donor", c["id"])
            if hit is None:
                watch.check(c["id"])
                res, st = run_case(model, lm, tok, prompt_nocount(c).rstrip("\n"), loops=d, extract=extract, record=True)
                arr = torch.stack(st).numpy()
                parts.save("donor", c["id"], res, {"slots": arr})
                hit = (res, {"slots": arr})
            states_of[c["id"]] = (hit[0], torch.from_numpy(hit[1]["slots"]))
        return states_of[c["id"]]

    for a in recipients:
        a_res, a_states = states(a)
        for k in args.patch_loops:
            rng = random.Random(f"{args.seed}:{a['id']}:{k}")
            b, pred = choose_donor(a, k, d, cases, rng)
            if b is None:
                records.append({"recipient": a["id"], "k": k, "d": d, "donor": None})
                continue
            for kind, src in (("patch", b), ("self", a)):
                rid = f"{kind}:{a['id']}:{k}"
                hit = parts.load("patch", rid)
                if hit is None:
                    watch.check(rid)
                    donor_slots = states(src)[1][k]
                    res, _ = run_case(model, lm, tok, prompt_nocount(a).rstrip("\n"), loops=d, extract=extract,
                                      patch=(k, donor_slots.to(next(model.parameters()).device)))
                    rec = {"kind": kind, "recipient": a["id"], "recipient_family": a["family"], "donor": src["id"],
                           "k": k, "d": d, "answer_unpatched": a_res["answer"], "predicted": res["answer"],
                           "text": res["text"], **{f"pred_{n}": str(v) for n, v in pred.items()}}
                    parts.save("patch", rid, rec)
                else:
                    rec = hit[0]
                records.append(rec)
        log(f"patch {a['id']} done ({time.perf_counter() - t0:.0f}s)")
    pat = [r for r in records if r.get("kind") == "patch"]
    slf = [r for r in records if r.get("kind") == "self"]
    rate = lambda rs, key: {"n": len(rs), "rate": sum(r["predicted"] == r[key] for r in rs) / max(len(rs), 1),
                            "wilson_95": wilson(sum(r["predicted"] == r[key] for r in rs) / max(len(rs), 1), len(rs))}
    summary = {"patch_follow": rate(pat, "pred_follow"), "patch_own": rate(pat, "pred_own"),
               "patch_donor_own": rate(pat, "pred_donor_own"),
               "self_reproduces_unpatched": rate(slf, "answer_unpatched"),
               "by_k": {k: {"follow": rate([r for r in pat if r["k"] == k], "pred_follow")["rate"],
                            "own": rate([r for r in pat if r["k"] == k], "pred_own")["rate"]} for k in args.patch_loops},
               "pairs_without_donor": sum(1 for r in records if r.get("donor") is None)}
    f, o, s = summary["patch_follow"], summary["patch_own"], summary["self_reproduces_unpatched"]
    summary["instrument_ok"] = s["rate"] == 1.0
    summary["slots_carry_state"] = bool(s["rate"] == 1.0 and f["rate"] >= 0.80 and f["wilson_95"][0] > 0.5
                                        and o["rate"] <= 0.10)
    return {"roster_sha256": roster["roster_sha256"], "d": d, "k": list(args.patch_loops), "seed": args.seed,
            "summary": summary, "records": records}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("depth", "patch"))
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--depths", type=int, nargs="+", default=[16, 20, 24, 32, 48, 64])
    ap.add_argument("--patch-depth", type=int, default=16)
    ap.add_argument("--patch-loops", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--min-power-w", type=float, default=50)
    ap.add_argument("--cuda-fraction", type=float, default=0.70)
    args = ap.parse_args()
    import torch
    from .train_lotus import load_trained
    meta = json.loads(args.checkpoint.with_name("train_manifest.json").read_text(encoding="utf-8"))
    if meta.get("fixed_slots") != SLOTS:
        raise SystemExit(f"{args.checkpoint} is not a {SLOTS}-slot loop-phase checkpoint")
    roster_sha = json.loads(args.roster.read_text(encoding="utf-8"))["roster_sha256"]
    model, lm, tok = load_trained(args.upstream, args.base, args.checkpoint, device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.cuda_fraction, 0)
    args.out.mkdir(parents=True, exist_ok=True)
    key = {"checkpoint_sha256": meta["final_sha256"], "roster": roster_sha, "slots": SLOTS, "cmd": args.cmd,
           "patch_depth": args.patch_depth, "seed": args.seed}
    parts = Parts(args.out / "parts", key)
    watch = PowerWatch(args.min_power_w, active=str(args.device).startswith("cuda"))
    extract = upstream_extract_answer(args.upstream)
    try:
        res = (depth if args.cmd == "depth" else patch)(args, model, lm, tok, extract, parts, watch)
    except Paused as e:
        log(f"PAUSE: {e}; exit {PAUSE_EXIT}")
        sys.exit(PAUSE_EXIT)
    out = {"schema_version": "algernon-02e-v1", "experiment": "H-A2e", "cmd": args.cmd,
           "checkpoint_sha256": meta["final_sha256"], **res}
    write_atomic(args.out / f"{args.cmd}.json", json.dumps(out, indent=1, sort_keys=True) + "\n")
    log(json.dumps(out["summary"] if args.cmd == "patch" else
                   {"holds_to_depth": out["summary"]["holds_to_depth"],
                    "heldout": out["summary"]["accuracy_by_depth"].get("heldout")}))


if __name__ == "__main__":
    main()
