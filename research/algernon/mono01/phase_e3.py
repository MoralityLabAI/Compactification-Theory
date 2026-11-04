"""MONO-01 Phase E3: loop-direction localization and analytic down_proj edits on a Qwen3.5 text model (0.8B bf16 / 9B NF4).

Stages (each skipped when its output exists, so a power pause or restart resumes):
  gen_calib  base-model generations on the calibration roster (thinking on, sampled)
  collect    one batched teacher-forced pass: per-layer down_proj input reservoirs, loop/non-loop output means
  proxy      per-layer teacher-forced ablation of the loop direction -> loop-suppression minus damage score
  tf_arms    held-out teacher-forced damage / suppression for every edit arm, with predicted preservation cost
  gen_arms   generations on the frozen test roster for every arm (the only task outcome)
Edits are applied as factorized fp32 forward hooks on mlp.down_proj (works for NF4 weights; no merge rounding).
Usage: python phase_e3.py --model PATH --quant bf16|nf4 --out DIR [--layers 0,1,...] [--smoke]
Exit code 75 = paused because the GPU power cap dropped (rerun to resume).
"""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monoshot as M

PRACTICE = "C:/Users/patri/Documents/Codex/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl"
ROSTER_SEED = 20261104
SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
NGRAM, MIN_SEEN = 8, 2
C_SCALE = 6.0
SUPPORT = 64


# ---------------------------------------------------------------- utilities
def power_ok(min_w=100.0) -> bool:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        return float(o) >= min_w
    except Exception:
        return True


def maybe_pause():
    if not power_ok():
        print("power cap below 100 W: pausing (exit 75)", flush=True)
        sys.exit(75)


def roster(n_cal, n_test):
    rows = [json.loads(l) for l in open(PRACTICE, encoding="utf-8")]
    perm = np.random.default_rng(ROSTER_SEED).permutation(len(rows))
    cal, tf, test = perm[:n_cal], perm[n_cal:n_cal + max(n_cal // 2, 4)], perm[n_cal + n_cal:n_cal + n_cal + n_test]
    sel = lambda ix: [rows[i] for i in ix]
    ids = lambda ix: [rows[i]["id"] for i in ix]
    h = hashlib.sha256(json.dumps([ids(cal), ids(tf), ids(test)]).encode()).hexdigest()
    assert not (set(ids(cal)) & set(ids(test))) and not (set(ids(tf)) & set(ids(test)))
    return sel(cal), sel(tf), sel(test), h


def prompt_ids(tok, problem):
    msgs = [{"role": "user", "content": problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return tok(s, add_special_tokens=False).input_ids


def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(text) and d:
        d += (text[j] == "{") - (text[j] == "}"); j += 1
    return text[i + 7:j - 1] if d == 0 else None


def norm_num(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\\$|\$|,|\\%|%|\\!|\s", "", s)
    try:
        return float(s)
    except ValueError:
        m = re.findall(r"-?\d+\.?\d*", s)
        return float(m[-1]) if m else None


def loop_mask(ids, n=NGRAM, min_seen=MIN_SEEN):
    """Mark tokens inside n-grams that have already appeared at least min_seen times earlier in the same generation."""
    seen, mask = {}, np.zeros(len(ids), bool)
    for t in range(n - 1, len(ids)):
        g = tuple(ids[t - n + 1:t + 1])
        c = seen.get(g, 0)
        if c >= min_seen:
            mask[t - n + 1:t + 1] = True
        seen[g] = c + 1
    return mask


def layers_of(model):
    base = model.model
    return getattr(base, "language_model", base).layers


def dense_w(lin):
    w = lin.weight
    if hasattr(w, "quant_state"):
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(w.data, w.quant_state).float()
    return w.float()


def load(path, quant):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0})
    m.eval()
    return tok, m


# ---------------------------------------------------------------- generation
@torch.inference_mode()
def generate(model, tok, items, out_file, max_new, bs, seed, keep_ids=False):
    done = {}
    if out_file.exists():
        for l in open(out_file, encoding="utf-8"):
            r = json.loads(l); done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for b in range(0, len(todo), bs):
        maybe_pause()
        chunk = todo[b:b + bs]
        enc = [prompt_ids(tok, it["problem"]) for it in chunk]
        L = max(map(len, enc))
        pad = tok.pad_token_id
        x = torch.tensor([[pad] * (L - len(e)) + e for e in enc], device=0)
        am = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device=0)
        torch.manual_seed(seed + b)
        t0 = time.time()

        def run(ix, depth=0):
            """Generate for rows ix of this chunk; on CUDA OOM split the rows in half and retry (rows are independent)."""
            try:
                return model.generate(input_ids=x[ix], attention_mask=am[ix], max_new_tokens=max_new, pad_token_id=pad, **SAMPLING)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() or len(ix) == 1:
                    raise
                print(f"  OOM at batch {len(ix)}: splitting", flush=True)
                torch.cuda.empty_cache()
                h = len(ix) // 2
                a_, b_ = run(ix[:h], depth + 1), run(ix[h:], depth + 1)
                W = max(a_.shape[1], b_.shape[1])
                padf = lambda t: torch.nn.functional.pad(t, (0, W - t.shape[1]), value=pad)
                return torch.cat([padf(a_), padf(b_)])
        out = run(torch.arange(len(chunk), device=0))
        with open(out_file, "a", encoding="utf-8") as f:
            for it, e, row in zip(chunk, enc, out[:, L:].tolist()):
                fin = eos in row
                g = row[:row.index(eos)] if fin else [t for t in row if t != pad]
                text = tok.decode(g, skip_special_tokens=True)
                ans = norm_num(boxed(text)); gold = norm_num(it["answer"])
                r = {"id": it["id"], "n_tokens": len(g), "finished": bool(fin), "correct": bool(ans is not None and gold is not None and abs(ans - gold) < 1e-6),
                     "has_boxed": ans is not None, "loop_frac": float(loop_mask(g).mean()) if g else 0.0}
                if keep_ids:
                    r["prompt"], r["gen"] = e, g
                f.write(json.dumps(r) + "\n")
        print(f"  gen {out_file.name} {b + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    return [json.loads(l) for l in open(out_file, encoding="utf-8")]


# ---------------------------------------------------------------- teacher-forced passes
def batches(seqs, bs):
    for i in range(0, len(seqs), bs):
        yield seqs[i:i + bs]


def make_batch(chunk, pad):
    L = max(len(p) + len(g) for p, g, _ in chunk)
    ids = torch.full((len(chunk), L), pad, dtype=torch.long)
    gm = torch.zeros(len(chunk), L, dtype=torch.bool)
    lm = torch.zeros(len(chunk), L, dtype=torch.bool)
    am = torch.zeros(len(chunk), L, dtype=torch.long)
    for i, (p, g, lmask) in enumerate(chunk):
        n = len(p) + len(g)
        ids[i, :n] = torch.tensor(p + g); am[i, :n] = 1
        gm[i, len(p):n] = True
        lm[i, len(p):n] = torch.from_numpy(lmask)
    return ids.to(0), am.to(0), gm.to(0), lm.to(0)


@torch.inference_mode()
def token_logprobs(model, chunk, pad, hooks=()):
    """Log-prob of each generated token (positions predicted from t-1), for loop / non-loop masks; hooks = (layer, fn)."""
    ids, am, gm, lm = make_batch(chunk, pad)
    hs = [layers_of(model)[l].mlp.down_proj.register_forward_hook(fn) for l, fn in hooks]
    try:
        h = model.model(input_ids=ids, attention_mask=am).last_hidden_state
    finally:
        for hd in hs:
            hd.remove()
    sel = gm[:, 1:] | False
    pos_b, pos_t = torch.nonzero(gm, as_tuple=True)
    keep = pos_t > 0
    pos_b, pos_t = pos_b[keep], pos_t[keep]
    lp = []
    for i in range(0, len(pos_b), 256):
        hb, tb = pos_b[i:i + 256], pos_t[i:i + 256]
        logits = model.lm_head(h[hb, tb - 1]).float()
        lp.append(torch.log_softmax(logits, -1).gather(1, ids[hb, tb][:, None])[:, 0])
    lp = torch.cat(lp)
    isl = lm[pos_b, pos_t]
    return lp, isl


def tf_stats(model, seqs, pad, hooks=(), bs=4):
    lps, isl = [], []
    for chunk in batches(seqs, bs):
        a, b = token_logprobs(model, chunk, pad, hooks)
        lps.append(a.cpu()); isl.append(b.cpu())
    return torch.cat(lps), torch.cat(isl)


@torch.inference_mode()
def collect(model, seqs, pad, layers, n_ref=2048, n_key=1024, bs=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    acc = {l: {"y_loop": 0.0, "n_loop": 0, "y_non": 0.0, "n_non": 0, "a_loop": 0.0, "ref": [], "key": []} for l in layers}
    state = {}

    def mk(l):
        def hook(mod, inp, out):
            a, gm, lm = inp[0], state["gm"], state["lm"]
            A = acc[l]
            for name, m in (("loop", gm & lm), ("non", gm & ~lm)):
                if not m.any():
                    continue
                y = out[m].float()
                A["y_" + name] = A["y_" + name] + y.sum(0); A["n_" + name] += y.shape[0]
                at = inp[0][m]
                if name == "loop":
                    A["a_loop"] = A["a_loop"] + at.float().sum(0)
                cap, store = (n_key, "key") if name == "loop" else (n_ref, "ref")
                have = sum(x.shape[0] for x in A[store])
                if have < cap:
                    k = min(at.shape[0], max(cap // 8, 1), cap - have)
                    ix = torch.randperm(at.shape[0], generator=g)[:k].to(at.device)
                    A[store].append(at[ix].half().cpu())
        return hook
    hs = [layers_of(model)[l].mlp.down_proj.register_forward_hook(mk(l)) for l in layers]
    try:
        for chunk in batches(seqs, bs):
            maybe_pause()
            ids, am, gm, lm = make_batch(chunk, pad)
            state["gm"], state["lm"] = gm, lm
            model.model(input_ids=ids, attention_mask=am)
    finally:
        for h in hs:
            h.remove()
    out = {}
    for l, A in acc.items():
        out[l] = {"u_raw": (A["y_loop"] / max(A["n_loop"], 1) - A["y_non"] / max(A["n_non"], 1)).cpu(),
                  "k_mean": (A["a_loop"] / max(A["n_loop"], 1)).cpu(), "n_loop": A["n_loop"], "n_non": A["n_non"],
                  "ref": torch.cat(A["ref"]) if A["ref"] else None, "key": torch.cat(A["key"]) if A["key"] else None}
    return out


# ---------------------------------------------------------------- edits
def proj_hook(u, c):
    def hook(mod, inp, out):
        y = out.float()
        return (y - c * (y @ u)[..., None] * u).to(out.dtype)
    return hook


def edit_hook(S, w, v):
    def hook(mod, inp, out):
        a = inp[0]
        s = (a[..., S].float() @ w) if S is not None else (a.float() @ w)
        return out + (s[..., None] * v).to(out.dtype)
    return hook


def cost_of(Xref, S, w, v, lam):
    X = Xref.float() if S is None else Xref[:, S].float()
    w = w.float().cpu()
    return float(v.float().norm().item() ** 2 * ((X @ w).pow(2).mean() + lam * w.pow(2).sum()).item())


def woodbury_solve(X, lam, b):
    """C^-1 b, C = lam I + X^T X / N (X on GPU float32)."""
    N = X.shape[0]
    X, b = X.double(), b.double()
    inner = torch.linalg.solve(N * lam * torch.eye(N, device=X.device, dtype=X.dtype) + X @ X.T, X @ b)
    return ((b - X.T @ inner) / lam).float()


def build_arms(model, info, layers_sel, c=C_SCALE, seed=0):
    """Return {arm: {layer: (S, w, v, cost)}} (tensors on GPU)."""
    rng = np.random.default_rng(seed)
    arms = {}
    for l in layers_sel:
        d = info[l]
        u = d["u"].cuda()
        W = dense_w(layers_of(model)[l].mlp.down_proj)               # (H, I) float32
        Xref = d["ref"].cuda().float()
        lam = float(1e-3 * Xref.pow(2).mean().item())
        uW = u @ W                                                    # (I,)
        kbar = d["k_mean"].cuda().float()
        t = float(u @ (W @ kbar))                                     # current u-component at the loop key
        v_pub = (-c * u)
        top = torch.topk(uW.abs(), SUPPORT).indices
        arms.setdefault("A_pub_c6", {})[l] = (top, uW[top], v_pub, None)
        ur = torch.tensor(rng.normal(size=u.shape[0]), dtype=torch.float32, device="cuda"); ur = ur / ur.norm()
        uWr = ur @ W; topr = torch.topk(uWr.abs(), SUPPORT).indices
        arms.setdefault("R_dir_c6", {})[l] = (topr, uWr[topr], -c * ur, None)
        pool = torch.topk(kbar.abs(), 256).indices.cpu().numpy()
        Xp = Xref[:, torch.tensor(pool, device="cuda")].double().cpu().numpy()
        Cp = M.second_moment(Xp, lam)
        kp = kbar[torch.tensor(pool, device="cuda")].double().cpu().numpy()
        for name, S_loc in (("B_greedy", M.select_greedy(kp, Cp, SUPPORT)[0]), ("B_beam", M.select_beam(kp, Cp, SUPPORT, width=4)[0]),
                            ("B_rand", sorted(rng.choice(256, SUPPORT, replace=False).tolist()))):
            Sg = np.array(pool)[S_loc]
            css = Cp[np.ix_(S_loc, S_loc)]; ks = kp[S_loc]
            sol = np.linalg.solve(css, ks); w = torch.tensor(sol / (ks @ sol), dtype=torch.float32, device="cuda")
            for cc, tag in ((c, "c6"), (1.0, "c1")):
                if name != "B_greedy" and tag == "c1":
                    continue
                arms.setdefault(f"{name}_{tag}", {})[l] = (torch.tensor(Sg, device="cuda"), w, -cc * t * u, None)
        w = woodbury_solve(Xref, lam, kbar); w = w / (kbar @ w)
        arms.setdefault("B_full_c6", {})[l] = (None, w, -c * t * u, None)
        for arm in arms:
            if l in arms[arm]:
                S, w, v, _ = arms[arm][l]
                arms[arm][l] = (S, w, v, cost_of(d["ref"], None if S is None else S.cpu(), w, v, lam))
        del W
    return arms


def install(model, spec):
    return [layers_of(model)[l].mlp.down_proj.register_forward_hook(edit_hook(S, w, v)) for l, (S, w, v, _) in spec.items()]


def published_tensors(model_dir, exp8_dir):
    """Rebuild the published p50c edited down_proj tensors (hash-checked) as bf16 tensors keyed by layer."""
    import phase_e2_qwen08b as E
    from safetensors import safe_open
    root = Path(exp8_dir) / "runs" / "p50c" / "edit"
    man = json.loads((root / "manifest.json").read_text())
    z = np.load(root / "edit.npz", allow_pickle=False)
    res = {}
    with safe_open(str(Path(model_dir) / "model.safetensors-00001-of-00001.safetensors"), framework="pt") as f:
        for i, t in enumerate(man["tensors"]):
            e = E.rebuild(E.bits(f.get_tensor(t["name"])), t, z, i)
            assert E.sha(e) == t["edited_sha256"]
            res[int(re.search(r"layers\.(\d+)\.", t["name"]).group(1))] = torch.from_numpy(e.view(np.int16).copy()).view(torch.bfloat16)
    return res


class swap_weights:
    def __init__(self, model, new):
        self.model, self.new, self.old = model, new, {}
    def __enter__(self):
        for l, w in self.new.items():
            p = layers_of(self.model)[l].mlp.down_proj.weight
            self.old[l] = p.data.clone(); p.data.copy_(w.to(p.device))
    def __exit__(self, *e):
        for l, w in self.old.items():
            layers_of(self.model)[l].mlp.down_proj.weight.data.copy_(w)


def paired_boot(a, b, n=4000, seed=1):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--quant", default="bf16"); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=""); ap.add_argument("--n-cal", type=int, default=64); ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=1024); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n-ref", type=int, default=2048); ap.add_argument("--published-edit", default=""); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_cal, a.n_test, a.max_new, a.bs = 8, 8, 96, 8
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cal, tfs, test, rh = roster(a.n_cal, a.n_test)
    T0 = time.time()
    meta = {"model": a.model, "quant": a.quant, "roster_sha256": rh, "n_cal": len(cal), "n_tf": len(tfs), "n_test": len(test), "max_new": a.max_new,
            "sampling": SAMPLING, "ngram": NGRAM, "min_seen": MIN_SEEN, "support": SUPPORT, "c": C_SCALE, "roster_seed": ROSTER_SEED}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    tok, model = load(a.model, a.quant)
    pad = tok.pad_token_id
    nl = len(layers_of(model))
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(nl))
    print("layers", len(layers), "of", nl, "roster", rh[:12], flush=True)

    # stage gen_calib (calibration + held-out teacher-forced texts, base model)
    base_cal = generate(model, tok, cal + tfs, out / "gen_calib.jsonl", a.max_new, a.bs, 11, keep_ids=True)
    byid = {r["id"]: r for r in base_cal}
    mk_seq = lambda items: [(byid[i["id"]]["prompt"], byid[i["id"]]["gen"], loop_mask(byid[i["id"]]["gen"])) for i in items if byid[i["id"]]["gen"]]
    cal_seq, tf_seq = mk_seq(cal), mk_seq(tfs)
    meta["calib_cut_rate"] = float(np.mean([not byid[i["id"]]["finished"] for i in cal]))
    meta["calib_loop_token_share"] = float(np.mean([m.mean() for _, _, m in cal_seq]))

    # stage collect + proxy
    if not (out / "info.pt").exists():
        info = collect(model, cal_seq, pad, layers, n_ref=a.n_ref, n_key=max(a.n_ref // 2, 256))
        for l, d in info.items():
            d["u"] = d["u_raw"] / d["u_raw"].norm().clamp_min(1e-9)
        base_lp, isl = tf_stats(model, cal_seq, pad)
        proxy = {}
        for l in layers:
            maybe_pause()
            lp, _ = tf_stats(model, cal_seq, pad, hooks=[(l, proj_hook(info[l]["u"].cuda(), 1.0))])
            sup = float((base_lp[isl] - lp[isl]).mean()); dmg = float((base_lp[~isl] - lp[~isl]).mean())
            proxy[l] = {"suppression_loop_tokens": sup, "damage_nonloop_tokens": dmg, "score": sup - dmg, "u_norm": float(info[l]["u_raw"].norm()),
                        "n_loop_tokens": info[l]["n_loop"]}
            print(f"  proxy layer {l}: suppress {sup:+.4f} damage {dmg:+.4f}", flush=True)
        torch.save(info, out / "info.pt"); (out / "proxy.json").write_text(json.dumps(proxy, indent=1))
    info = torch.load(out / "info.pt"); proxy = json.loads((out / "proxy.json").read_text())
    ranked = sorted(proxy, key=lambda l: -proxy[l]["score"])
    layers_sel = [int(l) for l in ranked[:2]]
    meta["layers_selected"] = layers_sel; meta["proxy_rank_top5"] = [int(l) for l in ranked[:5]]

    # stage tf_arms
    arms = build_arms(model, {l: info[l] for l in layers_sel}, layers_sel)
    if a.published_edit:
        meta["published_edit_dir"] = a.published_edit
    base_lp, isl = tf_stats(model, tf_seq, pad)
    tf = {}
    for name, spec in arms.items():
        hs = install(model, spec)
        try:
            lp, _ = tf_stats(model, tf_seq, pad)
        finally:
            [h.remove() for h in hs]
        tf[name] = {"predicted_cost": float(sum(c for (_, _, _, c) in spec.values())),
                    "nonloop_logprob_drop": float((base_lp[~isl] - lp[~isl]).mean()), "loop_logprob_drop": float((base_lp[isl] - lp[isl]).mean()),
                    "nonloop_abs_change": float((lp[~isl] - base_lp[~isl]).abs().mean())}
        print(f"  tf {name}: cost {tf[name]['predicted_cost']:.3g} nonloop drop {tf[name]['nonloop_logprob_drop']:+.4f} loop drop {tf[name]['loop_logprob_drop']:+.4f}", flush=True)
    pub = published_tensors(a.model, a.published_edit) if a.published_edit else None
    if pub:
        with swap_weights(model, pub):
            lp, _ = tf_stats(model, tf_seq, pad)
        tf["P_published_p50c"] = {"predicted_cost": None, "nonloop_logprob_drop": float((base_lp[~isl] - lp[~isl]).mean()),
                                  "loop_logprob_drop": float((base_lp[isl] - lp[isl]).mean()), "nonloop_abs_change": float((lp[~isl] - base_lp[~isl]).abs().mean())}
        print("  tf published p50c:", tf["P_published_p50c"], flush=True)
    (out / "tf_arms.json").write_text(json.dumps(tf, indent=1))

    # stage gen_arms (frozen test roster; no choices depend on these outcomes)
    results = {"base": generate(model, tok, test, out / "gen_base.jsonl", a.max_new, a.bs, 21)}
    for name, spec in arms.items():
        hs = install(model, spec)
        try:
            results[name] = generate(model, tok, test, out / f"gen_{name}.jsonl", a.max_new, a.bs, 21)
        finally:
            [h.remove() for h in hs]
    if pub:
        with swap_weights(model, pub):
            results["P_published_p50c"] = generate(model, tok, test, out / "gen_P_published_p50c.jsonl", a.max_new, a.bs, 21)
    summ = {}
    bcut = [not r["finished"] for r in results["base"]]; bacc = [r["correct"] for r in results["base"]]
    for name, rs in results.items():
        assert [r["id"] for r in rs] == [r["id"] for r in results["base"]]
        cut = [not r["finished"] for r in rs]; acc = [r["correct"] for r in rs]
        summ[name] = {"cut_rate": float(np.mean(cut)), "accuracy": float(np.mean(acc)), "mean_tokens": float(np.mean([r["n_tokens"] for r in rs])),
                      "mean_loop_frac": float(np.mean([r["loop_frac"] for r in rs])),
                      "cut_delta_vs_base": paired_boot(bcut, cut) if name != "base" else None,
                      "acc_delta_vs_base": paired_boot(bacc, acc) if name != "base" else None, **(tf.get(name, {}))}
    meta["wall_seconds"] = time.time() - T0
    (out / "results.json").write_text(json.dumps({"meta": meta, "arms": summ, "proxy": proxy}, indent=1))
    for n, s in summ.items():
        print(f"{n:14s} cut {s['cut_rate']:.3f} acc {s['accuracy']:.3f} tokens {s['mean_tokens']:.0f} loopfrac {s['mean_loop_frac']:.3f}"
              + (f" dcut {s['cut_delta_vs_base'][0]:+.3f} {s['cut_delta_vs_base'][1]}" if s["cut_delta_vs_base"] else ""), flush=True)


if __name__ == "__main__":
    main()
