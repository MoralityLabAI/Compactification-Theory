"""MONO-01 Phase D: robust (minimax) edit across nominal/crosswind covariances + bf16 merge vs factorized precision.

Run in the pinned policy-surgery environment: uv run --frozen --extra weights python phase_d.py OUT_DIR PHASE_C_JSON
Only development (50000-50063) and selection (60000-60063) seeds are used.
"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monoshot as M
import phase_c as P
from policy_surgery import campaign
from policy_surgery.policy import fetch_actor


def quant_report(W: np.ndarray, D: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    """W bf16 base weights, D float64 analytic delta. Compare factorized high precision with merging into bf16."""
    Wb = M.to_bf16(W.astype(np.float32)).astype(np.float64)
    merged = M.to_bf16((Wb + D).astype(np.float32)).astype(np.float64)
    realized_merged = (merged - Wb) @ K
    # factorized: rank-r factors kept in float32 (a separate LoRA path), applied on top of the bf16 base
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    r = int(np.sum(s > 1e-9 * max(s[0], 1e-30)))
    A, B = (U[:, :r] * s[:r]).astype(np.float32), Vt[:r].astype(np.float32)
    realized_fact32 = (A @ (B @ K.astype(np.float32))).astype(np.float64)
    A16, B16 = M.to_bf16(A).astype(np.float32), M.to_bf16(B).astype(np.float32)
    realized_fact16 = (A16 @ (B16 @ K.astype(np.float32))).astype(np.float64)
    nv = np.linalg.norm(V)
    return {"rank": r, "entries_targeted": int(np.count_nonzero(D)), "entries_changed_in_bf16_merge": int(np.count_nonzero(merged != Wb)),
            "factor_params_stored": int(A.size + B.size),
            "rel_corr_error_merged_bf16": float(np.linalg.norm(realized_merged - V) / nv),
            "rel_corr_error_factorized_fp32": float(np.linalg.norm(realized_fact32 - V) / nv),
            "rel_corr_error_factorized_bf16": float(np.linalg.norm(realized_fact16 - V) / nv),
            "max_abs_error_merged_bf16": float(np.max(np.abs(realized_merged - V)))}


def synthetic_qwen_shaped(seed=0, d=1024, n=3584, N=64, s=64):
    """Analog of the workorder's synthetic smoke (d x n layer, N reference tokens, s selected columns). Not the same RNG draws."""
    rng = np.random.default_rng(seed)
    W = (rng.normal(size=(d, n)) * 0.02).astype(np.float32)
    mix = rng.normal(size=(n, 24)) / np.sqrt(24)
    X = rng.normal(size=(N, 24)) @ mix.T + 0.3 * rng.normal(size=(N, n))
    lam = 1e-2
    k = rng.normal(size=n) @ np.diag(np.ones(n)) * 1.0
    k = k + 3.0 * X[0]                                       # fault key lying near the reference span
    v = rng.normal(size=d) * 0.1
    # support selection in the Woodbury-reduced space is out of scope here; use greedy on a sub-sampled candidate pool of 512 columns
    pool = np.argsort(-np.abs(k))[:512]
    Cpool = M.second_moment(X[:, pool], lam)
    Sg, _ = M.select_greedy(k[pool], Cpool, s)
    S = sorted(pool[Sg].tolist())
    Css = M.second_moment(X[:, S], lam)
    D = np.zeros((d, n))
    D[:, S] = np.outer(v, np.linalg.solve(Css, k[S])) / float(k[S] @ np.linalg.solve(Css, k[S]))
    rep = quant_report(W, D, k[:, None], v[:, None])
    rep["constraint_residual_fp64"] = float(np.linalg.norm(D @ k - v))
    rep["shape"] = [d, n]; rep["support"] = len(S)
    return rep


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    pc = json.loads(Path(sys.argv[2]).read_text())
    base, meta = fetch_actor(campaign.CACHE)
    pool = ProcessPoolExecutor(6)
    T0, episodes = time.time(), 0
    nom = list(pool.map(P._roll, [(base.tensors, "nominal", s, None, True) for s in P.DEV])); episodes += 64
    cwr = list(pool.map(P._roll, [(base.tensors, "crosswind", s, None, True) for s in P.DEV])); episodes += 64
    Xn = np.concatenate([r["h2"] for r in nom]).astype(np.float64)
    Xc = np.concatenate([r["h2"] for r in cwr]).astype(np.float64)
    lam = 1e-3 * float(np.trace(Xn.T @ Xn / len(Xn)) / 65)
    Cn, Cc = M.second_moment(Xn, lam), M.second_moment(Xc, lam)
    W3a = np.concatenate([base.tensors["w3"], base.tensors["b3"][:, None]], 1).astype(np.float64)
    keys = []
    for p in [q for q in pc["patches"] if q["delta_return"] > 20][:2]:
        i = P.DEV.index(p["seed"]); h = cwr[i]["h2"][p["t"]].astype(np.float64)
        logits = W3a @ h
        v = np.zeros(4); v[p["action"]] = max(0.0, float(np.max(np.delete(logits, p["action"])) - logits[p["action"]])) + P.MARGIN
        keys.append((h, v))
    hold = list(pool.map(P._roll, [(base.tensors, "nominal", s, None, True) for s in range(50064, 50096)])); episodes += 32
    Xh = np.concatenate([r["h2"] for r in hold]).astype(np.float64); top0 = np.argmax(Xh @ W3a.T, 1)

    def ev(t, cond):
        chunks = [P.SEL[i::6] for i in range(6)]
        parts = list(pool.map(P._evaluate, [(t, cond, c) for c in chunks]))
        return {k: float(np.mean([q[k] for q in parts])) for k in parts[0]}
    rows = [{"name": "base", "cw": ev(base.tensors, "crosswind"), "nom": ev(base.tensors, "nominal")}]; episodes += 128
    quant = {}
    for m in (1, 2):
        K, V = np.stack([k for k, _ in keys[:m]], 1), np.stack([v for _, v in keys[:m]], 1)
        Dn = M.min_energy_edit(K, V, Cn)
        Dmean = M.min_energy_edit(K, V, 0.5 * (Cn + Cc))
        Drob, w, worst = M.robust_mixture(K, V, [Cn, Cc])
        for name, D in ((f"m{m}_nominalC", Dn), (f"m{m}_meanC", Dmean), (f"m{m}_robust", Drob)):
            t = P.with_output_delta(base, D)
            rows.append({"name": name, "cost_nominal": M.edit_cost(D, Cn), "cost_crosswind": M.edit_cost(D, Cc),
                         "worst_case_cost": max(M.edit_cost(D, Cn), M.edit_cost(D, Cc)), "mixture_weights": w.tolist() if "robust" in name else None,
                         "nominal_flip": float(np.mean(np.argmax(Xh @ (W3a + D).T, 1) != top0)),
                         "cw": ev(t, "crosswind"), "nom": ev(t, "nominal")}); episodes += 128
            quant[name] = quant_report(W3a, D, K, V)
    syn = synthetic_qwen_shaped()
    res = {"phase_c_json": str(sys.argv[2]), "simulator_episodes": episodes, "wall_seconds": time.time() - T0, "rows": rows,
           "quantization_actor_output_layer": quant, "quantization_synthetic_qwen_shaped": syn}
    (out / "phase_d_results.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"{r['name']:14s} worst={r.get('worst_case_cost', 0):.3g} cn={r.get('cost_nominal', 0):.3g} cc={r.get('cost_crosswind', 0):.3g} "
              f"flip={r.get('nominal_flip', 0):.3f} cw_goal={r['cw']['goal']:.3f} nom_goal={r['nom']['goal']:.3f}")
    for k, v in quant.items():
        print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items()})
    print("synthetic", syn)


if __name__ == "__main__":
    main()
