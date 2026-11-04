"""MONO-01 Phase C: causal localization + analytic output-layer edits on the pretrained LunarLander actor.

Run in the pinned policy-surgery environment:
  uv run --frozen --extra weights python phase_c.py OUT_DIR [--max-fail N] [--workers W]

Seeds (upstream protocol): development 50000-50063 (localization + reference covariance + edit construction),
selection 60000-60063 (all model selection here). Approval 61000-61255 and final test 62000-62255 are never touched.
Every simulator episode is counted. Simulator results are platform-specific; all arms are compared on this machine.
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monoshot as M
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar
from policy_surgery import campaign
from policy_surgery.policy import Policy, fetch_actor
from policy_surgery.world import CONDITIONS, engine_power

DEV = list(range(50000, 50064))
SEL = list(range(60000, 60064))
WINDOW, STRIDE, MARGIN = 8, 16, 0.5


def rollout(tensors, cond_name, seed, override=None, record=False):
    """Deterministic episode; override=(t0, action, width) forces an action on steps [t0, t0+width)."""
    pol, cond = Policy(tensors), CONDITIONS[cond_name]
    env = gym.make("LunarLander-v3", gravity=cond.gravity, enable_wind=cond.wind > 0, wind_power=cond.wind,
                   turbulence_power=cond.turbulence, max_episode_steps=1000)
    total, h2s, obs_l, acts = 0.0, [], [], []
    try:
        obs, _ = env.reset(seed=int(seed))
        for step in range(1000):
            logits, h1, h2 = pol.forward(obs, hidden=True)
            a = int(np.argmax(logits))
            if override and override[0] <= step < override[0] + override[2]:
                a = override[1]
            if record:
                h2s.append(np.append(h2, 1.0)); obs_l.append(obs.copy()); acts.append(a)
            with engine_power(cond):
                obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        safe = bool(term and not env.unwrapped.game_over and not env.unwrapped.lander.awake)
        goal = bool(safe and all(leg.ground_contact and env.unwrapped.helipad_x1 <= leg.position.x <= env.unwrapped.helipad_x2
                                 for leg in env.unwrapped.legs))
        out = {"return": float(total), "goal": goal, "steps": step + 1, "crashed": bool(env.unwrapped.game_over)}
        if record:
            out.update(h2=np.array(h2s, dtype=np.float32), obs=np.array(obs_l, dtype=np.float32), acts=np.array(acts))
        return out
    finally:
        env.close()


def _roll(args):
    return rollout(*args)


def _evaluate(args):
    tensors, cond, seeds = args
    r = [rollout(tensors, cond, s) for s in seeds]
    return {"goal": float(np.mean([x["goal"] for x in r])), "return": float(np.mean([x["return"] for x in r])),
            "crashed": float(np.mean([x["crashed"] for x in r]))}


def with_output_delta(base: Policy, D: np.ndarray) -> dict:
    t = {k: v.copy() for k, v in base.tensors.items()}
    aug = np.concatenate([t["w3"], t["b3"][:, None]], 1).astype(np.float64) + D
    t["w3"], t["b3"] = aug[:, :64].astype(np.float32), aug[:, 64].astype(np.float32)
    return t


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--max-fail", type=int, default=64); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base, meta = fetch_actor(campaign.CACHE)
    T0, episodes = time.time(), 0
    pool = ProcessPoolExecutor(a.workers)
    log = {"upstream_commit": "46f95e64dc9057e6d025b7141b607230905c04ee", "base_sha256": meta["actor_sha256"],
           "dev_seeds": [DEV[0], DEV[-1]], "selection_seeds": [SEL[0], SEL[-1]], "window": WINDOW, "stride": STRIDE, "margin": MARGIN}

    # --- stage 1: nominal reference second moment (dev seeds) and crosswind base episodes with traces
    nom = list(pool.map(_roll, [(base.tensors, "nominal", s, None, True) for s in DEV])); episodes += len(DEV)
    Xn = np.concatenate([r["h2"] for r in nom]).astype(np.float64)
    lam = 1e-3 * float(np.trace(Xn.T @ Xn / len(Xn)) / Xn.shape[1])
    C = M.second_moment(Xn, lam)
    cw = list(pool.map(_roll, [(base.tensors, "crosswind", s, None, True) for s in DEV])); episodes += len(DEV)
    # determinism check: a no-override replay must equal the recorded episode exactly
    chk = rollout(base.tensors, "crosswind", DEV[0], None, True); episodes += 1
    assert chk["return"] == cw[0]["return"] and chk["steps"] == cw[0]["steps"], "simulator not deterministic in-process"
    fail = [i for i, r in enumerate(cw) if not r["goal"]][: a.max_fail]
    log["dev_nominal_goal_rate"] = float(np.mean([r["goal"] for r in nom]))
    log["dev_crosswind_goal_rate"] = float(np.mean([r["goal"] for r in cw]))
    log["reference_states"], log["lambda"] = int(len(Xn)), lam

    # --- stage 2: counterfactual action-window patching on failed crosswind episodes
    jobs, meta_j = [], []
    for i in fail:
        T = cw[i]["steps"]
        for t in range(0, max(T - WINDOW, 1), STRIDE):
            for act in range(4):
                jobs.append((base.tensors, "crosswind", DEV[i], (t, act, WINDOW), False)); meta_j.append((i, t, act))
    res = list(pool.map(_roll, jobs, chunksize=8)); episodes += len(jobs)
    best = {}
    for (i, t, act), r in zip(meta_j, res):
        d = r["return"] - cw[i]["return"]
        if cw[i]["acts"][t] == act:
            continue                                           # not an intervention
        if i not in best or d > best[i]["delta_return"]:
            best[i] = {"episode": i, "t": t, "action": act, "delta_return": d, "goal": r["goal"]}
    patches = sorted(best.values(), key=lambda p: -p["delta_return"])
    log["patch_candidates"] = len(jobs)
    log["patches"] = [{**p, "seed": DEV[p["episode"]]} for p in patches[:16]]
    log["patches_goal_recovered"] = int(sum(p["goal"] for p in patches))
    log["patches_positive"] = int(sum(p["delta_return"] > 0 for p in patches))

    # --- stage 3: keys/values -> analytic edits and baselines
    def keyval(p):
        h = cw[p["episode"]]["h2"][p["t"]].astype(np.float64)
        logits = (np.concatenate([base.tensors["w3"], base.tensors["b3"][:, None]], 1) @ h)
        gap = max(0.0, float(np.max(np.delete(logits, p["action"])) - logits[p["action"]])) + MARGIN
        v = np.zeros(4); v[p["action"]] = gap
        return h, v
    W3n = np.concatenate([base.tensors["w3"], base.tensors["b3"][:, None]], 1).astype(np.float64)
    lg = Xn @ W3n.T
    srt = np.sort(lg, 1)
    margin_n = srt[:, -1] - srt[:, -2]
    wgt = 1.0 / (margin_n ** 2 + 0.25)
    Cm = 1e-3 * float(np.trace(Xn.T @ (Xn * wgt[:, None])) / len(Xn) / 65) * np.eye(65) + (Xn * wgt[:, None]).T @ Xn / len(Xn)
    Cm *= float(np.trace(C) / np.trace(Cm))                    # same overall scale as C so predicted costs are comparable
    def soft_edit(K, V, Cx, rho):
        return rho * V @ K.T @ np.linalg.inv(rho * K @ K.T + Cx)
    good = [p for p in patches if p["delta_return"] > 20]
    log["patches_used_pool"] = len(good)
    edits, kv, rng = {}, {}, np.random.default_rng(7)
    for m in (1, 2, 4, 8):
        if len(good) < m:
            continue
        H_, V_ = zip(*[keyval(p) for p in good[:m]])
        K, V = np.stack(H_, 1), np.stack(V_, 1)
        before = set(edits)
        DC = M.min_energy_edit(K, V, C)
        DI = M.min_energy_edit(K, V, np.eye(65) * float(np.trace(C) / 65))
        edits[f"m{m}_analyticC"] = DC
        edits[f"m{m}_analyticI"] = DI
        edits[f"m{m}_analyticCm"] = M.min_energy_edit(K, V, Cm)
        for rho in (1.0, 10.0, 100.0):
            edits[f"m{m}_softC_rho{rho:g}"] = soft_edit(K, V, C, rho)
            edits[f"m{m}_softCm_rho{rho:g}"] = soft_edit(K, V, Cm, rho)
        for r in range(3):
            R = rng.normal(size=DC.shape); edits[f"m{m}_randnorm{r}"] = R * (np.linalg.norm(DC) / np.linalg.norm(R))
        if m == 1:
            k, v = K[:, 0], V[:, 0]
            for s in (4, 8, 16):
                Sg, _ = M.select_greedy(k, C, s)
                Sb, _ = M.select_beam(k, C, s, width=8)
                Sr = sorted(rng.choice(65, s, replace=False).tolist())
                for nm, S in (("greedy", Sg), ("beam", Sb), ("randsupport", Sr)):
                    edits[f"m1_s{s}_{nm}"] = M.support_edit(K, V, C, S)
        for n_ in set(edits) - before:
            kv[n_] = (K, V)
    # held-out nominal states for flip-rate (dev seeds 50064..50095, base policy)
    hold = list(pool.map(_roll, [(base.tensors, "nominal", s, None, True) for s in range(50064, 50096)])); episodes += 32
    Xh = np.concatenate([r["h2"] for r in hold]).astype(np.float64)
    W3a = np.concatenate([base.tensors["w3"], base.tensors["b3"][:, None]], 1).astype(np.float64)
    top0 = np.argmax(Xh @ W3a.T, 1)

    # --- stage 4: evaluate on SELECTION seeds only (crosswind task, nominal retention)
    def ev(tensors, cond):
        chunks = [SEL[i::a.workers] for i in range(a.workers)]
        parts = list(pool.map(_evaluate, [(tensors, cond, c) for c in chunks]))
        return {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
    rows = [{"name": "base", "predicted_cost": 0.0, "nominal_flip": 0.0, "cw": ev(base.tensors, "crosswind"), "nom": ev(base.tensors, "nominal")}]
    episodes += 2 * len(SEL)
    for name, D in edits.items():
        t = with_output_delta(base, D)
        flip = float(np.mean(np.argmax(Xh @ (W3a + D).T, 1) != top0))
        rows.append({"name": name, "predicted_cost": M.edit_cost(D, C), "fro": float(np.linalg.norm(D)), "rank": int(np.linalg.matrix_rank(D)),
                     "support": int(np.count_nonzero(np.abs(D).sum(0) > 1e-12)), "nominal_flip": flip, "constraint_residual": float(np.linalg.norm(D @ kv[name][0] - kv[name][1]) / np.linalg.norm(kv[name][1])),
                     "cw": ev(t, "crosswind"), "nom": ev(t, "nominal")})
        episodes += 2 * len(SEL)
    log["rows"] = rows
    pc = np.array([r["predicted_cost"] for r in rows[1:]]); fl = np.array([r["nominal_flip"] for r in rows[1:]])
    nl = np.array([rows[0]["nom"]["goal"] - r["nom"]["goal"] for r in rows[1:]])
    log["spearman_predcost_vs_nominal_flip"] = spearman(pc, fl)
    log["spearman_predcost_vs_nominal_goal_loss"] = spearman(pc, nl)
    log["simulator_episodes"], log["wall_seconds"] = episodes, time.time() - T0
    (out / "phase_c_results.json").write_text(json.dumps(log, indent=1))
    for r in rows:
        print(f"{r['name']:22s} pred={r['predicted_cost']:.3g} flip={r['nominal_flip']:.3f} cw_goal={r['cw']['goal']:.3f} ret={r['cw']['return']:.1f}"
              f" nom_goal={r['nom']['goal']:.3f}")
    print({k: log[k] for k in log if k.startswith("spearman") or k in ("patches_positive", "patches_goal_recovered", "simulator_episodes", "wall_seconds", "dev_nominal_goal_rate", "dev_crosswind_goal_rate")})


if __name__ == "__main__":
    main()
