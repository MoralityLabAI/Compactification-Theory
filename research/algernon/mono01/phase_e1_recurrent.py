"""MONO-01 Phase E1: rank-one eigenvalue-shift edit on a small trained tanh RNN (CPU, 4 threads).

Task (noisy delayed recall): x_t = (s*[t==0], 0.5*N(0,1)), s ~ U(-1,1); the net outputs s at every step. Trained at
T=40, read out at t=40 (preservation) and t=160 (long horizon). The edit shifts the leading real eigenvalue of the mean
Jacobian diag(mean(1-h^2)) W_hh; controls are random / same-support random / one-step-gradient edits of equal norm.
Stages: train | calib | test | all. calib never touches the test generator; test runs once (test_ledger.json).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monoshot as M

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent / "receipts" / "phase_e1"
CFG = dict(nin=2, hidden=48, noise=0.5, T_train=40, T_long=160, steps=2500, batch=128, lr=3e-3, model_seed=0)
SPLITS = {"train": dict(lo=10_000, hi=19_999), "calib": dict(lo=20_000, hi=20_999, chunks=4, per=250),
          "test": dict(lo=30_000, hi=30_999, chunks=8, per=250)}
# Pre-registered before any edited model was evaluated. Rules drafted first; the baseline calibration spectrum of the noise-0.5 model
# (leading real mu=0.996, well conditioned) was then inspected and no rule was changed. Same rules for every configuration.
PREREG = dict(
    mode="leading real eigenvalue (largest value) of the mean Jacobian diag(mean_{b,t=1..39}(1-h_t^2)) W_hh, calibration T=40 trajectories",
    rules={"x0.95": ["scale", 0.95], "x0.90": ["scale", 0.90], "unit": ["fixed", 1.0]},
    spaces={"J": "edit W so the mean Jacobian eigenvalue moves: dW = diag(1/Dbar) dJ", "W": "edit the eigenvalue of W_hh itself"},
    variants=[["J", False], ["W", False], ["J", True]],
    ls_scales=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01], extra_arms=["Jc:unit"],
    extension_note="J+comp variants (one scalar readout gain refit on calibration t=40), the gradient line search, the zero_edit control and "
                   "the extra test arm Jc:unit (the principled slow-mode fix) were added after the first 6 candidates of the noise-0.5 model "
                   "failed preservation on calibration; the test roster had not been touched",
    selection="argmin calibration MSE@160 over the 9 candidates (3 rules x variants J, W, J+comp) with MSE@40 <= 1.10*base+1e-4; "
              "if none feasible, argmin MSE@40; test runs the selected candidate even if it does not beat baseline on calibration",
    controls="rho=||dW_sel||_F; 20 dense random rank-one; 20 random rank-one on the top-12 rows x top-12 cols of dW_sel; "
             "one-step gradient (full rank, top singular component, calibration line search over step scales) of calibration T=160 MSE; "
             "zero edit (readout refit only); best-on-calibration dense draw; controls get the same readout refit as the selected candidate",
    n_draws=20, support_k=12, boot=4000, grid=[-0.12, -0.09, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.09, 0.12], rel_tol=0.10)
log_lines: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


# ---------------- data ----------------
def gen(seed: int, n: int, T: int):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n)
    x = np.zeros((n, T, CFG["nin"]))
    x[:, 0, 0] = s
    x[:, :, 1] = CFG["noise"] * rng.normal(size=(n, T))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)


def split_data(name: str, T: int):
    sp = SPLITS[name]
    parts = [gen(sp["lo"] + k, sp["per"], T) for k in range(sp["chunks"])]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def roster() -> dict:
    spans = [(v["lo"], v["hi"]) for v in SPLITS.values()]
    assert all(spans[i][1] < spans[j][0] for i in range(len(spans)) for j in range(i + 1, len(spans))), "seed ranges overlap"
    r = {"cfg": CFG, "splits": SPLITS, "train_seeds_used": [SPLITS["train"]["lo"], SPLITS["train"]["lo"] + CFG["steps"] - 1]}
    r["sha256"] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
    return r


def data_hash(X: torch.Tensor, s: torch.Tensor) -> str:
    return hashlib.sha256(X.numpy().tobytes() + s.numpy().tobytes()).hexdigest()


# ---------------- model ----------------
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(CFG["nin"], CFG["hidden"], nonlinearity="tanh", batch_first=True)
        self.out = torch.nn.Linear(CFG["hidden"], 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return h, self.out(h).squeeze(-1)


def train(steps=None):
    torch.manual_seed(CFG["model_seed"])
    net = Net()
    steps = steps or CFG["steps"]
    opt = torch.optim.Adam(net.parameters(), lr=CFG["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve = []
    for k in range(steps):
        X, s = gen(SPLITS["train"]["lo"] + k, CFG["batch"], CFG["T_train"])
        _, y = net(X)
        loss = ((y - s[:, None]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if k % 100 == 0 or k == steps - 1:
            curve.append((k, loss.item()))
    return net, curve


def get_W(net) -> np.ndarray:
    return net.rnn.weight_hh_l0.detach().double().numpy().copy()


def apply_edit(net, dW: np.ndarray) -> Net:
    e = copy.deepcopy(net)
    with torch.no_grad():
        e.rnn.weight_hh_l0.add_(torch.as_tensor(dW, dtype=torch.float32))
    return e


def edited(net, dW, comp, Xc, sc) -> Net:
    """Edit W_hh; with comp, also refit one scalar readout gain alpha on calibration labels at t=40 (never test labels)."""
    e = apply_edit(net, dW)
    e.alpha = 1.0
    if comp:
        with torch.no_grad():
            y = e(Xc[:, :CFG["T_train"]])[1][:, -1]
            e.alpha = float((y * sc).sum() / (y ** 2).sum())
            e.out.weight.mul_(e.alpha)
            e.out.bias.mul_(e.alpha)
    return e


# ---------------- spectral diagnostics ----------------
def mode_of(W: np.ndarray, d: np.ndarray) -> dict:
    """Leading real eigenmode of J = diag(d) W with l^T r = 1 normalisation."""
    J = d[:, None] * W
    ev, R = np.linalg.eig(J)
    L = np.linalg.inv(R)
    real = np.flatnonzero(np.abs(ev.imag) < 1e-9)
    i = int(real[np.argmax(ev.real[real])])
    r, l = R[:, i].real, L[i].real
    return dict(J=J, ev=ev, R=R, i=i, mu=float(ev[i].real), r=r, l=l, kappa=float(np.linalg.norm(r) * np.linalg.norm(l) / abs(l @ r)))


def spectrum(W, d, ref_r=None, K=200) -> dict:
    m = mode_of(W, d)
    R, ev, J = m["R"], m["ev"], m["J"]
    j = m["i"] if ref_r is None else int(np.argmax(np.abs(R.conj().T @ ref_r) / np.linalg.norm(R, axis=0)))
    P, g = np.eye(len(W)), [1.0]
    for _ in range(K):
        P = J @ P
        g.append(float(np.linalg.norm(P, 2)))
    top = np.argsort(-np.abs(ev))[:5]
    return dict(rho=float(np.abs(ev).max()), mu_lead=m["mu"], kappa_lead=m["kappa"], cond_R=float(np.linalg.cond(R)),
                mu_tracked=float(ev[j].real), mu_tracked_imag=float(ev[j].imag), transient_max=max(g), transient_argk=int(np.argmax(g)),
                top_eigs=[[float(ev[t].real), float(ev[t].imag)] for t in top])


def lyapunov(h: torch.Tensor, W: np.ndarray, seed=0) -> dict:
    """Mean finite-time top Lyapunov exponent: one random tangent vector pushed through J_t = diag(1-h_t^2) W."""
    B, T, n = h.shape
    v = torch.randn(B, n, generator=torch.Generator().manual_seed(seed))
    v = v / v.norm(dim=1, keepdim=True)
    Wt = torch.as_tensor(W, dtype=torch.float32).T
    logs, out = torch.zeros(B), {}
    for t in range(1, T):
        v = (1 - h[:, t] ** 2) * (v @ Wt)
        nr = v.norm(dim=1).clamp_min(1e-30)
        logs += nr.log()
        v = v / nr[:, None]
        if t == CFG["T_train"] - 1:
            out["lyap40"] = float(logs.mean() / t)
    out["lyap160"] = float(logs.mean() / (T - 1))
    return out


def amp_diag(net, X, s) -> list:
    """Is the drift amplitude dependent? Per |s| tercile: readout gain (slope of y on s) at t=40/160, Jacobian eigenvalue at t=100..160."""
    with torch.no_grad():
        h, y = net(X)
    W, out = get_W(net), []
    for lo, hi in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
        m = (s.abs() >= lo) & (s.abs() < hi)
        d = (1 - h[m][:, 100:] ** 2).mean((0, 1)).double().numpy()
        gain = lambda t_: float((y[m, t_] * s[m]).sum() / (s[m] ** 2).sum())
        out.append(dict(abs_s=[round(lo, 2), round(min(hi, 1.0), 2)], n=int(m.sum()), gain40=gain(39), gain160=gain(-1),
                        mu_late=mode_of(W, d)["mu"], mean_d_late=float(d.mean())))
    return out


def evaluate(net, X, s, ref_r=None):
    W = get_W(net)
    with torch.no_grad():
        h, y = net(X)
    e = ((y - s[:, None]) ** 2).numpy()
    res = dict(mse40=float(e[:, 39].mean()), mse160=float(e[:, -1].mean()),
               sat160=float((h[:, -1].abs() > 0.95).float().mean()), h_norm160=float(h[:, -1].norm(dim=1).mean()))
    for H in (40, 160):
        d = (1 - h[:, 1:H] ** 2).mean((0, 1)).double().numpy()
        res[f"jac{H}"] = spectrum(W, d, ref_r)
    res.update(lyapunov(h, W), alpha=getattr(net, "alpha", 1.0))
    ew = np.linalg.eigvals(W)
    res["rho_W"], res["mu_W_lead"] = float(np.abs(ew).max()), float(ew[np.abs(ew.imag) < 1e-9].real.max())
    return res, dict(sq40=e[:, 39], sq160=e[:, -1]), h


# ---------------- edits ----------------
def target_mu(rule, mu):
    return rule[1] * mu if rule[0] == "scale" else rule[1]


def eig_edit(W, Dbar, space, mu_t):
    d = Dbar if space == "J" else np.ones_like(Dbar)
    m = mode_of(W, d)
    mu_t = mu_t(m["mu"]) if callable(mu_t) else mu_t
    return M.eigen_shift_edit(m["J"], m["r"], m["l"], mu_t, m["mu"]) / d[:, None], m, mu_t


def scaled(D, rho):
    return D * (rho / np.linalg.norm(D))


def rand_dense(rng, n, rho):
    return scaled(np.outer(rng.normal(size=n), rng.normal(size=n)), rho)


def rand_support(rng, rows, cols, n, rho):
    u, v = np.zeros(n), np.zeros(n)
    u[rows], v[cols] = rng.normal(size=len(rows)), rng.normal(size=len(cols))
    return scaled(np.outer(u, v), rho)


def grad_edits(net, X, s, rho):
    m = copy.deepcopy(net)
    _, y = m(X)
    ((y[:, -1] - s) ** 2).mean().backward()
    g = m.rnn.weight_hh_l0.grad.double().numpy()
    U, S, Vt = np.linalg.svd(g)
    return {"grad_full": scaled(-g, rho), "grad_rank1": scaled(-np.outer(U[:, 0], Vt[0]), rho)}


def paired_boot(d: np.ndarray, seed=0) -> list:
    idx = np.random.default_rng(seed).integers(0, len(d), (PREREG["boot"], len(d)))
    m = d[idx].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def rank_corr(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def point(base, res):
    """Compact per-candidate record: deltas against the baseline evaluation of the same split."""
    j, jb = res["jac40"], base["jac40"]
    return dict(mse40=res["mse40"], mse160=res["mse160"], d_mse40=res["mse40"] - base["mse40"], d_mse160=res["mse160"] - base["mse160"],
                mu_tracked40=j["mu_tracked"], d_mu40=j["mu_tracked"] - jb["mu_lead"], mu_lead40=j["mu_lead"], rho40=j["rho"],
                d_rho40=j["rho"] - jb["rho"], transient40=j["transient_max"], mu_tracked160=res["jac160"]["mu_tracked"],
                rho160=res["jac160"]["rho"], lyap40=res["lyap40"], lyap160=res["lyap160"], d_lyap160=res["lyap160"] - base["lyap160"],
                d_transient40=j["transient_max"] - jb["transient_max"], sat160=res["sat160"], rho_W=res["rho_W"], alpha=res["alpha"])


def prediction_table(pts):
    out = {}
    for xk in ("d_mu40", "d_rho40", "d_transient40"):
        for yk in ("d_mse160", "d_lyap160", "d_mse40"):
            out[f"spearman({xk},{yk})"] = rank_corr([p[xk] for p in pts], [p[yk] for p in pts])
    out["n_points"] = len(pts)
    return out


# ---------------- stages ----------------
def stage_train(args, out):
    t0 = time.time()
    net, curve = train(args.steps)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "model.npz", **{k: v.numpy() for k, v in net.state_dict().items()})
    log(f"train {time.time() - t0:.1f}s loss {curve[0][1]:.4f} -> {curve[-1][1]:.4f}")
    return net, dict(curve=curve, wall_s=time.time() - t0)


def calib_context(net):
    X, s = split_data("calib", CFG["T_long"])
    base, sq, h = evaluate(net, X, s)
    W, Dbar = get_W(net), (1 - h[:, 1:40] ** 2).mean((0, 1)).double().numpy()
    return X, s, base, sq, W, Dbar, mode_of(W, Dbar)


def support_of(dW):
    k = PREREG["support_k"]
    return np.argsort(-np.linalg.norm(dW, axis=1))[:k], np.argsort(-np.linalg.norm(dW, axis=0))[:k]


def control_edits(net, Xc, sc, dW):
    """Equal-norm controls for the edit dW: random dense / random on the same top rows x cols / gradient / zero (readout refit only)."""
    rho, n = float(np.linalg.norm(dW)), len(dW)
    rows, cols = support_of(dW)
    out = {}
    for j in range(PREREG["n_draws"]):
        out[f"dense{j}"] = rand_dense(np.random.default_rng(5000 + j), n, rho)
        out[f"support{j}"] = rand_support(np.random.default_rng(6000 + j), rows, cols, n, rho)
    out.update(grad_edits(net, Xc, sc, rho))                             # gradient of calibration T=160 MSE (uses calibration labels)
    out["zero_edit"] = np.zeros((n, n))
    return out, rho, float(np.linalg.norm(dW[np.ix_(rows, cols)]) ** 2 / rho ** 2)


def stage_calib(net, args):
    t0 = time.time()
    X, s, base, _, W, Dbar, bm = calib_context(net)
    log(f"calib baseline mse40 {base['mse40']:.5f} mse160 {base['mse160']:.5f} mu {bm['mu']:.4f} kappa {bm['kappa']:.2f} "
        f"rho {base['jac40']['rho']:.4f} transient {base['jac40']['transient_max']:.2f} lyap40/160 {base['lyap40']:.4f}/{base['lyap160']:.4f}")
    res = dict(roster=roster(), data_sha256=data_hash(X, s), baseline=base, candidates=[], grid=[], arms={})
    pts, cands = [], {}
    ev = lambda D, comp: evaluate(edited(net, D, comp, X, s), X, s, bm["r"])[0]
    for space, comp in PREREG["variants"]:
        for name, rule in PREREG["rules"].items():
            dW, m, mu_t = eig_edit(W, Dbar, space, lambda mu, rule=rule: target_mu(rule, mu))
            p = dict(name=f"{space}{'c' if comp else ''}:{name}", space=space, comp=comp, rule=name, mu_base=m["mu"], mu_target=mu_t,
                     kappa=m["kappa"], fro=float(np.linalg.norm(dW)), **point(base, ev(dW, comp)))
            res["candidates"].append(p), pts.append(p), cands.__setitem__(p["name"], (dW, p))
            log(f"  cand {p['name']:8s} fro {p['fro']:.4f} mu {p['mu_base']:.4f}->{mu_t:.4f} meas {p['mu_tracked40']:.4f} alpha {p['alpha']:.3f} "
                f"mse40 {p['mse40']:.5f} mse160 {p['mse160']:.5f} lyap160 {p['lyap160']:.4f}")
    for dl in PREREG["grid"]:
        dW, m, mu_t = eig_edit(W, Dbar, "J", lambda mu, dl=dl: mu + dl)
        p = dict(name=f"grid{dl:+.2f}", delta=dl, mu_target=mu_t, fro=float(np.linalg.norm(dW)), **point(base, ev(dW, False)))
        res["grid"].append(p), pts.append(p)
        log(f"  grid {dl:+.2f} fro {p['fro']:.4f} meas {p['mu_tracked40']:.4f} mse40 {p['mse40']:.5f} mse160 {p['mse160']:.5f} lyap160 {p['lyap160']:.4f}")
    feas = [p for p in res["candidates"] if p["mse40"] <= (1 + PREREG["rel_tol"]) * base["mse40"] + 1e-4]
    sel = min(feas, key=lambda p: p["mse160"]) if feas else min(res["candidates"], key=lambda p: p["mse40"])
    res["selected"] = dict(name=sel["name"], comp=sel["comp"], feasible=bool(feas), beats_baseline_160=sel["mse160"] < base["mse160"])
    log("selected", res["selected"])
    for name in dict.fromkeys([sel["name"]] + PREREG["extra_arms"]):
        dW, comp = cands[name][0], cands[name][1]["comp"]
        edits, rho, sup = control_edits(net, X, s, dW)
        ctrl = {"dense": [], "support": [], "grad": {}}
        for j in range(PREREG["n_draws"]):
            for fam in ("dense", "support"):
                ctrl[fam].append(dict(name=f"{fam}{j}", **point(base, ev(edits[f"{fam}{j}"], comp))))
        for q in ("grad_full", "grad_rank1", "zero_edit"):
            ctrl["grad"][q] = point(base, ev(edits[q], comp))
        ls = {f: point(base, ev(edits["grad_full"] * f, comp)) for f in PREREG["ls_scales"]}
        f_best = min(ls, key=lambda f: ls[f]["mse160"])
        ctrl["grad"]["grad_ls"], ctrl["grad_ls_scale"] = ls[f_best], f_best
        res["arms"][name] = dict(comp=comp, fro=rho, support_energy_frac=sup, controls=ctrl, amp_diag=amp_diag(edited(net, dW, comp, X, s), X, s),
                                 prediction_calib=prediction_table(pts + ctrl["dense"] + ctrl["support"] + list(ctrl["grad"].values())[:3]))
        log(f"  arm {name}: fro {rho:.4f} support energy {sup:.2f}")
        for fam in ("dense", "support"):
            a = np.array([[p["d_mse40"], p["d_mse160"], p["d_mu40"]] for p in ctrl[fam]])
            log(f"    ctrl {fam}: d_mse40 {a[:,0].mean():+.5f} d_mse160 {a[:,1].mean():+.5f} (min {a[:,1].min():+.5f}) d_mu {a[:,2].mean():+.4f} +- {a[:,2].std():.4f}")
        for q, p in ctrl["grad"].items():
            log(f"    ctrl {q}: d_mse40 {p['d_mse40']:+.5f} d_mse160 {p['d_mse160']:+.5f} d_mu {p['d_mu40']:+.4f}")
        log(f"    grad line search best scale {f_best}: " + ", ".join(f"{f}:{ls[f]['mse160']:.4f}" for f in ls))
    res["amp_diag_baseline"] = amp_diag(net, X, s)
    log("  amplitude diagnostic (baseline):", json.dumps(res["amp_diag_baseline"]))
    res["wall_s"] = time.time() - t0
    log(f"calib done {res['wall_s']:.1f}s")
    return res


def stage_test(net, calib, X, s, tag="test"):
    """Held-out evaluation of the selected candidate (and the declared extra arms) with their controls. Ran once on the test roster."""
    t0 = time.time()
    Xc, sc, _, _, W, Dbar, bm = calib_context(net)                       # edits, alpha and gradients use calibration only
    base, bsq, _ = evaluate(net, X, s, bm["r"])
    out = dict(tag=tag, n_seq=len(s), baseline=base, selected=calib["selected"], arms={})
    for name, arm in calib["arms"].items():
        space, rname = name.split(":")
        comp, space = arm["comp"], space[0]
        dW, _, _ = eig_edit(W, Dbar, space, lambda mu: target_mu(PREREG["rules"][rname], mu))
        edits, rho, _ = control_edits(net, Xc, sc, dW)
        edits["grad_ls"] = edits["grad_full"] * arm["controls"]["grad_ls_scale"]
        best = int(np.argmin([p["mse160"] for p in arm["controls"]["dense"]]))
        edits["dense_best_on_calib"], edits["eigen"] = edits[f"dense{best}"], dW
        recs, sqs = {}, {}
        for q, D in edits.items():
            r, sq, _ = evaluate(edited(net, D, comp, Xc, sc), X, s, bm["r"])
            recs[q], sqs[q] = point(base, r), sq
        res = dict(fro=rho, comp=comp, dense_best_idx=best, records=recs)
        for h in ("40", "160"):
            k_ = f"sq{h}"
            fam = {f: np.mean([sqs[f"{f}{j}"][k_] for j in range(PREREG["n_draws"])], axis=0) for f in ("dense", "support")}
            fam.update({q: sqs[q][k_] for q in ("grad_full", "grad_rank1", "grad_ls", "zero_edit", "dense_best_on_calib")})
            res[f"boot_{h}"] = {"eigen_vs_base": paired_boot(sqs["eigen"][k_] - bsq[k_])}
            for f, v in fam.items():
                res[f"boot_{h}"][f"{f}_vs_base"] = paired_boot(v - bsq[k_])
                res[f"boot_{h}"][f"eigen_vs_{f}"] = paired_boot(sqs["eigen"][k_] - v)
        res["prediction_test"] = prediction_table([dict(p, name=nm) for nm, p in recs.items()])
        out["arms"][name] = res
        log(f"{tag} {name}: base mse40 {base['mse40']:.5f} mse160 {base['mse160']:.5f}; eigen mse40 {recs['eigen']['mse40']:.5f} mse160 {recs['eigen']['mse160']:.5f}")
    out["wall_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["train", "calib", "test", "all", "dry_test"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--noise", type=float, default=CFG["noise"])
    ap.add_argument("--seed", type=int, default=CFG["model_seed"])
    ap.add_argument("--out", default=None, help="output dir (default receipts/phase_e1/noise<noise>_seed<seed>)")
    args = ap.parse_args()
    CFG.update(noise=args.noise, model_seed=args.seed)
    tag = f"noise{args.noise}_seed{args.seed}"
    out = Path(args.out) if args.out else OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    net, tr = stage_train(args, out)
    (out / "roster.json").write_text(json.dumps(roster(), indent=1))
    (out / "prereg.json").write_text(json.dumps(PREREG, indent=1))
    Xc, sc = split_data("calib", CFG["T_long"])
    base_c, _, _ = evaluate(net, Xc, sc)
    tr.update(calib_mse40=base_c["mse40"], calib_mse160=base_c["mse160"], var_s=1 / 3, w_in_col_norms=net.rnn.weight_ih_l0.norm(dim=0).tolist())
    (out / "train.json").write_text(json.dumps(tr, indent=1))
    log(f"[{tag}] calib MSE40 {base_c['mse40']:.5f} MSE160 {base_c['mse160']:.5f} (mean-predictor MSE 0.3333)")
    if args.stage == "train":
        return
    calib = stage_calib(net, args)
    (out / "calib_results.json").write_text(json.dumps(calib, indent=1))
    if args.stage in ("calib", "dry_test"):
        if args.stage == "dry_test":                                     # code-path check on calibration data, not the test roster
            res = stage_test(net, calib, Xc[:200], sc[:200], tag="dry_test_on_calibration_subset")
            (out / "dry_test_results.json").write_text(json.dumps(res, indent=1))
        return
    ledger = OUT / f"test_ledger_{tag}.json"
    assert not ledger.exists(), "held-out test already evaluated; refusing to look again"
    Xt, st = split_data("test", CFG["T_long"])
    res = stage_test(net, calib, Xt, st)
    res["data_sha256"] = data_hash(Xt, st)
    (out / "test_results.json").write_text(json.dumps(res, indent=1))
    ledger.write_text(json.dumps(dict(evaluated_at=time.strftime("%Y-%m-%d %H:%M:%S"), roster_sha256=roster()["sha256"], data_sha256=res["data_sha256"])))
    log(f"[{tag}] total wall {time.time() - t0:.1f}s")
    (out / "run_log.txt").write_text("\n".join(log_lines))


if __name__ == "__main__":
    main()
