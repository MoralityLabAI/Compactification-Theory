"""Algernon-02B analysis of captured LOTUS states (CPU): decoders, controls, gate.

LOTUS supervises latent block b with CoT step b after its final loop, so the
state after pointer step k is read from block k-1. Views of the capture
[case, pass, block, pool, hidden]:

- diagonal:  block k-1 after k refinement passes (one pass per step);
- final:     block k-1 after all passes (where upstream supervision applies);
- lastblock: the last block after k passes (the registered A2a view);
- pass0:     block k-1 before any refinement (no recurrence);
- static:    the start-latent token, which sees only the prompt.

Each view gets the same decoder search (linear and small MLP, pooling), fitted
on train-family stages 1..8 only and selected on calibration-family stages
1..8 only; held-out families and stages 9..16 are scored once. Stage 0 is the
public start state, which the R1CS witness binds anyway. The exact verifier
scores traces and is never used to train or correct anything.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from .capture import file_sha256
from .counterfactual import validate_counterfactual
from .probe import group_metrics, wilson
from .roster import validate_roster

VIEWS = ("diagonal", "final", "lastblock", "pass0", "static")
LINEAR_L2 = (1e-4, 1e-3, 1e-2, 1e-1)
MLP_HIDDEN = (64,)
GATE_MARGIN = 0.10


def view(blocks, static, kind, pool):
    """[cases, stages=B+1, hidden] features for stage k (k=0 is the static token)."""
    import numpy as np
    n, p, b = blocks.shape[:3]
    if p != b + 1:
        raise ValueError("expected passes = blocks + 1")
    x = np.empty((n, b + 1, blocks.shape[-1]), dtype=np.float32)
    x[:, 0] = static[:, 0]
    for k in range(1, b + 1):
        x[:, k] = {"diagonal": lambda: blocks[:, k, k-1, pool], "final": lambda: blocks[:, b, k-1, pool],
                   "lastblock": lambda: blocks[:, k, b-1, pool], "pass0": lambda: blocks[:, 0, k-1, pool],
                   "static": lambda: static[:, k]}[kind]()
    return x


class Decoder:
    """Standardised features -> logits over states; linear or one-hidden-layer MLP."""

    def __init__(self, kind, param, seed):
        self.kind, self.param, self.seed = kind, param, seed

    def fit(self, X, y, n_states, extra=None):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(self.seed)
        X = torch.as_tensor(X, dtype=torch.float32)
        self.mean, self.std = X.mean(0), X.std(0, unbiased=False).clamp_min(1e-4)
        Z = self._z(X, extra)
        y = torch.as_tensor(y, dtype=torch.long)
        d = Z.shape[1]
        if self.kind == "linear":
            self.net = torch.nn.Linear(d, n_states)
            opt = torch.optim.LBFGS(self.net.parameters(), max_iter=300, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(self.net(Z), y) + self.param * self.net.weight.square().sum()
                loss.backward()
                return loss
            opt.step(closure)
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(d, self.param), torch.nn.GELU(),
                                           torch.nn.Linear(self.param, n_states))
            opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=0.05)
            for _ in range(600):
                opt.zero_grad()
                F.cross_entropy(self.net(Z), y).backward()
                opt.step()
        self.net.eval()
        return self

    def _z(self, X, extra):
        import torch
        Z = (torch.as_tensor(X, dtype=torch.float32) - self.mean) / self.std
        if extra is not None:
            Z = torch.cat([Z, torch.as_tensor(extra, dtype=torch.float32)], 1)
        return Z

    def predict(self, X, extra=None):
        import torch
        with torch.inference_mode():
            return self.net(self._z(X, extra)).argmax(-1).numpy()

    @property
    def parameters(self):
        return sum(p.numel() for p in self.net.parameters())

    def randomized(self, seed):
        """Same architecture and per-tensor Frobenius norms, random directions."""
        import copy
        import torch
        g = torch.Generator().manual_seed(seed)
        other = copy.deepcopy(self)
        with torch.no_grad():
            for p in other.net.parameters():
                r = torch.randn(p.shape, generator=g)
                p.copy_(r * p.norm() / r.norm().clamp_min(1e-12))
        return other


def stage_rows(X, idx, stages):
    import numpy as np
    return X[np.asarray(idx)][:, stages].reshape(-1, X.shape[-1])


def decode_all(dec, X, rows, extra_fn=None):
    """Decode stages 1..B for every case; stage 0 is the public start."""
    import numpy as np
    n, s = X.shape[:2]
    flat = X[:, 1:].reshape(-1, X.shape[-1])
    extra = extra_fn(n) if extra_fn else None
    pred = dec.predict(flat, extra).reshape(n, s - 1)
    return np.concatenate([np.asarray([[r["initial"]] for r in rows]), pred], 1)


def stage_onehot(n_cases, stages, width):
    """Case-major rows for stages 1..stages, one-hot over `width` depth slots."""
    import numpy as np
    out = np.zeros((n_cases * stages, width), dtype=np.float32)
    out[np.arange(n_cases * stages), np.tile(np.arange(stages), n_cases)] = 1.0
    return out


def metrics(pred, rows, steps, *, boot_seed=0, n_boot=1000):
    """probe.group_metrics plus conditional transition accuracy and case-bootstrap CI."""
    import numpy as np
    out = group_metrics(pred, rows, steps)
    truth = np.asarray([r["trace"] for r in rows])
    ok = pred == truth
    prev, nxt = ok[:, :-1], ok[:, 1:]
    out["transition_accuracy"] = float(nxt[prev].mean()) if prev.any() else None
    succ = np.asarray([[r["successors"][int(p)] for p in pr[:-1]] for r, pr in zip(rows, pred)])
    out["defect_by_step"] = [round(float(x), 6) for x in (pred[:, 1:] != succ).mean(0)]
    per_case = ok[:, 9:17].mean(1)
    rng = np.random.default_rng(boot_seed)
    boots = [per_case[rng.integers(0, len(rows), len(rows))].mean() for _ in range(n_boot)]
    out["heldout_9_16_case_bootstrap_95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    return out


def family_metrics(pred, rows, steps, seed):
    import numpy as np
    out = {"all_test": metrics(pred, rows, steps, boot_seed=seed)}
    for fam in sorted({r["family"] for r in rows}):
        m = np.asarray([r["family"] == fam for r in rows])
        out[fam] = metrics(pred[m], [r for r, k in zip(rows, m) if k], steps, boot_seed=seed)
    return out


def paired_diff(pa, pb, rows, *, seed, n_boot=2000, family=None):
    """Bootstrap over cases of acc_9_16(a) - acc_9_16(b)."""
    import numpy as np
    truth = np.asarray([r["trace"] for r in rows])
    keep = np.asarray([family is None or r["family"] == family for r in rows])
    a = (pa == truth)[keep][:, 9:17].mean(1)
    b = (pb == truth)[keep][:, 9:17].mean(1)
    rng = np.random.default_rng(seed)
    d = [(a[i] - b[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


def select_decoder(X, truth, train, cal, rows, n_states, seed, extra=None):
    """Grid over pools and decoders on train stages 1..8; choose on calibration 1..8."""
    import numpy as np
    best, trials = None, []
    for pool, Xp in enumerate(X):
        Xtr, ytr = stage_rows(Xp, train, slice(1, 9)), truth[train][:, 1:9].reshape(-1)
        for kind, param in [("linear", l) for l in LINEAR_L2] + [("mlp", h) for h in MLP_HIDDEN]:
            t0 = time.perf_counter()
            dec = Decoder(kind, param, seed).fit(Xtr, ytr, n_states)
            cal_pred = dec.predict(stage_rows(Xp, cal, slice(1, 9))).reshape(len(cal), 8)
            acc = float((cal_pred == truth[cal][:, 1:9]).mean())
            trials.append({"pool": pool, "kind": kind, "param": param, "calibration_acc_1_8": acc,
                           "train_acc_1_8": float((dec.predict(Xtr) == ytr).mean()),
                           "cpu_seconds": round(time.perf_counter() - t0, 3), "parameters": dec.parameters})
            if best is None or acc > best[0]:
                best = (acc, pool, dec)
    return best[2], best[1], trials


def input_probes(S, cases, train, cal, test, n_states, seed):
    """Linear readout of one vector per case for the prompt's own inputs and the answer.

    Targets: start state, each successor entry f(i), and the final state. Fit on
    train cases, L2 chosen on calibration cases, scored on held-out cases; the
    train-majority label is the reference. (The step count is 16 in every
    prompt, so it cannot be probed.)
    """
    import numpy as np
    n = len(cases[0]["successors"])
    targets = {"start": [r["initial"] for r in cases], "final_state": [r["trace"][-1] for r in cases]}
    targets.update({f"successor[{i}]": [r["successors"][i] for r in cases] for i in range(n)})
    out = {}
    for name, y in targets.items():
        y = np.asarray(y)
        best = None
        for lam in LINEAR_L2:
            dec = Decoder("linear", lam, seed).fit(S[train], y[train], n_states)
            acc = float((dec.predict(S[cal]) == y[cal]).mean())
            if best is None or acc > best[0]:
                best = (acc, lam, dec)
        majority = np.bincount(y[train], minlength=n_states).argmax()
        out[name] = {"heldout_accuracy": float((best[2].predict(S[test]) == y[test]).mean()),
                     "train_majority_accuracy": float((y[test] == majority).mean()),
                     "calibration_accuracy": best[0], "l2": best[1]}
    return out


def stepwise_baseline(roster, n_states, seed):
    """Learned one-step transition on the symbolic map (same task info), iterated."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    train = [r for r in roster["cases"] if r["split"] == "train"]
    S, E, Y = [], [], []
    for r in train:
        for k in range(8):
            S.append(r["trace"][k]); E.append(r["successors"]); Y.append(r["trace"][k+1])
    s = F.one_hot(torch.tensor(S), n_states).float()
    e = F.one_hot(torch.tensor(E), n_states).float()
    W = torch.zeros(n_states * n_states, n_states, requires_grad=True)
    bias = torch.zeros(n_states, requires_grad=True)
    opt = torch.optim.AdamW([W, bias], lr=0.05, weight_decay=0.0)
    for _ in range(400):
        opt.zero_grad()
        F.cross_entropy((s.unsqueeze(-1) * e).flatten(1) @ W + bias, torch.tensor(Y)).backward()
        opt.step()

    def run(rows):
        out = []
        for r in rows:
            cur, tr = r["initial"], [r["initial"]]
            edges = F.one_hot(torch.tensor(r["successors"]), n_states).float()[None]
            for _ in range(len(r["trace"]) - 1):
                st = F.one_hot(torch.tensor([cur]), n_states).float()
                cur = int(((st.unsqueeze(-1) * edges).flatten(1) @ W + bias).argmax())
                tr.append(cur)
            out.append(tr)
        return np.asarray(out)
    return run, n_states ** 3 + n_states


def heuristics(rows, train_rows, steps):
    import numpy as np
    from collections import Counter
    majority = [Counter(r["trace"][k] for r in train_rows).most_common(1)[0][0] for k in range(steps + 1)]
    out = {}
    out["copy_start"] = np.asarray([[r["initial"]] * (steps + 1) for r in rows])
    out["train_majority_by_depth"] = np.asarray([[r["initial"]] + majority[1:] for r in rows])
    indeg = [int(np.argmax(np.bincount(r["successors"], minlength=len(r["successors"])))) for r in rows]
    out["max_indegree"] = np.asarray([[r["initial"]] + [d] * steps for r, d in zip(rows, indeg)])
    out["exact_solver"] = np.asarray([r["trace"] for r in rows])
    return out


def answer_accuracy(items, rows_by_id):
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0, 0])
    for it in items:
        row = rows_by_id[it["id"]]
        for key in ("all", row["family"]):
            acc[key][0] += int(str(it["answer"]).strip() == str(row["trace"][-1]))
            acc[key][1] += 1
            acc[key][2] += int(str(it["answer"]).strip() in {str(i) for i in range(len(row["successors"]))})
    return {k: {"correct": c, "n": n, "accuracy": c / n, "wilson_95": wilson(c / n, n), "answer_in_state_range": v / n}
            for k, (c, n, v) in sorted(acc.items())}


def readout_states(texts, rows):
    """First integer in each block's own LOTUS readout text, as a decoded state."""
    import re
    import numpy as np
    out = []
    for t, r in zip(texts, rows):
        tr = [r["initial"]]
        for s in t:
            m = re.search(r"-?\d+", s)
            v = int(m.group()) if m else -1
            tr.append(v if 0 <= v < len(r["successors"]) else -1)
        out.append(tr)
    return np.asarray(out)


def readout_metrics(pred, rows):
    import numpy as np
    truth = np.asarray([r["trace"] for r in rows])
    return {"accuracy_by_step": [round(float(x), 6) for x in (pred == truth).mean(0)],
            "parsable_state_rate": float((pred[:, 1:] >= 0).mean()),
            "heldout_steps_9_16_accuracy": float((pred == truth)[:, 9:17].mean()),
            "accuracy_step_16": float((pred == truth)[:, 16].mean())}


def analyze(roster, cf, cap, cf_cap, *, per_case=None, cf_per_case=None, behaviour=(), seed=20261104):
    """Everything except file IO. cap/cf_cap: dicts with float arrays blocks, static."""
    import numpy as np
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cases, steps = roster["cases"], roster["parameters"]["steps"]
    n_states = roster["parameters"]["n"]
    truth = np.asarray([r["trace"] for r in cases])
    split = np.asarray([r["split"] for r in cases])
    train, cal, test = (np.flatnonzero(split == s) for s in ("train", "calibration", "test"))
    assert not (set(train) & set(cal) | set(train) & set(test) | set(cal) & set(test))
    test_rows = [cases[i] for i in test]
    cf_rows = cf["cases"]
    t_all = time.perf_counter()
    feats = {v: [view(cap["blocks"], cap["static"], v, p) for p in (0, 1)] for v in VIEWS}
    cf_feats = {v: [view(cf_cap["blocks"], cf_cap["static"], v, p) for p in (0, 1)] for v in VIEWS}
    results, records, selection, preds = {}, [], {}, {}

    def add(name, pred_test, extra_note=None):
        preds[name] = pred_test
        results[name] = family_metrics(pred_test, test_rows, steps, seed)
        if extra_note:
            results[name]["note"] = extra_note

    decoders = {}
    for v in VIEWS:
        dec, pool, trials = select_decoder(feats[v], truth, train, cal, cases, n_states, seed)
        decoders[v] = (dec, pool)
        selection[v] = {"pool": ["last_latent", "block_mean"][pool], "decoder": f"{dec.kind}:{dec.param}",
                        "parameters": dec.parameters, "trials": trials}
        add(f"view:{v}", decode_all(dec, feats[v][pool][test], test_rows))
    results["input_encoding"] = {
        "static_token": input_probes(cap["static"][:, 0], cases, train, cal, test, n_states, seed),
        "pass0_block0_mean": input_probes(cap["blocks"][:, 0, 0, 1], cases, train, cal, test, n_states, seed)}
    primary = max(("diagonal", "final"), key=lambda v: max(t["calibration_acc_1_8"] for t in selection[v]["trials"]))
    dec, pool = decoders[primary]
    X = feats[primary][pool]
    P0 = feats["pass0"][pool]
    rng = np.random.default_rng(seed + 1)
    Xt, P0t = X[test], P0[test]
    const = Xt.copy(); const[:, 1:] = Xt[:, 1:2]
    shuf = Xt.copy()
    for i in range(len(shuf)):
        shuf[i, 1:] = Xt[i, 1 + rng.permutation(steps)]
    rnd = P0t.copy()
    delta = np.linalg.norm(Xt - P0t, axis=-1, keepdims=True)
    u = rng.standard_normal(Xt.shape).astype(np.float32)
    rnd[:, 1:] = (P0t + delta * u / np.linalg.norm(u, axis=-1, keepdims=True))[:, 1:]
    add("control:constant_state", decode_all(dec, const, test_rows), "primary decoder; stage-1 features repeated")
    add("control:shuffled_stage", decode_all(dec, shuf, test_rows), "primary decoder; stages permuted per case")
    add("control:recurrence_removed", decode_all(dec, P0t, test_rows), "primary decoder on the same blocks before any refinement")
    add("control:random_direction", decode_all(dec, rnd, test_rows),
        "pass-0 block plus a random direction with the norm of the recurrent update")
    add("control:random_probe", decode_all(dec.randomized(seed + 2), Xt, test_rows), "random weights, trained norms")
    ytr = truth[train][:, 1:9].reshape(-1).copy()
    np.random.default_rng(seed + 3).shuffle(ytr)
    shuffled_dec = Decoder(dec.kind, dec.param, seed).fit(stage_rows(X, train, slice(1, 9)), ytr, n_states)
    add("control:label_shuffled_probe", decode_all(shuffled_dec, Xt, test_rows), "same decoder class, train labels permuted")
    # Parameter-matched feedforward: static prompt representation + one-hot depth, no recurrence.
    S = feats["static"][0]
    lin_params = (S.shape[-1] + 1) * n_states
    hidden = max(1, round((lin_params - n_states) / (S.shape[-1] + steps + 1 + n_states)))
    ff = Decoder("mlp", hidden, seed).fit(stage_rows(S, train, slice(1, 9)), truth[train][:, 1:9].reshape(-1),
                                          n_states, extra=stage_onehot(len(train), 8, steps))
    pad = lambda n: stage_onehot(n, steps, steps)
    add("control:param_matched_feedforward", decode_all(ff, S[test], test_rows, pad),
        f"MLP on static token + one-hot depth; {ff.parameters} parameters vs {lin_params} linear")
    # Mapping-change controls.
    by_fam = {}
    for j, r in enumerate(test_rows):
        by_fam.setdefault(r["family"], []).append(j)
    wrong_rows = list(test_rows)
    for fam, idx in by_fam.items():
        perm = list(idx)
        random.Random(seed + 4).shuffle(perm)
        perm = perm[1:] + perm[:1]
        for a, b in zip(idx, perm):
            wrong_rows[a] = {**test_rows[b], "initial": test_rows[a]["initial"],
                             "trace": _trace(test_rows[b]["successors"], test_rows[a]["initial"], steps)}
    p_main = preds[f"view:{primary}"]
    results["control:wrong_map_scoring"] = family_metrics(p_main, wrong_rows, steps, seed)
    results["control:wrong_map_scoring"]["note"] = "primary predictions scored against another same-family map from the same start"
    cf_pred = decode_all(dec, cf_feats[primary][pool], cf_rows)
    results["counterfactual:relabelled_truth"] = family_metrics(cf_pred, cf_rows, steps, seed)
    parent = {r["id"]: r for r in cases}
    unrel = [{**parent[r["parent_id"]], "initial": r["initial"],
              "trace": [r["initial"]] + parent[r["parent_id"]]["trace"][1:]} for r in cf_rows]
    results["counterfactual:parent_labels"] = {
        "all_test": readout_metrics(cf_pred, unrel),
        "note": "relabelled-prompt predictions scored against the parent's unrelabelled states (label-prior check)"}
    # Symbolic baselines (no LOTUS features).
    for name, p in heuristics(test_rows, [cases[i] for i in train], steps).items():
        add(f"baseline:{name}", p)
    run_step, step_params = stepwise_baseline(roster, n_states, seed)
    add("baseline:learned_stepwise_symbolic", run_step(test_rows),
        f"bilinear one-step model on the explicit map, {step_params} parameters, iterated 16 times")
    # LOTUS's own readouts and answers.
    rows_by_id = {r["id"]: r for r in cases}
    native = {}
    if per_case:
        meta = {m["id"]: m for m in per_case}
        for kind in ("final_pass", "diagonal", "pass0"):
            p = readout_states([meta[r["id"]]["readout_text"][kind] for r in test_rows], test_rows)
            native[kind] = readout_metrics(p, test_rows)
        answers = {"loops16_capture": [{"id": m["id"], "answer": m.get("answer")} for m in per_case]}
        for it in behaviour:
            answers.setdefault(f"loops{it['loops']}" if it["loops"] else "explicit_no_latents", []).append(it)
        results["behaviour"] = {k: answer_accuracy(v, rows_by_id) for k, v in answers.items()}
        for k, items in answers.items():
            for it in items:
                r = rows_by_id[it["id"]]
                records.append({"case_id": r["id"], "family": r["family"], "split": r["split"],
                                "intervention": f"behaviour:{k}", "answer": it.get("answer"),
                                "answer_text": it.get("answer_text", meta[r["id"]].get("answer_text")),
                                "truth_final": r["trace"][-1]})
    results["lotus_native_readout"] = native
    # Gate for Phase 4 (registered before the GPU run in EXPERIMENT.md).
    no_rec = ["view:pass0", "view:static", "control:param_matched_feedforward", "control:recurrence_removed",
              "control:shuffled_stage", "control:random_direction", "control:label_shuffled_probe"]
    acc = lambda k, fam="all_test": results[k][fam]["heldout_steps_9_16_accuracy"]
    strongest = max(no_rec, key=acc)
    vs_strongest = paired_diff(p_main, preds[strongest], test_rows, seed=seed)
    vs_pass0_cycle = paired_diff(p_main, preds["view:pass0"], test_rows, seed=seed, family="long_cycle")
    trivial = ["baseline:copy_start", "baseline:train_majority_by_depth", "baseline:max_indegree"]
    best_trivial_cycle = max(trivial, key=lambda k: acc(k, "long_cycle"))
    vs_trivial_cycle = paired_diff(p_main, preds[best_trivial_cycle], test_rows, seed=seed, family="long_cycle")
    gate = {"primary_view": primary, "strongest_no_recurrence_control": strongest,
            "primary_acc_9_16": acc(f"view:{primary}"), "control_acc_9_16": acc(strongest),
            "vs_strongest_control": vs_strongest, "long_cycle_vs_pass0": vs_pass0_cycle,
            "long_cycle_vs_best_trivial": {"baseline": best_trivial_cycle, **vs_trivial_cycle},
            "margin_required": GATE_MARGIN}
    gate["passed"] = (vs_strongest["diff"] >= GATE_MARGIN and vs_strongest["ci95"][0] > 0
                      and vs_pass0_cycle["ci95"][0] > 0)
    if gate["passed"]:
        gate["classification"] = "recurrence_specific_state"
    elif vs_trivial_cycle["ci95"][0] > 0 or vs_strongest["ci95"][0] > 0:
        gate["classification"] = "static_representation_decoding"
    else:
        gate["classification"] = "no_iterative_state_signal"
    # Per-case records: every test and counterfactual case under every method.
    for name, p in preds.items():
        for r, tr in zip(test_rows, p):
            records.append({"case_id": r["id"], "family": r["family"], "split": r["split"], "intervention": name,
                            "decoded": [int(x) for x in tr], "truth": r["trace"],
                            "correct_9_16": int(sum(int(a == b) for a, b in zip(tr[9:], r["trace"][9:])))})
    for r, tr in zip(cf_rows, cf_pred):
        records.append({"case_id": r["id"], "family": r["family"], "split": r["split"],
                        "intervention": f"counterfactual:view:{primary}", "sigma": r["sigma"],
                        "decoded": [int(x) for x in tr], "truth": r["trace"],
                        "correct_9_16": int(sum(int(a == b) for a, b in zip(tr[9:], r["trace"][9:])))})
    summary = {"primary_view": primary, "selection": selection, "results": results, "gate": gate,
               "analysis_cpu_seconds": round(time.perf_counter() - t_all, 2),
               "decoder_parameters": {v: decoders[v][0].parameters for v in VIEWS},
               "stepwise_parameters": step_params}
    return summary, records


def stage_summary(per_case, params, loops):
    """Time, memory and an approximate FLOP count (2 x parameters x tokens through the stack)."""
    import statistics
    secs = [m["seconds"] for m in per_case]
    tokens = [m["prompt_tokens"] + (loops + 1) * (m["latent_tokens"] + 1) + 1 + m.get("generated_tokens", 0)
              for m in per_case]
    out = {"cases": len(per_case), "seconds_total": round(sum(secs), 2),
           "seconds_per_case": {"mean": round(statistics.mean(secs), 4), "min": min(secs), "max": max(secs)},
           "tokens_through_stack_per_case_mean": round(statistics.mean(tokens), 1),
           "approx_tflop_per_case_mean": round(2 * params * statistics.mean(tokens) / 1e12, 3)}
    if per_case and "peak_allocated_bytes" in per_case[0]:
        out["peak_allocated_bytes_max"] = max(m["peak_allocated_bytes"] for m in per_case)
        out["allocated_after_case_bytes"] = [min(m["allocated_bytes"] for m in per_case),
                                             max(m["allocated_bytes"] for m in per_case)]
    return out


def compact_manifest(run, run_dir, summary, argv):
    """Committed manifest: provenance, environment, commands, timing; no per-case text."""
    import subprocess
    params = run["model"]["unique_parameters"]
    loops = run["stages"]["parity"]["loops"]
    stages = {k: stage_summary(run["stages"][k]["per_case"], params, loops)
              for k in ("smoke", "calibration_subset", "full", "counterfactual") if k in run["stages"]}
    root = Path(__file__).resolve().parents[4]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    outputs = {p.name: file_sha256(p) for p in sorted(run_dir.iterdir())
               if p.suffix in (".npz", ".jsonl") or p.name in ("parity.json", "gsm8k_sample.json", "gsm8k_preds.json")}
    return {"schema_version": "algernon-02b-manifest-v1", "run_status": run["status"],
            "started": run["started"], "finished": run.get("finished"),
            "source": run["source"], "inputs": run["inputs"], "environment": run["environment"],
            "device": run.get("device"), "precision": run.get("precision"), "seeds": {"roster": 20261104, "counterfactual": 20261104,
                                           "gsm8k_sample": run["stages"].get("gsm8k", {}).get("sample_seed"),
                                           "analysis": summary.get("seed")},
            "commands": {"gpu_run": run["argv"], "analysis": argv,
                         "upstream_eval": run["stages"].get("gsm8k", {}).get("command")},
            "analysis_code_commit": head, "model": run["model"],
            "gsm8k": {k: v for k, v in run["stages"].get("gsm8k", {}).items() if k != "command"},
            "parity": {k: run["stages"]["parity"][k] for k in ("checks", "passed", "loops", "c_thought",
                                                                "lm_head_max_abs_diff", "lm_head_argmax_agreement",
                                                                "static_token_max_drift_across_passes")},
            "stages": stages, "drift": run["stages"].get("drift"),
            "behaviour_seconds": run["stages"].get("behaviour"),
            "peak_cuda_allocated_bytes": run.get("peak_cuda_allocated_bytes"),
            "peak_cuda_reserved_bytes": run.get("peak_cuda_reserved_bytes"),
            "full_roster_bitwise_repeat": run["stages"]["full"]["smoke_and_subset_bitwise_equal_full"],
            "outputs_outside_git_sha256": outputs}


def render_tables(summary):
    """Markdown depth-by-depth tables for the report."""
    res = summary["results"]
    names = [k for k in res if k.split(":")[0] in ("view", "control", "baseline")]
    names += ["counterfactual:relabelled_truth"]
    lines = []
    for fam in ("all_test", "long_cycle", "sink"):
        lines += [f"### {fam}", "",
                  "| method | " + " | ".join(str(k) for k in range(1, 17)) + " | 1-8 | 9-16 [95% CI] | defect | valid trace |",
                  "| --- |" + " ---: |" * 20]
        for n in names:
            m = res[n][fam]
            ci = m["heldout_9_16_case_bootstrap_95"]
            lines.append(f"| {n} | " + " | ".join(f"{x:.2f}" for x in m["accuracy_by_step"][1:])
                         + f" | {m['trained_steps_1_8_accuracy']:.3f} | {m['heldout_steps_9_16_accuracy']:.3f}"
                         f" [{ci[0]:.2f}, {ci[1]:.2f}] | {m['commuting_square_defect_rate']:.3f}"
                         f" | {m['exact_trace_valid_full_rate']:.3f} |")
        lines.append("")
    if res.get("input_encoding"):
        targets = list(next(iter(res["input_encoding"].values())))
        lines += ["### Do the prompt representations carry the task inputs? (held-out cases, linear readout)", "",
                  "| representation | " + " | ".join(targets) + " |", "| --- |" + " ---: |" * len(targets)]
        for rep, v in res["input_encoding"].items():
            lines.append(f"| {rep} | " + " | ".join(f"{v[t]['heldout_accuracy']:.2f} ({v[t]['train_majority_accuracy']:.2f})"
                                                    for t in targets) + " |")
        lines += ["", "Parenthesised: train-majority label on the same cases.", ""]
    if res.get("behaviour"):
        lines += ["### LOTUS answers (final state after 16 steps)", "",
                  "| condition | all | long_cycle | sink | random_function | permutation | answer is a state id |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for k, v in res["behaviour"].items():
            cell = lambda f: f"{v[f]['correct']}/{v[f]['n']}" if f in v else "-"
            lines.append(f"| {k} | {cell('all')} | {cell('long_cycle')} | {cell('sink')} | {cell('random_function')}"
                         f" | {cell('permutation')} | {v['all']['answer_in_state_range']:.2f} |")
        lines.append("")
    if res.get("lotus_native_readout"):
        lines += ["### LOTUS's own block readout, first integer per block (held-out)", "",
                  "| readout | parsable | 9-16 | step 16 |", "| --- | ---: | ---: | ---: |"]
        for k, v in res["lotus_native_readout"].items():
            lines.append(f"| {k} | {v['parsable_state_rate']:.2f} | {v['heldout_steps_9_16_accuracy']:.3f}"
                         f" | {v['accuracy_step_16']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def _trace(succ, start, steps):
    out = [start]
    for _ in range(steps):
        out.append(succ[out[-1]])
    return out


def load_npz(path, ids):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        if z["ids"].tolist() != ids:
            raise ValueError(f"{path}: case ids or order differ from the frozen roster")
        return {"blocks": z["blocks"].astype(np.float32), "static": z["static"].astype(np.float32)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--counterfactual", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    roster = json.loads(a.roster.read_text(encoding="utf-8"))
    validate_roster(roster)
    cf = json.loads(a.counterfactual.read_text(encoding="utf-8"))
    validate_counterfactual(cf, roster)
    run = json.loads((a.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run["status"] != "complete":
        raise ValueError(f"run status {run['status']}; refuse to analyse")
    for key, name in (("full", "full_blocks.npz"), ("counterfactual", "cf_blocks.npz")):
        if run["stages"][key]["capture"]["sha256"] != file_sha256(a.run_dir / name):
            raise ValueError(f"{name} bytes differ from the run manifest")
    cap = load_npz(a.run_dir / "full_blocks.npz", [r["id"] for r in roster["cases"]])
    cf_cap = load_npz(a.run_dir / "cf_blocks.npz", [r["id"] for r in cf["cases"]])
    behaviour = [json.loads(l) for l in (a.run_dir / "behaviour.jsonl").read_text(encoding="utf-8").splitlines()]
    summary, records = analyze(roster, cf, cap, cf_cap, per_case=run["stages"]["full"]["per_case"],
                               behaviour=behaviour, seed=a.seed)
    summary["inputs"] = {"run_manifest_sha256": file_sha256(a.run_dir / "run_manifest.json"),
                         "full_blocks_sha256": run["stages"]["full"]["capture"]["sha256"],
                         "cf_blocks_sha256": run["stages"]["counterfactual"]["capture"]["sha256"],
                         "roster_sha256": roster["roster_sha256"], "counterfactual_sha256": cf["counterfactual_sha256"]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "metrics.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with (a.out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    (a.out / "tables.md").write_text(render_tables(summary) + "\n", encoding="utf-8")
    from .gpu_run import scrub
    import sys
    manifest = compact_manifest(run, a.run_dir, {"seed": a.seed}, [scrub(x) for x in sys.argv])
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    g = summary["gate"]
    print(f"primary {g['primary_view']}: acc 9-16 {g['primary_acc_9_16']:.3f} vs {g['strongest_no_recurrence_control']} "
          f"{g['control_acc_9_16']:.3f}; gate {'PASS' if g['passed'] else 'FAIL'} ({g['classification']})")


if __name__ == "__main__":
    main()
