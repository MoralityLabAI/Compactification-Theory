"""CPU observer fitting and state/rank-specific Haar nulls. No model imports."""
import csv
import hashlib
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from core import RANKS, LAYERS, seed, haar, ridge, rank_p, holm, features, write_csv
from preflight import ROOT, RESULTS, save, sha


def optimize(x, y, initial, gradients=None, medians=None):
    import torch
    torch.set_num_threads(2)
    x, y = torch.tensor(x), torch.tensor(y)
    u = torch.nn.Parameter(torch.tensor(initial.copy()))
    opt = torch.optim.Adam([u], lr=0.015)
    g = None if gradients is None else torch.tensor(gradients)
    med = None if medians is None else torch.tensor(medians)
    history = []
    for _ in range(100):
        opt.zero_grad()
        q, _ = torch.linalg.qr(u, mode="reduced")
        z = x @ q
        w = torch.linalg.solve(z.T @ z / len(x) + 0.01*torch.eye(z.shape[1]), z.T @ y / len(x))
        mse = ((z @ w-y)**2).mean()
        penalty = torch.tensor(0.)
        if g is not None:
            dark = 1-(g @ q).square().sum(1)/g.square().sum(1)
            penalty = (dark.clamp(min=0)/med).square().mean()
        loss = mse+penalty
        loss.backward()  # CPU observer only; Qwen is never in this graph.
        opt.step()
        history.append([float(mse.detach()), float(penalty.detach())])
    with torch.no_grad():
        q, _ = torch.linalg.qr(u, mode="reduced")
    return q.numpy(), history


def fit_all(rows, activations, margins, sketches):
    revision = json.loads((ROOT/"model_manifest.json").read_text())["revision"]
    train = np.asarray([i for i,r in enumerate(rows) if r["split"] == "train"])
    ys = np.asarray([1. if r["answer"] == 0 else -1. for r in rows])
    gmap = {x["state_id"]: np.asarray(x["gradient"], dtype=float) for x in sketches}
    gradient_states = [i for i in train if rows[i]["state_id"] in gmap
                       and np.linalg.norm(gmap[rows[i]["state_id"]]) > 1e-10]
    gtrain = np.asarray([gmap[rows[i]["state_id"]] for i in gradient_states])
    null_medians = {}
    # Nested Haar frames have the correct marginal Haar law at every rank.
    # Distinctness fingerprint hashes a small gauge-invariant projector block.
    null_path = RESULTS / "null_distribution.csv"
    with open(null_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state_id", "layer", "rank", "null_index", "subspace_fingerprint",
                         "activation_capture", "sketch_capture"])
        for li, layer in enumerate(LAYERS):
            mu = activations[train, li].astype(float).mean(0)
            for i, row in enumerate(rows):
                x = activations[i, li].astype(float)-mu
                gx = gmap.get(row["state_id"]) if layer == 12 else None
                rng_seed = seed("null", revision, row["state_id"], layer)
                a = np.random.default_rng(rng_seed).standard_normal((128, 1024, 16))
                q, _ = np.linalg.qr(a, mode="reduced")
                xc = np.einsum("ndk,d->nk", q, x)**2
                xcap = np.cumsum(xc, axis=1)/max(float(x@x), 1e-30)
                gc = None if gx is None or gx@gx < 1e-20 else np.cumsum(
                        np.einsum("ndk,d->nk", q, gx)**2, axis=1)/(gx@gx)
                for rank in RANKS:
                    fps = [hashlib.sha256((b[:8,:rank] @ b[:8,:rank].T).round(13).tobytes()).hexdigest()[:16]
                           for b in q]
                    if len(set(fps)) != 128:
                        raise ValueError("Non-distinct null subspaces")
                    captures = [] if gc is None else gc[:,rank-1]
                    null_medians[(i,layer,rank)] = dict(
                        activation=xcap[:,rank-1].copy(), sensitivity=np.asarray(captures).copy(),
                        dark=None if gc is None else float(np.median(1-captures)))
                    for j in range(128):
                        writer.writerow([row["state_id"], layer, rank, j, fps[j],
                                         format(xcap[j,rank-1], ".12g"),
                                         "" if gc is None else format(gc[j,rank-1], ".12g")])
                if i % 64 == 0:
                    print(f"CPU nulls layer {layer}, state {i}/{len(rows)}", flush=True)
    observers, fit_records, per_state, per_rank = {}, {}, [], []
    npz = {}
    for li, layer in enumerate(LAYERS):
        x = activations[:,li].astype(float)
        mu = x[train].mean(0)
        scale = float(np.sqrt(np.mean((x[train]-mu)**2)))
        normalized = (x-mu)/scale
        _, _, vh = np.linalg.svd(normalized[train], full_matrices=False)
        for rank in RANKS:
            choices = {"pca": vh[:rank].T, "random": haar(1024, rank, ("observer", layer, rank))}
            choices["task_only"], history = optimize(normalized[train], ys[train], choices["pca"])
            fit_records[f"{layer}_{rank}_task_only"] = history
            if layer == 12 and len(gtrain):
                med = np.asarray([null_medians[(i,12,rank)]["dark"] for i in gradient_states])
                if np.any(med <= 0):
                    raise ValueError("Undefined null-normalized residual objective")
                choices["causal"], history = optimize(normalized[train], ys[train], choices["pca"], gtrain, med)
                fit_records[f"{layer}_{rank}_causal"] = history
            for name,u in choices.items():
                z = normalized @ u
                w = ridge(z[train], ys[train])
                # Same training population/budget and no held-out margin labels.
                margin_center = float(np.mean(margins[train]))
                wm = ridge(z[train], margins[train]-margin_center)
                key = f"{layer}_{rank}_{name}"
                observer = dict(u=u, mean=mu, scale=scale, task_w=w, margin_w=wm,
                                margin_center=margin_center)
                observers[key] = observer
                npz.update({f"{key}_{k}": np.asarray(v) for k,v in observer.items()})
                for i,row in enumerate(rows):
                    nm = null_medians[(i,layer,rank)]
                    xv = x[i]-mu
                    capture = float(np.linalg.norm(u.T @ xv)**2/max(xv@xv,1e-30))
                    gx = gmap.get(row["state_id"]) if layer == 12 else None
                    sc = None if gx is None or gx@gx < 1e-20 else float(np.linalg.norm(u.T @ gx)**2/(gx@gx))
                    per_state.append(dict(state_id=row["state_id"], split=row["split"], rule=row["rule"],
                        delay=row["delay"], layer=layer, rank=rank, observer=name,
                        task_prediction=float(z[i]@w), task_target=ys[i], task_correct=int((z[i]@w>=0)==(ys[i]>0)),
                        task_squared_error=float((z[i]@w-ys[i])**2), activation_capture=capture,
                        activation_null_median=float(np.median(nm["activation"])),
                        activation_upper_p=rank_p(capture,nm["activation"]),
                        sketch_capture="" if sc is None else sc,
                        sketch_null_median="" if sc is None else float(np.median(nm["sensitivity"])),
                        sketch_upper_p="" if sc is None else rank_p(sc,nm["sensitivity"]),
                        A="", B="", C="", recurrent_closure="unavailable", null_count=128,
                        sensitivity_scope="measured_16_direction_projection" if sc is not None else "unmeasured"))
                for split in ("train","calibration","heldout"):
                    group = [r for r in per_state if r["layer"]==layer and r["rank"]==rank
                             and r["observer"]==name and r["split"]==split]
                    per_rank.append(dict(layer=layer, rank=rank, observer=name, split=split, n=len(group),
                        task_accuracy=float(np.mean([r["task_correct"] for r in group])),
                        task_mse=float(np.mean([r["task_squared_error"] for r in group])),
                        optimizer_steps=100 if name in ("causal","task_only") else 0))
            print(f"CPU observers layer {layer} rank {rank}", flush=True)
    # Holm family contains four methods even when causal is unavailable.
    families = {}
    for r in per_state:
        families.setdefault((r["state_id"],r["layer"],r["rank"]),[]).append(r)
    for group in families.values():
        by = {r["observer"]:r for r in group}
        for metric in ("activation", "sketch"):
            pvals = [by[n].get(metric+"_upper_p") if n in by else None
                     for n in ("causal","task_only","random","pca")]
            pvals = [None if p == "" else p for p in pvals]
            for name,p in zip(("causal","task_only","random","pca"),holm(pvals)):
                if name in by:
                    by[name][metric+"_holm_p"] = "" if p is None else p
    # Leave-one-rule-out readouts for the complete layer/rank grid.
    transfer = []
    for key, o in observers.items():
        layer, rank, name = key.split("_",2)
        li = LAYERS.index(int(layer))
        z = ((activations[:,li].astype(float)-o["mean"])/o["scale"]) @ o["u"]
        for source_rule in ("copy","flip"):
            tr = [i for i in train if rows[i]["rule"] == source_rule]
            te = [i for i,r in enumerate(rows) if r["split"]=="heldout" and r["rule"]!=source_rule]
            w = ridge(z[tr],ys[tr])
            transfer.append(dict(layer=layer,rank=rank,observer=name,trained_readout_rule=source_rule,
                observer_training_rules="both; readout-only cross-rule transfer",n=len(te),
                accuracy=float(np.mean((z[te]@w>=0)==(ys[te]>0)))))
    # Genuine observer transfer at the primary layer/rank: all fits see one rule.
    for source_rule in ("copy","flip"):
        tr = [i for i in train if rows[i]["rule"]==source_rule]
        te = [i for i,r in enumerate(rows) if r["split"]=="heldout" and r["rule"]!=source_rule]
        raw = activations[:,1].astype(float)
        mu = raw[tr].mean(0)
        scale = float(np.sqrt(np.mean((raw[tr]-mu)**2)))
        x = (raw-mu)/scale
        _,_,vh = np.linalg.svd(x[tr],full_matrices=False)
        choices = dict(pca=vh[:7].T,random=haar(1024,7,("transfer",source_rule)))
        choices["task_only"],_ = optimize(x[tr],ys[tr],choices["pca"])
        gi = [i for i in gradient_states if rows[i]["rule"]==source_rule]
        if not gi:
            raise ValueError(f"No measured training sensitivities for {source_rule}")
        choices["causal"],_ = optimize(x[tr],ys[tr],choices["pca"],
            np.asarray([gmap[rows[i]["state_id"]] for i in gi]),
            np.asarray([null_medians[(i,12,7)]["dark"] for i in gi]))
        for name,u in choices.items():
            z = x@u
            w = ridge(z[tr],ys[tr])
            mc = float(np.mean(margins[tr]))
            wm = ridge(z[tr],margins[tr]-mc)
            key = f"transfer_{source_rule}_{name}"
            observer = dict(u=u,mean=mu,scale=scale,task_w=w,margin_w=wm,margin_center=mc)
            observers[key] = observer
            npz.update({f"{key}_{k}":np.asarray(v) for k,v in observer.items()})
            transfer.append(dict(layer=12,rank=7,observer=name,trained_readout_rule=source_rule,
                observer_training_rules=source_rule,n=len(te),accuracy=float(np.mean((z[te]@w>=0)==(ys[te]>0)))))
    f = features(rows)
    iw = ridge(f[train],ys[train])
    for split in ("train","calibration","heldout"):
        ix = [i for i,r in enumerate(rows) if r["split"]==split]
        per_rank.append(dict(layer="input",rank=5,observer="input_rule_cue",split=split,n=len(ix),
            task_accuracy=float(np.mean((f[ix]@iw>=0)==(ys[ix]>0))),
            task_mse=float(np.mean((f[ix]@iw-ys[ix])**2)),optimizer_steps=0))
    write_csv(RESULTS/"per_state.csv", per_state)
    write_csv(RESULTS/"per_rank.csv", per_rank)
    write_csv(RESULTS/"cross_rule_transfer.csv", transfer)
    save(RESULTS/"optimizer_history.json", fit_records)
    np.savez(RESULTS/"observers.npz", **npz)
    return observers, per_rank


def main():
    rows = json.loads((RESULTS/"dataset.json").read_text())
    activations = np.load(RESULTS/"activations.npy", mmap_mode="r")
    margins = np.load(RESULTS/"margins.npy", mmap_mode="r")
    sketches = json.loads((RESULTS/"sensitivity_sketches.json").read_text())
    fit_all(rows,activations,margins,sketches)


if __name__ == "__main__":
    main()
