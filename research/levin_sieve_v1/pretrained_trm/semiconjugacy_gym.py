"""LSV-18: source-locked finite-horizon semiconjugacy falsification on Sudoku TRM.

An observable q(h) and small learned transition F are tested by comparing
F(q(h)) with q(R(h)) AND by rolling F without teacher-state refreshes.
The goal-cell-only representation is not assumed Markov: the frozen block
mixes all 81 tokens. A context-aware observation is a second candidate.

CPU empirical protocol; no formal neural robustness/semiconjugacy claim.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SCHEMA = "LSV-18/learned-semiconjugacy-v1"
WIDTH = 16
RADII = (0.5, 2.0)
HORIZONS = (1, 2, 3)
SITES = ("goal", "outside")
SEEDS = {"train": (71, 103), "validation": (211, 223), "test": (307, 311)}

def split_for_puzzle(i: int) -> str:
    return ("train", "train", "validation", "test")[i % 4]

def perturbation(state, cell: int, site: str, radius: float, seed: int):
    """A deterministic matched-L2 perturbation with an independent seed."""
    if site not in SITES or not (0 < radius <= 10):
        raise ValueError("bad intervention site or L2 radius")
    if state.shape != (81, 48):
        raise ValueError("expected one complete 81x48 hidden state")
    index = cell if site == "goal" else (cell + 17) % 81
    rng = torch.Generator().manual_seed(seed)
    v = torch.randn(48, generator=rng, dtype=torch.float64)
    v = v / v.norm()
    result = state.clone()
    result[index] += radius * v
    return result

def gather_observations(states: torch.Tensor, cells: torch.Tensor, context: bool):
    """Select goal token, optionally adding grid/row/column means.

    states: [sample, time=4, row-major token=81, width=48]
    """
    if states.ndim != 4 or states.shape[1:] != (4, 81, 48):
        raise ValueError("states shape must be [N,4,81,48]")
    n = len(states)
    if cells.shape != (n,) or bool(((cells < 0) | (cells >= 81)).any()):
        raise ValueError("invalid goal-cell index")
    i = torch.arange(n)[:, None]
    t = torch.arange(4)[None, :]
    goal = states[i, t, cells[:, None]]
    if not context:
        return goal
    grid = states.view(n, 4, 9, 9, 48)
    row = grid[i, t, cells[:, None] // 9].mean(dim=-2)
    col = grid[i, t, :, cells[:, None] % 9].mean(dim=-2)
    global_mean = states.mean(dim=2)
    return torch.cat((goal, global_mean, row, col), dim=-1)

def collect_trajectories(model, inputs, targets, *, max_cases: int):
    """Frozen teacher; sample splits and noise seeds fixed before training."""
    if type(max_cases) is not int or max_cases < 8 or max_cases > len(inputs):
        raise ValueError("max_cases must be 8..fixed roster size")
    torch.set_num_threads(1)
    raw = model.eval()
    with torch.inference_mode():
        states1 = raw.reason_states(inputs[:max_cases])[1].detach().double()
    teacher = copy.deepcopy(raw).double().eval()
    initial, cells, puzzle_ids, kinds, splits = [], [], [], [], []
    for i in range(max_cases):
        cell = int((inputs[i] == 0).to(torch.int64).argmax())
        if inputs[i, cell] != 0:
            raise ValueError("frozen puzzle lacks a goal cell")
        split = split_for_puzzle(i)
        options = [("clean", 0.0, -1, states1[i])]
        for seed in SEEDS[split]:
            for radius in RADII:
                for site in SITES:
                    options.append((site, radius, seed,
                                    perturbation(states1[i], cell, site, radius,
                                                 1800 + i * 10000 + seed)))
        for site, radius, seed, state in options:
            initial.append(state)
            cells.append(cell)
            puzzle_ids.append(i)
            kinds.append({"site": site, "radius": radius, "seed": seed})
            splits.append(split)

    z0 = torch.stack(initial)
    batch = len(z0)
    sequence = [torch.empty((batch, 81, 48), dtype=torch.float32)
                for _ in range(4)]
    logits = torch.empty((batch, 4, 10), dtype=torch.float32)
    cell_index = torch.tensor(cells, dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, batch, 48):
            end = min(start + 48, batch)
            z = z0[start:end]
            idx = torch.arange(end - start)
            selected = cell_index[start:end]
            for step in range(4):
                sequence[step][start:end] = z.float()
                logits[start:end, step] = teacher.head(z)[idx, selected].float()
                if step < 3:
                    z = teacher.block(z)
    return {
        "states": torch.stack(sequence, dim=1),
        "logits": logits,
        "cells": cell_index,
        "puzzle_ids": puzzle_ids,
        "splits": splits,
        "kinds": kinds,
        "answers": torch.tensor([int(targets[i, cells[j]])
                                 for j, i in enumerate(puzzle_ids)], dtype=torch.long),
    }

def nonclosure_witness(dataset):
    """Same goal token at t=0, different remote token, changed future logits."""
    indices = {}
    for i, (p, k) in enumerate(zip(dataset["puzzle_ids"], dataset["kinds"])):
        if k["site"] == "clean":
            indices[p] = i
    rows = []
    for i, (p, k) in enumerate(zip(dataset["puzzle_ids"], dataset["kinds"])):
        if dataset["splits"][i] != "test" or k["site"] != "outside":
            continue
        reference = indices[p]
        local_equal = bool(torch.equal(dataset["states"][i, 0, dataset["cells"][i]],
                                       dataset["states"][reference, 0, dataset["cells"][i]]))
        if not local_equal:
            raise AssertionError("off-site negative control changed local token")
        delta = (dataset["logits"][i, 1:] - dataset["logits"][reference, 1:]).abs()
        rows.append({"puzzle_id":p,"radius":k["radius"],"seed":k["seed"],
                     "max_future_logit_abs":float(delta.max()),
                     "horizon_max_abs":[float(delta[h].max()) for h in range(3)]})
    # A goal-token-only predictor must output the same future logits for the
    # clean source and all four off-site edits of a single puzzle. The
    # squared-error minimizer for this alias class is its arithmetic mean.
    # This gives a measurable lower bound independent of any student training.
    group_floors,all_targets=[],[]
    for pid,reference in sorted(indices.items()):
        if dataset["splits"][reference]!="test":
            continue
        members=[reference]+[j for j,(p,k) in
            enumerate(zip(dataset["puzzle_ids"],dataset["kinds"]))
            if p==pid and k["site"]=="outside"]
        targets=dataset["logits"][members,1:]
        if len(members)!=5:
            raise ValueError("expected clean and four off-site edits per puzzle")
        floor=(targets-targets.mean(0,keepdim=True)).square().mean((0,2))
        group_floors.append(floor)
        all_targets.append(targets)
    mean_floor=torch.stack(group_floors).mean(0) if group_floors else torch.zeros(3)
    total=torch.cat(all_targets).var(dim=0,unbiased=False).mean(-1) if all_targets else torch.zeros(3)
    return {"pairs":len(rows),
            "different_future_1e-6":sum(r["max_future_logit_abs"]>1e-6 for r in rows),
            "max_future_logit_abs":max((r["max_future_logit_abs"] for r in rows),default=0),
            "local_only_logit_mse_floor_by_horizon":[float(x) for x in mean_floor],
            "floor_as_fraction_of_variance_by_horizon":[float(x/y.clamp_min(1e-12))
                                                         for x,y in zip(mean_floor,total)],
            "rows":rows,
            "interpretation":"finite floating-point witness and within-alias squared-error floor"}

class LatentModel(nn.Module):
    def __init__(self, input_dim: int, nonlinear: bool, width: int = WIDTH):
        super().__init__()
        self.encoder = (nn.Sequential(nn.Linear(input_dim, 32), nn.Tanh(),
                                      nn.Linear(32, width), nn.Tanh())
                        if nonlinear else nn.Linear(input_dim, width))
        self.delta = (nn.Sequential(nn.Linear(width, 32), nn.Tanh(),
                                    nn.Linear(32, width))
                      if nonlinear else nn.Linear(width, width))
        self.readout = nn.Linear(width, 10)
        # Start with approximately stationary macro dynamics.
        final = self.delta[-1] if nonlinear else self.delta
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def next(self, z):
        return z + 0.15 * self.delta(z)

    def forward(self, x):
        z = self.encoder(x)
        output = self.readout(z)
        rolled = []
        current = z[:, 0]
        for _ in range(3):
            current = self.next(current)
            rolled.append(self.readout(current))
        next_error = self.next(z[:, :-1]) - z[:, 1:]
        return z, output, torch.stack(rolled, dim=1), next_error

def make_normalization(x, y, train):
    x_train, y_train = x[train].reshape(-1, x.shape[-1]), y[train].reshape(-1, 10)
    xm, xs = x_train.mean(0), x_train.std(0, unbiased=False).clamp_min(0.05)
    ym, ys = y_train.mean(0), y_train.std(0, unbiased=False).clamp_min(0.05)
    return (xm, xs, ym, ys)

def objective(model, x, y):
    z, forced, rolled, comm = model(x)
    dynamics = comm.square().mean()
    variance = z.reshape(-1, z.shape[-1]).std(dim=0, unbiased=False)
    noncollapse = torch.relu(0.12 - variance).square().mean()
    reconstruction = (forced - y).square().mean()
    rollout = (rolled - y[:, 1:]).square().mean()
    loss = rollout + 0.6 * reconstruction + 0.15 * dynamics + 0.03 * noncollapse
    return loss, {"rollout":rollout,"forced":reconstruction,
                  "dynamics":dynamics,"noncollapse":noncollapse}

def metrics(model, x, y, norm, selected, truth, puzzle_ids):
    xm,xs,ym,ys = norm
    model.eval()
    with torch.inference_mode():
        z, forced, rolled, comm = model(x[selected])
        yy = y[selected] * ys + ym
        rr = rolled * ys + ym
        ff = forced[:, 1:] * ys + ym
        target = yy[:, 1:]
        next_latent = z[:, 1:]
        spread = (next_latent -
                  next_latent.reshape(-1, next_latent.shape[-1]).mean(dim=0)[None,None,:]
                  ).square().mean()
        dynamical = comm.square().mean()
        out = {
            "test_sample_count":len(selected),
            "test_puzzles":len(set(puzzle_ids[i] for i in selected.tolist())),
            "rolled_logit_mse":float((rr-target).square().mean()),
            "teacher_forced_logit_mse":float((ff-target).square().mean()),
            "semiconjugacy_mse":float(dynamical),
            "test_latent_variance":float(spread),
            "scale_normalized_semiconjugacy_mse":
              float(dynamical / spread.clamp_min(1e-12)),
            "future_teacher_label_agreement":[],
            "future_target_correct_fraction":[],
            "rollout_sudoku_correct_fraction":[],
            "future_mean_tv":[],
            "future_teacher_forced_mse":[float((ff[:,h]-target[:,h]).square().mean())
                                          for h in range(3)],
            "future_rolled_mse":[float((rr[:,h]-target[:,h]).square().mean())
                                   for h in range(3)]
        }
        labels=truth[selected]
        for h in range(3):
            output=rr[:,h].argmax(dim=-1)
            teacher=target[:,h].argmax(dim=-1)
            out["future_teacher_label_agreement"].append(float((output==teacher).float().mean()))
            out["future_target_correct_fraction"].append(float((teacher==labels).float().mean()))
            out["rollout_sudoku_correct_fraction"].append(float((output==labels).float().mean()))
            out["future_mean_tv"].append(float((rr[:,h].softmax(-1)-
                                           target[:,h].softmax(-1)).abs().sum(-1).mean()/2))
        # Aggregate per-puzzle metrics, since edited states share the same source puzzle.
        per_case = {}
        for j,i in enumerate(selected.tolist()):
            pid = puzzle_ids[i]
            err=float((rr[j]-target[j]).square().mean())
            tv=float((rr[j,-1].softmax(-1)-target[j,-1].softmax(-1)).abs().sum()/2)
            per_case.setdefault(pid,[]).append((err,tv))
        out["per_puzzle"]=[{"case":pid,"mean_rollout_mse":sum(a for a,b in vals)/len(vals),
                            "mean_final_tv":sum(b for a,b in vals)/len(vals)}
                           for pid,vals in sorted(per_case.items())]
        return out

def train_candidate(x, y, dataset, kind:str, *, max_epochs=200,seed=1818):
    nonlinear = kind.endswith("_nonlinear")
    context = kind.startswith("context_")
    obs = gather_observations(dataset["states"],dataset["cells"],context).float()
    case_ids=torch.tensor(dataset["puzzle_ids"])
    labels=dataset["answers"]
    split=dataset["splits"]
    idx={s:torch.tensor([i for i,name in enumerate(split) if name==s],
                        dtype=torch.long) for s in ("train","validation","test")}
    if any(len(ids)==0 for ids in idx.values()):
        raise ValueError("missing a puzzle partition")
    norm=make_normalization(obs,y,idx["train"])
    xm,xs,ym,ys=norm
    features=((obs-xm)/xs).clamp(-40,40)
    targets=((y-ym)/ys).clamp(-40,40)
    torch.manual_seed(seed)
    model=LatentModel(features.shape[-1],nonlinear)
    count=sum(p.numel() for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.0001)
    best=float("inf")
    saved=None
    best_epoch=0
    age=0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss,_=objective(model,features[idx["train"]],targets[idx["train"]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        model.eval()
        with torch.no_grad():
            vl,_=objective(model,features[idx["validation"]],targets[idx["validation"]])
            val=float(vl)
        if val < best - 1e-5:
            best=val
            saved={k:v.detach().clone() for k,v in model.state_dict().items()}
            best_epoch=epoch+1
            age=0
        else:
            age+=1
            if age>=35:
                break
    if saved is None:
        raise RuntimeError("training failed to produce finite validation")
    model.load_state_dict(saved)
    scores=metrics(model,features,targets,norm,idx["test"],labels,dataset["puzzle_ids"])
    scores.update({"best_validation_objective":best,"best_epoch":best_epoch,
                   "trainable_parameters":count,"bottleneck":WIDTH,
                   "input_feature_dimension":features.shape[-1],
                   "uses_context":context,
                   "nonlinear":nonlinear})
    return scores

def baselines(dataset):
    logits=dataset["logits"]
    selected=torch.tensor([i for i,s in enumerate(dataset["splits"]) if s=="test"])
    teacher=logits[selected,1:]
    last=logits[selected,0:1].expand(-1,3,-1)
    train=torch.tensor([i for i,s in enumerate(dataset["splits"]) if s=="train"])
    means=logits[train,1:].mean(0,keepdim=True)
    constant=means.expand(len(selected),-1,-1)
    truth=dataset["answers"][selected]
    def report(pred):
        return {"rolled_logit_mse":float((pred-teacher).square().mean()),
                "future_teacher_label_agreement":[float((pred[:,h].argmax(-1)==
                       teacher[:,h].argmax(-1)).float().mean()) for h in range(3)],
                "future_sudoku_correct_fraction":[float((pred[:,h].argmax(-1)==
                       truth).float().mean()) for h in range(3)],
                "future_mean_tv":[float((pred[:,h].softmax(-1)-
                       teacher[:,h].softmax(-1)).abs().sum(-1).mean()/2)
                       for h in range(3)]}
    return {"persistence":report(last),"train_mean_only":report(constant)}

def run(model,inputs,answers,sha,roster,*,max_cases=64,max_epochs=200):
    dataset=collect_trajectories(model,inputs,answers,max_cases=max_cases)
    observed={}
    for kind in ("local_linear","local_nonlinear","context_linear","context_nonlinear"):
        observed[kind]=train_candidate(None,dataset["logits"],dataset,kind,
                                       max_epochs=max_epochs,seed=1818+
                                       ("nonlinear" in kind)*13+
                                       ("context" in kind)*71)
    splits=dataset["splits"]
    samples={s:sum(t==s for t in splits) for s in SEEDS}
    puzzle_counts={s:len(set(dataset["puzzle_ids"][i] for i,t in enumerate(splits)
                              if t==s)) for s in SEEDS}
    return {"schema":SCHEMA,"status":"empirical_teacher_locked_macro_rollout",
            "checkpoint_sha256":sha,"roster_sha256":roster,
            "puzzle_count":max_cases,"training_epochs_max":max_epochs,
            "puzzle_partition":"i mod 4 (train,train,validation,test)",
            "perturbation_seeds":SEEDS,"radii":list(RADII),
            "samples_per_split":samples,"puzzles_per_split":puzzle_counts,
            "source":"44,001-parameter external TRM-inspired Sudoku checkpoint",
            "no_gpu_consumed":True,
            "aliasing":nonclosure_witness(dataset),
            "baselines":baselines(dataset),
            "candidates":observed,
            "caveat":"q sees selected token and optional context summary; no global semiconjugacy or general compression certified"}

def main():
    from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
    p=argparse.ArgumentParser()
    p.add_argument("--lock",type=Path,default=HERE/"source_lock.json")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--fetch",action="store_true")
    p.add_argument("--max-cases",type=int,default=64)
    p.add_argument("--max-epochs",type=int,default=200)
    p.add_argument("--out",type=Path,default=Path("runs/lsv18/receipt.json"))
    args=p.parse_args()
    lock=json.loads(args.lock.read_text())
    if lock.get("schema")!="LSV-05/pretrained-trm-source-lock-v1":
        raise ValueError("source lock schema mismatch")
    ckpt,sha=locked_bytes(lock,args.checkpoint,args.fetch)
    model=load_pretrained(lock,ckpt)
    (x,y,_),roster=prepare_roster(lock)
    result=run(model,x,y,sha,roster,max_cases=args.max_cases,max_epochs=args.max_epochs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    # Compact verdict while retaining complete per-case evidence in the file.
    summary={k:v for k,v in result.items() if k not in ("candidates","aliasing")}
    summary["candidates"]={k:{kk:vv for kk,vv in v.items() if kk!="per_puzzle"}
                            for k,v in result["candidates"].items()}
    summary["aliasing"]={k:v for k,v in result["aliasing"].items() if k!="rows"}
    print("LSV18",json.dumps(summary,sort_keys=True))

if __name__=="__main__":
    main()
