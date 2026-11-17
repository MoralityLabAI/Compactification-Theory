"""OIC-15: model-family inference from observable histories, with unknown initial phase.

Small CPU reference: generic GRU vs cyclic hypothesis filter with a learned
noise/likelihood scale. Both see the same visible-state samples, interventions
and declared phase count n; neither receives true initial phase or target family.
The cyclic filter knows the two candidate state-transition rules, which is a
substantial structural prior. The GRU learns from simulator-provided family labels.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TRAIN_NS = (4, 5, 6)
OOD_NS = (7, 8)
SEEDS = tuple(range(5))
SPLITS = ("full", "heldout")
STEPS = 500
BATCH = 128
HIDDEN = 32
LR_GRU = 0.007
LR_CYCLIC = 0.04


def holdout(n: int) -> tuple[int, int]:
    return (1, n - 2)


def oracle_batch(n: int, family: torch.Tensor, phase: torch.Tensor, u: torch.Tensor,
                 x0: torch.Tensor | None = None, marker: int = 0) -> torch.Tensor:
    """Return raw state samples (B,T+1) for deterministic hidden-phase simulator."""
    if x0 is None:
        x0 = torch.zeros_like(phase)
    z, x = phase.clone(), x0.clone()
    states = [x]
    for t in range(u.shape[1]):
        pulse = u[:, t].long() * (z == marker).long()
        x = x ^ 1 ^ pulse
        z = (z + family.long()) % n
        states.append(x)
    return torch.stack(states, 1)


def actions_for(n: int, name: str) -> tuple[int, ...]:
    if name == "passive":
        return (0,) * (2 * n)
    if name == "full_active":
        return (1,) * n
    if name == "latch":
        return (1,) * n + (0,) * n
    if name == "extended_latch":
        return (1,) * n + (0,) * (n + 2)
    if name == "alternating":
        return tuple(int(t % 2 == 0) for t in range(2 * n))
    if name == "random_budget":
        rng = np.random.default_rng(1951 + n)
        w = np.zeros(2 * n, dtype=np.int64)
        w[rng.choice(2 * n, n, replace=False)] = 1
        return tuple(map(int, w))
    raise ValueError(name)


def make_train_batch(n: int, batch: int, split: str, rng: np.random.Generator):
    fam = rng.integers(0, 2, size=batch, dtype=np.int64)
    allowed = np.array([i for i in range(n) if split == 'full' or i not in holdout(n)])
    phase = rng.choice(allowed, size=batch)
    # Actions and episode lengths are sampled without observing the target family.
    T = 2 * n
    action = np.zeros((batch, T), dtype=np.int64)
    # Five independent, family-blind intervention distributions.
    for i in range(batch):
        strategy = int(rng.integers(0, 5))
        if strategy == 0:
            action[i, :] = 0
        elif strategy == 1:
            action[i, :] = 1
        elif strategy == 2:
            action[i, :] = (rng.random(T) < .3)
        elif strategy == 3:
            action[i, :] = (rng.random(T) < .7)
        else:
            action[i, :n] = 1
    fam_t = torch.from_numpy(fam)
    phase_t = torch.from_numpy(phase)
    u_t = torch.from_numpy(action)
    x = oracle_batch(n, fam_t, phase_t, u_t)
    return fam_t, u_t, x


def known_mask(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0).to(torch.float32)


def derived_audit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw readout -> audit, marking adjacent transitions to erased samples unknown."""
    mask = known_mask(x)
    valid = mask[:, :-1] * mask[:, 1:]
    # Inputs are always integer 0/1 or -1 for erasure.
    event = (1 ^ x[:, :-1].clamp(min=0).long() ^ x[:, 1:].clamp(min=0).long()).float()
    return event, valid


class GenericGRU(nn.Module):
    def __init__(self, width: int = HIDDEN):
        super().__init__()
        self.hidden_width = width
        self.gru = nn.GRU(6, width, batch_first=True)
        self.head = nn.Linear(width, 1)

    def forward(self, n: int, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mask = known_mask(x)
        xx = x.clamp(min=0).float()
        # No initial phase, model-family label or simulator hidden state input.
        nnorm = torch.full_like(u.float(), 1. / n)
        inp = torch.stack([u.float(), xx[:, :-1], xx[:, 1:],
                           mask[:, :-1], mask[:, 1:], nnorm], dim=-1)
        h, _ = self.gru(inp)
        return self.head(h).squeeze(-1)


class CyclicFilter(nn.Module):
    """Equivariant phase-hypothesis filter with one learned mismatch penalty.

    Its n phase slots are fully initialized to a uniform prior and evolve as
    fixed / cyclic-shift hypotheses. The transition laws and phase-zero
    readout are specified architectural knowledge. Only sensor tolerance is
    estimated from labeled observation sequences.
    """
    def __init__(self):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, n: int, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, T = u.shape
        obs, valid = derived_audit(x)
        phase = torch.arange(n, device=u.device).reshape(1, 1, n, 1)
        family = torch.arange(2, device=u.device).reshape(1, 2, 1, 1)
        step = torch.arange(T, device=u.device).reshape(1, 1, 1, T)
        phase_now = (phase + family * step) % n
        expected = (phase_now == 0).float() * u.float()[:, None, None, :]
        mismatch = torch.abs(expected - obs[:, None, None, :]) * valid[:, None, None, :]
        cumulative = mismatch.cumsum(dim=-1)
        scale = F.softplus(self.log_beta) + 1.e-4
        evidence = torch.logsumexp(-scale * cumulative, dim=2)
        return evidence[:, 1, :] - evidence[:, 0, :]



class RawFilter(nn.Module):
    """Cyclic hypothesis filter with likelihood on raw state samples.

    The candidate traces are computed from known transition families and the
    declared initial visible bit 0. A single raw-bit substitution affects one
    likelihood factor (rather than two derived audit factors); erasures are
    skipped explicitly. The sensor mismatch scale alone is learned.
    """
    def __init__(self):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, n: int, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, T = u.shape
        phase = torch.arange(n, device=u.device).reshape(1, 1, n, 1)
        family = torch.arange(2, device=u.device).reshape(1, 2, 1, 1)
        times = torch.arange(T, device=u.device).reshape(1, 1, 1, T)
        pulse = (phase + family * times).remainder(n).eq(0).long() * u[:, None, None, :].long()
        parity_pulse = pulse.cumsum(dim=-1).remainder(2)
        baseline = torch.arange(1, T + 1, device=u.device).remainder(2).reshape(1, 1, 1, T)
        raw_pred = torch.cat((torch.zeros((B, 2, n, 1), device=u.device, dtype=torch.long),
                              baseline ^ parity_pulse), dim=-1)
        mask = known_mask(x)
        mismatch = raw_pred.ne(x[:, None, None, :]).float() * mask[:, None, None, :]
        cumulative = mismatch.cumsum(-1)[:, :, :, 1:]
        beta = F.softplus(self.log_beta) + 1.e-4
        evidence = torch.logsumexp(-beta * cumulative, dim=2)
        return evidence[:, 1, :] - evidence[:, 0, :]

def train_one(architecture: str, split: str, seed: int, steps: int = STEPS) -> tuple[nn.Module, dict]:
    if architecture not in ('gru', 'cyclic', 'raw') or split not in SPLITS:
        raise ValueError('invalid architecture/split')
    torch.manual_seed(seed * 100 + {'gru': 0, 'cyclic': 1, 'raw': 2}[architecture])
    model = GenericGRU() if architecture == 'gru' else CyclicFilter() if architecture == 'cyclic' else RawFilter()
    opt = torch.optim.Adam(model.parameters(), lr=LR_GRU if architecture == 'gru' else LR_CYCLIC)
    rng = np.random.default_rng(151100 + seed * 19)  # identical training examples across architectures
    first_loss = last_loss = None
    model.train()
    for step in range(steps):
        n = TRAIN_NS[step % len(TRAIN_NS)]
        fam, u, x = make_train_batch(n, BATCH, split, rng)
        logits = model(n, u, x)
        label = fam[:, None].float().expand_as(logits)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        if first_loss is None:
            first_loss = float(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        last_loss = float(loss.item())
    return model.eval(), {'train_steps':steps,'batch':BATCH,'first_loss':first_loss,
                          'last_loss':last_loss,'parameters':sum(p.numel() for p in model.parameters()),
                          'train_ns':list(TRAIN_NS),'train_phase_rule':split}


def balanced_panel(n: int, word: tuple[int, ...], marker: int = 0):
    fam = torch.tensor([f for f in (0, 1) for _ in range(n)],dtype=torch.int64)
    phase = torch.tensor(list(range(n))*2,dtype=torch.int64)
    u = torch.tensor([word]*(2*n),dtype=torch.int64)
    x = oracle_batch(n,fam,phase,u,marker=marker)
    return fam,phase,u,x


def score_panel(model: nn.Module, n: int, word: tuple[int,...],
                failures: tuple[int,...] = (), erasures: tuple[int,...] = ()) -> dict:
    fam,phase,u,x = balanced_panel(n,word)
    if failures or erasures:
        x=x.clone()
        for p in failures:
            x[:,p] ^= 1
        for p in erasures:
            x[:,p] = -1
    with torch.no_grad():
        prob = torch.sigmoid(model(n,u,x)[:,-1]).numpy()
    label=fam.numpy().astype(int)
    predicted=(prob>0.5).astype(int)
    return {'accuracy':float(np.mean(predicted==label)),
            'brier':float(np.mean((prob-label)**2)),
            'mean_probability':float(np.mean(prob)),
            'passive_deviation':float(np.mean(np.abs(prob-0.5))),
            'predicted_probabilities':[round(float(v),8) for v in prob],
            'correct_by_family':{
                'fixed':float(np.mean(predicted[:n]==0)),
                'rotating':float(np.mean(predicted[n:]==1))}}


def exact_posterior(n: int, u: tuple[int,...], observed: tuple[int,...],
                    raw_flip_prob: float = 0.0) -> float:
    """Exact finite Bayesian model posterior, with the optional known raw-bit noise law."""
    uu=torch.tensor([u],dtype=torch.int64)
    scores=[]
    for f in (0,1):
        count=0.0
        for z in range(n):
            x=oracle_batch(n,torch.tensor([f]),torch.tensor([z]),uu)[0].tolist()
            observed_positions=[i for i,v in enumerate(observed) if v!=-1]
            dist=sum(x[i]!=observed[i] for i in observed_positions)
            if raw_flip_prob==0:
                likelihood=float(dist==0)
            else:
                likelihood=(raw_flip_prob**dist)*((1-raw_flip_prob)**(len(observed_positions)-dist))
            count+=likelihood
        scores.append(count)
    den=sum(scores)
    return .5 if den==0 else scores[1]/den


def exact_min_distance_classifier(n:int, word:tuple[int,...], x:tuple[int,...]) -> int:
    """0 fixed, 1 rotating, -1 distance tie. Works with marked erasures."""
    u=torch.tensor([word],dtype=torch.int64)
    distances=[]
    for f in (0,1):
        d=min(sum(v!=-1 and v!=p for v,p in zip(x,oracle_batch(n,torch.tensor([f]),torch.tensor([z]),u)[0].tolist())) for z in range(n))
        distances.append(d)
    return -1 if distances[0]==distances[1] else int(distances[1]<distances[0])


def evaluate(model:nn.Module,split:str,architecture:str,seed:int) -> dict:
    panels={}
    for n in TRAIN_NS+OOD_NS:
        panels[str(n)]={}
        for schedule in ('passive','full_active','latch','extended_latch','alternating','random_budget'):
            word=actions_for(n,schedule)
            panels[str(n)][schedule]=score_panel(model,n,word)
    return {'architecture':architecture,'split':split,'seed':seed,'panels':panels,
            'cyclic_beta':float(F.softplus(model.log_beta).item()) if architecture in ('cyclic', 'raw') else None}


def validate_control() -> dict:
    tests={}
    for n in TRAIN_NS+OOD_NS:
        passive=actions_for(n,'passive')
        active=actions_for(n,'full_active')
        latch=actions_for(n,'latch')
        # All initial phases give same passive output for both model families.
        _,_,u,x=balanced_panel(n,passive)
        assert torch.equal(x[:n],x[n:])
        # Exactly n full probes separate every family trace with no phase information.
        _,_,u,x=balanced_panel(n,active)
        code_a={tuple(row.tolist()) for row in x[:n]}
        code_b={tuple(row.tolist()) for row in x[n:]}
        assert code_a.isdisjoint(code_b)
        # A fixed nonzero phase can never be distinguished from another fixed nonzero phase.
        a=tuple(x[1].tolist())
        _,_,u,x=balanced_panel(n,latch)
        assert len({tuple(row.tolist()) for row in x[1:n]})==1
        tests[str(n)]={'passive_family_identical':True,'n_probe_zero_error':True,
                       'fixed_nonzero_phases_indistinguishable':True,
                       'fixed_remaining_phase_entropy_bits':math.log2(n-1)}
    return tests


def run(seeds:tuple[int,...],splits:tuple[str,...],architectures:tuple[str,...],steps:int) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    begin=time.perf_counter()
    checks=validate_control()
    runs=[]
    for split in splits:
        for architecture in architectures:
            for seed in seeds:
                model,training=train_one(architecture,split,seed,steps)
                record=evaluate(model,split,architecture,seed)
                record['training']=training
                if architecture=='gru':
                    record['weights']={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}
                else:
                    record['weights']={'log_beta':float(model.log_beta.item())}
                runs.append(record)
                summ=[(n, round(record['panels'][str(n)]['full_active']['accuracy'],3),
                       round(record['panels'][str(n)]['passive']['brier'],3)) for n in TRAIN_NS+OOD_NS]
                print('run',architecture,split,seed,'beta',record['cyclic_beta'],'[n, full_active accuracy, passive Brier]',summ,flush=True)
    return {'protocol':{'name':'OIC-15-CPU-01','seeds':list(seeds),'splits':list(splits),
              'architectures':list(architectures),'train_ns':list(TRAIN_NS),
              'ood_ns':list(OOD_NS),'steps':steps,'batch':BATCH,'gru_width':HIDDEN,
              'lr_gru':LR_GRU,'lr_cyclic':LR_CYCLIC,'torch':str(torch.__version__),
              'numpy':str(np.__version__),'python':platform.python_version(),
              'observer_receives_phase':False,'observer_receives_family_label_at_inference':False,
              'observer_receives_n':True,'cyclic_model_knows_two_hypothesis_transition_rules':True,
              'no_pretrained_model':True},'controls':checks,'runs':runs,
              'wall_seconds':time.perf_counter()-begin}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    p.add_argument('--splits',nargs='+',choices=SPLITS,default=list(SPLITS))
    p.add_argument('--architectures',nargs='+',choices=['gru','cyclic','raw'],default=['gru','cyclic','raw'])
    p.add_argument('--steps',type=int,default=STEPS)
    args=p.parse_args()
    data=run(tuple(args.seeds),tuple(args.splits),tuple(args.architectures),args.steps)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=1,sort_keys=True)+'\n',encoding='utf-8')
    print('saved',args.out,'seconds',round(data['wall_seconds'],3))

if __name__=='__main__':
    main()
