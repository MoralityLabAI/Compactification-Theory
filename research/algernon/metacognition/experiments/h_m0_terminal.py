"""Algernon H-M0 v2: learn a budgeted experiment policy from terminal decision utility.

Sandbox-only synthetic CPU study. Outcomes follow independent Bernoulli likelihoods
conditional on one of four latent hypotheses. All selectors see the true likelihood
matrix; only probe choice is learned. The dynamic-programming expert optimizes the
expected terminal probability of correct MAP classification less cost penalty.
There is no Qwen or active GPU computation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.h_m0_pruner import (
    N_H, N_PROBES, BUDGET, COST_OPTIONS, Episode, entropy,
    information_gain, make_episode, posterior_update, run_episode as old_run,
)

COST_PENALTY = 0.004
STOP = N_PROBES


def legal_actions(ep: Episode, mask: int, spent: float) -> list[int]:
    return [i for i in range(N_PROBES) if not (mask >> i) & 1
            and spent + float(ep.costs[i]) <= BUDGET + 1e-9]


def posterior_from_outcomes(ep: Episode, mask: int, outcomes: int) -> np.ndarray:
    p = ep.prior.copy()
    for i in range(N_PROBES):
        if (mask >> i) & 1:
            p = posterior_update(p, ep.likelihoods[i], (outcomes >> i) & 1)
    return p


class TerminalPlanner:
    """Exact finite-horizon expectation over future probe observations.

    Q is Bayes-optimal within this *known simulator* under capped experiment costs.
    It is a mathematical comparison point, not a metacognitive neural learner.
    """
    def __init__(self, ep: Episode, penalty: float = COST_PENALTY):
        self.ep = ep
        self.penalty = penalty
        self.cost_by_mask = np.zeros(1 << N_PROBES, dtype=np.float64)
        for mask in range(1, 1 << N_PROBES):
            bit = mask & -mask
            i = bit.bit_length() - 1
            self.cost_by_mask[mask] = self.cost_by_mask[mask ^ bit] + float(ep.costs[i])
        self.calls = 0

    @lru_cache(maxsize=None)
    def evaluate(self, mask: int = 0, outcomes: int = 0) -> tuple[float, int]:
        self.calls += 1
        p = posterior_from_outcomes(self.ep, mask, outcomes)
        cost = self.cost_by_mask[mask]
        best_score = float(np.max(p) - self.penalty * cost)
        best_action = STOP
        # Legal to halt at any point, so both accuracy and probe costs matter.
        for i in legal_actions(self.ep, mask, cost):
            pi = float(np.dot(p, self.ep.likelihoods[i]))
            post_mask = mask | (1 << i)
            v_pos, _ = self.evaluate(post_mask, outcomes | (1 << i))
            v_neg, _ = self.evaluate(post_mask, outcomes)
            value = pi * v_pos + (1 - pi) * v_neg
            if value > best_score + 1e-10:
                best_score, best_action = value, i
        return best_score, best_action

    def action_scores(self, mask: int, outcomes: int) -> np.ndarray:
        p = posterior_from_outcomes(self.ep, mask, outcomes)
        cost = self.cost_by_mask[mask]
        scores = np.full(N_PROBES + 1, -1e9, dtype=np.float64)
        scores[STOP] = float(p.max() - self.penalty * cost)
        for i in legal_actions(self.ep, mask, cost):
            pi = float(np.dot(p, self.ep.likelihoods[i]))
            vm, _ = self.evaluate(mask | 1 << i, outcomes | 1 << i)
            vn, _ = self.evaluate(mask | 1 << i, outcomes)
            scores[i] = pi * vm + (1 - pi) * vn
        return scores


def inputs_for_state(ep: Episode, mask: int, outcomes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    p = posterior_from_outcomes(ep, mask, outcomes)
    spent = float(sum(float(ep.costs[i]) for i in range(N_PROBES) if (mask >> i) & 1))
    remaining = BUDGET - spent
    legal = legal_actions(ep, mask, spent)
    available = np.zeros(N_PROBES, dtype=np.float32)
    available[legal] = 1.0
    # Each probe row has the same semantics. Pooling preserves permutation equivariance.
    rows = []
    for i in range(N_PROBES):
        L = ep.likelihoods[i]
        rows.append(np.concatenate((p, L,
            [float(ep.costs[i]) / 3.0, remaining / BUDGET,
             float(available[i]), information_gain(p, L) / np.log(N_H),
             float(L.max() - L.min()), p.max()])) )
    return np.stack(rows).astype(np.float32), available, p, remaining


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(14, 48), nn.Tanh(), nn.Linear(48, 32), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(32*2 + 6, 48), nn.Tanh(), nn.Linear(48, 1))
        self.stop = nn.Sequential(nn.Linear(32 + 6, 24), nn.Tanh(), nn.Linear(24, 1))

    def forward(self, rows: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        # rows [B, 8, 14], available [B, 8]
        emb = self.embed(rows)
        pooled = (emb * available[...,None]).sum(1) / available.sum(1).clamp_min(1)[:,None]
        # global posterior and remaining budget are identical in every row
        glob = torch.cat((rows[:,0,:4], rows[:,0,9:10], rows[:,0,13:14]), -1)
        context = torch.cat((pooled, glob), -1)
        expanded_context = context[:,None,:].expand(-1,N_PROBES,-1)
        scores = self.score(torch.cat((emb,expanded_context),-1)).squeeze(-1)
        scores = scores.masked_fill(available < 0.5, -1e6)
        stop = self.stop(context).squeeze(-1)
        return torch.cat((scores,stop[:,None]),1)


def collect_training_worlds(start: int, n: int, histories: int, seed: int):
    """Expert labels from terminal optimal utility, for varied true observation histories."""
    rng = np.random.default_rng(seed)
    rows, avails, labels = [], [], []
    diagnostics = {'expert_states':0,'teacher_stop':0,'expert_nodes':0}
    for j in range(n):
        ep = make_episode(start+j)
        teacher = TerminalPlanner(ep)
        for rollout in range(histories):
            mask, outcomes, spent = 0, 0, 0.0
            # Mix on-policy teacher paths with random histories for covariate coverage.
            for step in range(5):
                feat, av, p, rem = inputs_for_state(ep,mask,outcomes)
                decision = teacher.evaluate(mask,outcomes)[1]
                rows.append(feat); avails.append(av); labels.append(decision)
                diagnostics['expert_states'] += 1
                diagnostics['teacher_stop'] += int(decision == STOP)
                legal=legal_actions(ep,mask,spent)
                if not legal: break
                if rollout == 0 or rng.random()<0.65:
                    action = decision
                else:
                    action = int(rng.choice(legal))
                if action == STOP: break
                outcomes |= int(ep.observations[action]) << action
                mask |= 1 << action
                spent += float(ep.costs[action])
        diagnostics['expert_nodes'] += teacher.calls
    return np.asarray(rows,np.float32),np.asarray(avails,np.float32),np.asarray(labels,np.int64), diagnostics


def fit_policy(rows,available,labels,seed,epochs=16,batch_size=512,holdout=None):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = Selector()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0015, weight_decay=1e-5)
    r = torch.from_numpy(rows); m=torch.from_numpy(available); y=torch.from_numpy(labels)
    history=[]
    rng=np.random.default_rng(seed+11)
    best = None; best_loss = float('inf'); best_epoch=None
    for epoch in range(epochs):
        model.train()
        for inds in np.array_split(rng.permutation(len(y)),max(1,int(np.ceil(len(y)/batch_size)))):
            logits=model(r[inds],m[inds]); loss=F.cross_entropy(logits,y[inds]);
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            train_ce = float(F.cross_entropy(model(r,m),y))
            train_accuracy = float((model(r,m).argmax(1)==y).float().mean())
            if holdout:
                vr,vm,vy=holdout
                valid_ce=float(F.cross_entropy(model(torch.from_numpy(vr),torch.from_numpy(vm)),torch.from_numpy(vy)))
                metric=valid_ce
            else:
                valid_ce=None; metric=train_ce
        history.append({'epoch':epoch+1,'train_CE':train_ce,'train_action_accuracy':train_accuracy,'validation_CE':valid_ce})
        if metric<best_loss:
            best_loss=metric; best_epoch=epoch+1
            best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best);model.eval()
    return model, {'epochs':epochs,'selected_epoch':best_epoch,'history':history,'train_state_count':len(y),'parameters':sum(p.numel() for p in model.parameters())}


def act(arm: str, ep: Episode, p: np.ndarray, mask: int, outcomes:int, spent:float, model:Selector|None, old_model:nn.Module|None, rng, teacher:TerminalPlanner|None, untrained_model:Selector|None):
    legal = legal_actions(ep,mask,spent)
    if not legal:return STOP
    if arm == 'exact_dp_oracle':
        return teacher.evaluate(mask,outcomes)[1]
    if arm == 'random': return int(rng.choice(legal))
    if arm == 'cheapest': return min(legal,key=lambda i:(ep.costs[i],i))
    if arm == 'fixed_heuristic':
        return max(legal,key=lambda i:(float(np.var(ep.likelihoods[i])/ep.costs[i]),-i))
    if arm == 'bayes_ig':
        return max(legal,key=lambda i:(information_gain(p,ep.likelihoods[i])/ep.costs[i],-i))
    if arm == 'old_greedy_mlp':
        from experiments.h_m0_pruner import feature
        x = np.stack([feature(p,ep.likelihoods[i],float(ep.costs[i]), BUDGET-spent) for i in legal])
        with torch.no_grad():
            scores = old_model(torch.from_numpy(x)).squeeze(-1).numpy()
        return legal[int(scores.argmax())]
    if arm in ['terminal_policy','terminal_policy_shuffled_feedback','untrained_terminal']:
        feats,available,_,_=inputs_for_state(ep,mask,outcomes)
        if arm=='terminal_policy_shuffled_feedback':
            # Rotate posterior and re-compute context features for an intervention.
            shuf = np.roll(p,1)
            feats[:,:4] = shuf
            feats[:,13] = shuf.max()
            for i in range(N_PROBES):
                feats[i,11] = information_gain(shuf,ep.likelihoods[i]) / np.log(N_H)
        predictor=model if arm!='untrained_terminal' else untrained_model
        with torch.no_grad():
            logits=predictor(torch.from_numpy(feats[None,:]),torch.from_numpy(available[None,:]))[0].numpy()
        return int(np.argmax(logits))
    raise ValueError(arm)


def run(arm: str,ep:Episode,model:Selector|None,old_model:nn.Module|None=None,teacher:TerminalPlanner|None=None,untrained_model:Selector|None=None) -> dict:
    rng = np.random.default_rng(10000000+ep.seed)
    mask=0;outcomes=0;spent=0.0;probes=[];decision_count=0
    while len(probes)<N_PROBES:
        p=posterior_from_outcomes(ep,mask,outcomes)
        if arm not in ('exact_dp_oracle','terminal_policy','terminal_policy_shuffled_feedback','untrained_terminal') and p.max()>=0.98:
            break # legacy baseline stop rule
        decision=act(arm,ep,p,mask,outcomes,spent,model,old_model,rng,teacher,untrained_model)
        decision_count += 1
        if decision==STOP:break
        if decision not in legal_actions(ep,mask,spent):raise RuntimeError('policy violates budget or repeats probe')
        probes.append(decision)
        spent += float(ep.costs[decision]); mask|=1<<decision; outcomes |= int(ep.observations[decision])<<decision
    p=posterior_from_outcomes(ep,mask,outcomes)
    return {'seed':int(ep.seed),'arm':arm,'truth':ep.truth,'predicted':int(p.argmax()),'correct':int(p.argmax()==ep.truth),
            'expected_correct':float(p.max()),'terminal_utility':float(p.max()-COST_PENALTY*spent),
            'brier':float(np.sum((p-np.eye(N_H)[ep.truth])**2)),'log_loss':float(-np.log(max(1e-12,p[ep.truth]))),
            'cost':spent,'probes':probes,'decisions':decision_count,'posterior':np.round(p,8).tolist()}


def paired_interval(records,a,b,key='correct',reps=2000,seed=8332):
    by_a={v['seed']:v[key] for v in records if v['arm']==a}
    by_b={v['seed']:v[key] for v in records if v['arm']==b}
    common=sorted(set(by_a)&set(by_b))
    diffs=np.asarray([by_a[s]-by_b[s] for s in common]);rng=np.random.default_rng(seed)
    draws=np.mean(diffs[rng.integers(0,len(diffs),size=(reps,len(diffs)))],axis=1)
    return {'difference':float(diffs.mean()),'paired_95pct_CI':np.quantile(draws,[0.025,0.975]).tolist(),'n':len(diffs)}


def evaluate(episodes:list[Episode],model,old_model,arms,untrained_model=None):
    records=[]
    for ep in episodes:
        teacher = TerminalPlanner(ep) if 'exact_dp_oracle' in arms else None
        for arm in arms:
            records.append(run(arm,ep,model,old_model,teacher,untrained_model))
    return records


def results_summary(records,arms):
    means={arm:{k:float(np.mean([r[k] for r in records if r['arm']==arm])) for k in
             ['correct','expected_correct','terminal_utility','brier','log_loss','cost','decisions']}
           for arm in arms}
    comparison={arm:{key:paired_interval(records,'terminal_policy',arm,key) for key in ('correct','terminal_utility')}
                for arm in arms if arm!='terminal_policy'}
    return means,comparison


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-worlds',type=int,default=700)
    parser.add_argument('--validation-worlds',type=int,default=150)
    parser.add_argument('--test-worlds',type=int,default=1000)
    parser.add_argument('--rollouts',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=18)
    parser.add_argument('--seed',type=int,default=20261104)
    parser.add_argument('--oracle-test-worlds',type=int,default=100)
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    train_start=100000000;val_start=350000000;test_start=400000000
    train_x,train_m,train_y,diag=collect_training_worlds(train_start,args.train_worlds,args.rollouts,args.seed)
    val_x,val_m,val_y,vdiag=collect_training_worlds(val_start,args.validation_worlds,1,args.seed+99)
    model,train_detail=fit_policy(train_x,train_m,train_y,args.seed,args.epochs,holdout=(val_x,val_m,val_y))
    # original H-M0 weights are separate, never trained on the new final test range.
    old_model=nn.Sequential(nn.Linear(12,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1));old_model.load_state_dict(torch.load(Path(__file__).resolve().parents[1]/'results/h_m0_01/policy.pt',map_location='cpu',weights_only=True));old_model.eval()
    torch.manual_seed(args.seed+123)
    untrained=Selector();untrained.eval()
    arms=['random','cheapest','fixed_heuristic','bayes_ig','old_greedy_mlp','untrained_terminal','terminal_policy_shuffled_feedback','terminal_policy']
    episodes=[make_episode(test_start+j) for j in range(args.test_worlds)]
    records=evaluate(episodes,model,old_model,arms,untrained)
    # Independent expert for a fixed subset of *fresh test* worlds, with no tuning from it.
    oracle_set=[make_episode(test_start+args.test_worlds+j) for j in range(args.oracle_test_worlds)]
    oracle_arms=['random','bayes_ig','terminal_policy','exact_dp_oracle']
    oracle_records=evaluate(oracle_set,model,old_model,oracle_arms,untrained)
    means,comparisons=results_summary(records,arms)
    oracle_means,oracle_comparisons=results_summary(oracle_records,oracle_arms)
    elapsed=time.perf_counter()-start
    metadata={'run_type':'exploratory_synthetic_cpu','experiment':'H-M0 terminal-utility learner',
        'seed':args.seed,'train_worlds':args.train_worlds,'validation_worlds':args.validation_worlds,
        'test_worlds':args.test_worlds,'oracle_test_worlds':args.oracle_test_worlds,
        'train_start':train_start,'validation_start':val_start,'test_start':test_start,
        'oracle_start':test_start+args.test_worlds,'rollouts':args.rollouts,'epochs':args.epochs,
        'budget':BUDGET,'penalty_per_cost_unit':COST_PENALTY,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'simulator_sha256':hashlib.sha256(Path(__file__).with_name('h_m0_pruner.py').read_bytes()).hexdigest(),
        'previous_model_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'results/h_m0_01/policy.pt').read_bytes()).hexdigest(),
        'numpy':np.__version__,'torch':torch.__version__,'python':platform.python_version(),'hardware':'CPU only',
        'selection_rule':'best validation expert-imitation cross entropy; final test never used to select epoch'}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    metrics={'status':'exploratory_synthetic_cpu','training':train_detail,'teacher_diagnostics':diag,'validation_teacher_diagnostics':vdiag,
       'means':means,'comparisons':comparisons,'oracle_subset_means':oracle_means,'oracle_subset_comparisons':oracle_comparisons,
       'elapsed_seconds':elapsed,'notes':[
           'Exact-Bayes belief update and known likelihoods are shared by all methods.',
           'Expert labels maximize expected terminal MAP posterior confidence with probe-cost penalty and a hard budget.',
           'Learned model trained by distillation on expert decisions; distinct from policy-gradient reinforcement learning.',
           'Original 300M seed test range remains excluded; final 400M test range evaluated once.',
           'Synthetic simulator and probabilistic expert, no Qwen/GPU model execution.',
           'Exact DP oracle has extra CPU compute and perfect simulator likelihood access; reference upper bound, not matched-cost policy.'
       ]}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for name,record_set in [('records.jsonl',records),('oracle_records.jsonl',oracle_records)]:
        with (out/name).open('w') as f:
            for rec in record_set:f.write(json.dumps(rec)+'\n')
    torch.save(model.state_dict(),out/'terminal_selector.pt')
    print(json.dumps({'elapsed':round(elapsed,2),'train_examples':len(train_y),
        'train_parameters':train_detail['parameters'],'selected_epoch':train_detail['selected_epoch'],
        'accuracy':{a:round(means[a]['correct'],4) for a in arms},
        'utility':{a:round(means[a]['terminal_utility'],4) for a in arms},
        'oracle_subset_accuracy':{a:round(oracle_means[a]['correct'],4) for a in oracle_arms},
        'vs_ig':comparisons['bayes_ig']['correct']},indent=2))


if __name__=='__main__':main()
