"""SHAFT-COMP-22: supervised temporal factorization versus direct behavioral probe.

Frozen prior COMP-21 24-unit GRUs; fresh probe training/test contexts. The
factorized heads use privileged automaton labels for previous input and latch,
so comparisons are descriptive and are NOT equal-supervision experiments.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from research.shaft_compactification.experiments import architecture_vocab_transfer as base
from research.shaft_compactification.experiments import persistent_goal_control as comp20
from research.shaft_compactification.factorized_latch_finite import reduced_two, canonical_two

FAMILIES={'repeat_toggle':(2202,2203),'successor_toggle':(2212,2213)}
DEPTHS=(12,40)
PROBE_DEPTHS=(4,8,12)
V=6
GOALS=64
TRAIN_PER_GOAL=12
EVAL_PER_GOAL=32
TRAIN_STEPS=400
PROBE_LR=.04
SEED_MARK=22000000


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load_frozen(root,family,seed):
    root=Path(root)
    path=root/f'held_{family}_{seed}_aux_state.pt'
    receipt=json.loads((root/f'held_{family}_{seed}_aux_state.json').read_text())
    if not receipt['probe_gate_passed'] or receipt['accuracy_d12']['bit_accuracy']<.90:
        raise ValueError(f'base model {seed} did not pass original COMP-21 gate')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=base.RecurrentMemory('gru',V)
    model.load_state_dict(ck['base'])
    model.eval()
    if base.stable_sha(model)!=receipt['model_sha256']:
        raise ValueError('checkpoint does not match saved model digest')
    return model,receipt,path


def natural_states(model,family,seed,depth,per_goal,train):
    goals=torch.arange(GOALS).repeat_interleave(per_goal)
    source=SEED_MARK + seed*91 + depth*107 + (71 if train else 600007)
    generator=torch.Generator().manual_seed(source)
    goals=goals[torch.randperm(len(goals),generator=generator)]
    state,latch,previous=comp20.prefixes(
        model,goals,depth,source+2200 if train else source+909001,family)
    return state.detach(),goals.detach(),torch.from_numpy(latch.copy()).long(),torch.from_numpy(previous.copy()).long()


def bits_from_goal(goals):
    return ((goals[:,None]>>torch.arange(V))&1).float()


def fit_head(X,y,seed,kind):
    torch.manual_seed(seed)
    head=nn.Linear(X.shape[1],y.shape[1] if kind=='bce' else V)
    opt=torch.optim.Adam(head.parameters(),lr=PROBE_LR)
    for _ in range(TRAIN_STEPS):
        out=head(X)
        loss=(F.binary_cross_entropy_with_logits(out,y)
              if kind=='bce' else F.cross_entropy(out,y[:,0].long()))
        opt.zero_grad();loss.backward();opt.step()
    head.eval()
    return head,round(float(loss.detach()),7)


def train_factor_heads(model,family,seed):
    Xs=[];gs=[];ls=[];ps=[]
    for depth in PROBE_DEPTHS:
        X,g,l,p=natural_states(model,family,seed,depth,TRAIN_PER_GOAL,True)
        Xs.append(X);gs.append(g);ls.append(l);ps.append(p)
    X=torch.cat(Xs);g=torch.cat(gs);l=torch.cat(ls);p=torch.cat(ps)
    with torch.no_grad():
        native_one=(base.counterfactual(model,X,1)>=0).float()
        native_two=(base.counterfactual(model,X,2)>=0).float()
        ground_goal=bits_from_goal(g)
    labels={'direct':(native_two,'bce'),'goal':(ground_goal,'bce'),
            'latch':(l[:,None].float(),'bce'),
            'previous':(p[:,None],'ce'),'effective':(native_one,'bce')}
    result={};losses={}
    for i,(name,(y,kind)) in enumerate(labels.items()):
        head,loss=fit_head(X,y,SEED_MARK + seed*19 + (i+1)*7001,kind)
        result[name]=head;losses[name]=loss
    return result,losses,len(X)


def trigger_tensor(previous,family):
    first=torch.arange(V).unsqueeze(0)
    if family=='repeat_toggle':return first==previous[:,None]
    if family=='successor_toggle':return first==((previous[:,None]+1)%V)
    raise ValueError(family)


def predicted_two(bits,previous,family,latch=None):
    bits=bits.bool()
    gate=trigger_tensor(previous,family)
    if latch is None:latch=torch.zeros(len(bits),dtype=torch.bool)
    return (bits[:,None,:] ^ gate[:,:,None] ^ latch[:,None,None].bool()).reshape(len(bits),V*V)


def checksum_predictions(pred):
    arr=pred.cpu().to(torch.uint8).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


@torch.inference_mode()
def evaluate_model(model,heads,family,seed,depth,per_goal):
    X,g,l,p=natural_states(model,family,seed,depth,per_goal,False)
    native_two=(base.counterfactual(model,X,2)>=0)
    true_goal=bits_from_goal(g).bool()
    exact_two=predicted_two(true_goal,p,family,latch=l)
    pred_goal=heads['goal'](X)>=0
    pred_latch=(heads['latch'](X)>=0).flatten()
    pred_prev=heads['previous'](X).argmax(dim=1)
    pred_effective=heads['effective'](X)>=0
    preds={
        'direct':heads['direct'](X)>=0,
        'canonical':predicted_two(pred_goal,pred_prev,family,pred_latch),
        'behavioral':predicted_two(pred_effective,pred_prev,family),
        'oracle':exact_two,
    }
    ground_effective=true_goal ^ l[:,None].bool()
    # Source of teacher mismatch assessed directly; no labels used in predictions.
    diagnostics={
        'ground_goal_exact':(pred_goal==true_goal).all(1),
        'ground_goal_bit':(pred_goal==true_goal).float().mean(1),
        'latch_accuracy':pred_latch==l.bool(),
        'previous_accuracy':pred_prev==p,
        'effective_exact_vs_canonical':(pred_effective==ground_effective).all(1),
        'effective_bit_vs_canonical':(pred_effective==ground_effective).float().mean(1),
        'effective_exact_vs_native':(pred_effective==(base.counterfactual(model,X,1)>=0)).all(1),
    }
    by_method={}
    for key,pred in preds.items():
        match_native=(pred==native_two)
        match_automaton=(pred==exact_two)
        by_method[key]={
            'native_bit_accuracy':round(float(match_native.float().mean()),6),
            'native_full_accuracy':round(float(match_native.all(1).float().mean()),6),
            'automaton_bit_accuracy':round(float(match_automaton.float().mean()),6),
            'automaton_full_accuracy':round(float(match_automaton.all(1).float().mean()),6),
            'predictions_sha256':checksum_predictions(pred),
        }
    metrics={key:round(float(x.float().mean()),6) for key,x in diagnostics.items()}
    native_agreement=((native_two==exact_two).all(1)).float()
    metrics['native_vs_exact_automaton_full']=round(float(native_agreement.mean()),6)
    metrics['native_vs_exact_automaton_bit']=round(float((native_two==exact_two).float().mean()),6)
    records=[]
    for i in range(len(g)):
        rec={'family':family,'seed':seed,'depth':depth,'index':i,
             'goal_for_audit_only':int(g[i]),'true_latch_for_audit_only':int(l[i]),
             'true_prev_for_audit_only':int(p[i]),
             'native_oracle_full_match':int(native_agreement[i].item())}
        for name,pred in preds.items():
            rec[name+'_native_bits_correct']=int((pred[i]==native_two[i]).sum())
            rec[name+'_native_exact']=int((pred[i]==native_two[i]).all())
            rec[name+'_automaton_bits_correct']=int((pred[i]==exact_two[i]).sum())
            rec[name+'_automaton_exact']=int((pred[i]==exact_two[i]).all())
        for name,pred in diagnostics.items():rec[name]=float(pred[i])
        records.append(rec)
    return {'seed':seed,'family':family,'depth':depth,'states':len(g),
            'diagnostics':metrics,'methods':by_method},records


def bootstrapped_difference(rows,n_boot=1000):
    rows=[r for r in rows if r['depth']==40]
    groups={}
    for r in rows:groups.setdefault(r['seed'],[]).append(r)
    rng=np.random.default_rng(220022)
    empirical=(sum(r['canonical_native_exact']-r['direct_native_exact'] for r in rows)/len(rows))
    diffs=[]
    by_model={str(seed):round(sum(r['canonical_native_exact']-r['direct_native_exact'] for r in rr)/len(rr),6)
              for seed,rr in groups.items()}
    for _ in range(n_boot):
        sample=[]
        for seed,rr in groups.items():
            draws=rng.integers(len(rr),size=len(rr))
            sample.extend(rr[j]['canonical_native_exact']-rr[j]['direct_native_exact'] for j in draws)
        diffs.append(np.mean(sample))
    lo,hi=np.quantile(diffs,[.025,.975])
    return {'primary_canonical_minus_direct_exact_native_depth40':round(float(empirical),6),
            'bootstrap95':[round(float(lo),6),round(float(hi),6)],
            'bootstrap_draws':n_boot,'by_model':by_model,
            'positive_models':sum(v>0 for v in by_model.values()),
            'support_gate':bool(lo>0 and sum(v>0 for v in by_model.values())>=3)}


def run(frozen_root,per_goal=EVAL_PER_GOAL):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    allrows=[];outcomes=[];probes={};models={}
    for family,seeds in FAMILIES.items():
        for seed in seeds:
            model,prior,path=load_frozen(frozen_root,family,seed)
            heads,loss,train_count=train_factor_heads(model,family,seed)
            key=f'{family}_{seed}'
            probes[key]={'training_contexts':train_count,
                         'loss':loss,'parameters':{n:sum(x.numel() for x in h.parameters()) for n,h in heads.items()},
                         'hashes':{n:base.stable_sha(h) for n,h in heads.items()}}
            models[key]={'model_sha256':prior['model_sha256'],
                         'checkpoint_sha256':sha_file(path),
                         'prior_accuracy_d12':prior['accuracy_d12']['bit_accuracy'],
                         'prior_accuracy_d32':prior['accuracy_d32']['bit_accuracy']}
            for depth in DEPTHS:
                metrics,rows=evaluate_model(model,heads,family,seed,depth,per_goal)
                outcomes.append(metrics);allrows.extend(rows)
                print(json.dumps({'model':key,'depth':depth,
                     'native_full':{k:v['native_full_accuracy'] for k,v in metrics['methods'].items()},
                     'ground_goal_exact':metrics['diagnostics']['ground_goal_exact'],
                     'previous_accuracy':metrics['diagnostics']['previous_accuracy']}),flush=True)
    result={'experiment':'SHAFT-COMP-22','registration_commit':'f539c28e0dc4409cc76741a48c5d39ba9531f9c0',
            'torch':torch.__version__,'numpy':np.__version__,
            'frozen_base_model_seeds':{k:list(v) for k,v in FAMILIES.items()},
            'evaluation_contexts_per_goal_depth':per_goal,'probe_training_contexts_per_goal_depth':TRAIN_PER_GOAL,
            'training_steps_per_head':TRAIN_STEPS,'model_receipts':models,'head_receipts':probes,
            'cells':outcomes,'primary_inference':bootstrapped_difference(allrows),
            'rows':allrows}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frozen-root',type=Path,required=True)
    p.add_argument('--per-goal',type=int,default=EVAL_PER_GOAL)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if not 1<=a.per_goal<=32:raise ValueError('per-goal must be 1..32')
    data=run(a.frozen_root,a.per_goal)
    s=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(s)
    else:print(s,end='')
if __name__=='__main__':main()
