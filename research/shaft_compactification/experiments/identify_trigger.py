"""SHAFT-COMP-24: estimate unknown temporal trigger law from paired behaviors.

Preregistered protocol SHAFT_COMP_24_PROTOCOL.md. The 36 binary trigger entries
are estimated solely from native B1 / B2 query responses and last *observed*
input token. No oracle goal/latch labels are used to fit the rule.
Local, frozen synthetic GRUs only; no external network or device operations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from research.shaft_compactification.experiments import architecture_vocab_transfer as base
from research.shaft_compactification.trigger_identifiability import true_trigger, FAMILIES, V

SEEDS={'repeat_toggle':{'calibration':(2301,),'heldout':(2302,2303)},
       'successor_toggle':{'calibration':(2311,),'heldout':(2312,2313)}}
DEPTHS=(12,40)
PER_GOAL=16
CALIBRATION_MARK=24000000
HELDOUT_MARK=25000000
PROTOCOL='SHAFT_COMP_24_PROTOCOL.md'
MANIFEST='MANIFEST.json'

def digest_bytes(data):return hashlib.sha256(data).hexdigest()
def file_sha(p):return digest_bytes(Path(p).read_bytes())

def paths_for(family,seed,phase):
    if seed not in SEEDS[family][phase]:raise ValueError('seed/phase mismatch')
    family_tag='repeat' if family=='repeat_toggle' else 'successor'
    if phase=='calibration':
        return (f'checkpoints/cal_{family_tag}_{seed}.pt',
                f'original/cal_{family_tag}_{seed}.json')
    return (f'checkpoints/held_{family}_{seed}.pt',
            f'original/held_{family}_{seed}.json')

def load_frozen(root,family,seed,phase):
    root=Path(root)
    chk,meta=paths_for(family,seed,phase)
    manifest=json.loads((root/MANIFEST).read_text())
    checkpoint=root/chk
    if file_sha(checkpoint)!=manifest['files'][chk]['sha256']:
        raise ValueError('frozen checkpoint hash mismatch')
    if file_sha(root/meta)!=manifest['files'][meta]['sha256']:
        raise ValueError('original COMP-23 receipt hash mismatch')
    receipt=json.loads((root/meta).read_text())
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if weights['seed']!=seed or weights['family']!=family:
        raise ValueError('checkpoint family/seed mismatch')
    model=base.RecurrentMemory('gru',V)
    model.load_state_dict(weights['base'])
    model.eval()
    stable=base.stable_sha(model)
    if stable!=receipt['base']['model_sha256']:
        raise ValueError('frozen model weights do not match original model digest')
    if not receipt['base']['gate_passed']:
        raise ValueError('base model failed original gate')
    return model, {'checkpoint_sha256':file_sha(checkpoint),'base_model_sha256':stable,
                   'original_receipt_sha256':file_sha(root/meta),
                   'native_accuracy_d12':receipt['base']['task_accuracy_d12'],
                   'native_accuracy_d32':receipt['base']['task_accuracy_d32']}

@torch.no_grad()
def sample(model,seed,depth,phase,per_goal=PER_GOAL):
    if phase not in ('calibration','heldout'):raise ValueError(phase)
    source=(CALIBRATION_MARK if phase=='calibration' else HELDOUT_MARK)+seed*101+depth*131
    goals=torch.arange(1<<V).repeat_interleave(per_goal)
    rng=torch.Generator().manual_seed(source)
    goals=goals[torch.randperm(len(goals),generator=rng)]
    context_gen=torch.Generator().manual_seed(seed*100000+depth*17+(24 if phase=='calibration' else 37))
    tokens=torch.randint(V,(len(goals),depth),generator=context_gen)
    h=model.embedding(goals)
    for t in range(depth):
        h=model.step(h,F.one_hot(tokens[:,t],V).float())
    p=tokens[:,-1].cpu().numpy().astype(np.int64)
    first=(base.counterfactual(model,h,1)>=0).cpu().numpy().astype(np.uint8)
    second=(base.counterfactual(model,h,2)>=0).cpu().numpy().astype(np.uint8).reshape(-1,V,V)
    return {'first':first,'second':second,'previous':p,'goals':goals.cpu().numpy(),
            'sample_seed':source,'input_seed':seed*100000+depth*17+(24 if phase=='calibration' else 37)}

def infer_rule(data_list):
    votes=np.zeros((V,V),dtype=np.int64)
    counts=np.zeros((V,V),dtype=np.int64)
    for obj in data_list:
        first,second,previous=obj['first'],obj['second'],obj['previous']
        assert first.shape==(len(previous),V) and second.shape==(len(previous),V,V)
        discrepancy=second ^ first[:,None,:]
        for p in range(V):
            mask=(previous==p)
            votes[p]+=discrepancy[mask].sum(axis=(0,2),dtype=np.int64)
            counts[p]+=int(mask.sum())*V
    assert np.all(counts>0)
    rule=(2*votes>counts).astype(np.uint8)
    return {'rule':rule.tolist(),'ones_votes':votes.tolist(),
            'total_votes':counts.tolist(),
            'vote_share':np.round(votes/counts,8).tolist(),
            'majority_vote_fraction_margin':np.round(np.abs(votes/counts-.5),8).tolist(),
            'exact_tie_entries':int((2*votes==counts).sum())}

def prediction(first,previous,rule):
    t=np.asarray(rule,dtype=np.uint8)
    if t.shape!=(V,V):raise ValueError('expected 6x6 trigger rule')
    return first[:,None,:]^t[previous,:,None]

def metrics(pred,truth):
    same=pred==truth
    return {'bit_accuracy':round(float(same.mean()),8),
            'exact_accuracy':round(float(same.all(axis=(1,2)).mean()),8),
            'bits_correct':int(same.sum()),'bits_total':int(same.size),
            'exact_correct':int(same.all(axis=(1,2)).sum()),
            'states':int(len(pred))}

def configure():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)

def calibration(root,per_goal=PER_GOAL):
    configure()
    output={'experiment':'SHAFT-COMP-24','phase':'calibration','protocol':PROTOCOL,
            'torch':str(torch.__version__),'numpy':str(np.__version__),
            'calibration_depths':list(DEPTHS),'calibration_states_per_family':64*per_goal*len(DEPTHS),
            'per_goal':per_goal,'families':{}}
    for family in FAMILIES:
        seed=SEEDS[family]['calibration'][0]
        model,origin=load_frozen(root,family,seed,'calibration')
        data=[sample(model,seed,depth,'calibration',per_goal) for depth in DEPTHS]
        learned=infer_rule(data)
        output['families'][family]={'source_seed':seed,'source_model':origin,
           'depths':[{ 'depth':d,'states':len(data[i]['previous']),
                       'previous_histogram':np.bincount(data[i]['previous'],minlength=V).tolist(),
                       'sample_seed':data[i]['sample_seed'],'input_seed':data[i]['input_seed']}
                     for i,d in enumerate(DEPTHS)],
           'inference':learned}
    return output

def bootstrap_difference(rows,n_draw=1000):
    # Registered depth-40 comparison; resample states within each fixed model.
    filtered=[r for r in rows if r['depth']==40]
    grouped={}
    for r in filtered:grouped.setdefault(r['seed'],[]).append(r)
    rng=np.random.default_rng(240024)
    diff=np.mean([(r['learned_bits_correct']-r['zero_bits_correct'])/(V*V) for r in filtered])
    samples=[]
    for _ in range(n_draw):
        totals=0;count=0
        for group in grouped.values():
            idx=rng.integers(0,len(group),size=len(group))
            totals+=sum(group[i]['learned_bits_correct']-group[i]['zero_bits_correct'] for i in idx)
            count+=len(idx)*V*V
        samples.append(totals/count)
    return {'primary_learned_minus_zero_bit_accuracy':round(float(diff),8),
            'bootstrap95':[round(float(x),8) for x in np.quantile(samples,[.025,.975])],
            'bootstrap_draws':n_draw,
            'model_differences':{str(seed):round(float(np.mean([
                 (r['learned_bits_correct']-r['zero_bits_correct'])/(V*V) for r in group])),8)
                  for seed,group in grouped.items()}}

def evaluate(root,freeze,per_goal=PER_GOAL):
    configure()
    assert freeze['phase']=='calibration' and freeze['experiment']=='SHAFT-COMP-24'
    if freeze['per_goal']!=per_goal:raise ValueError('calibration design differs')
    output={'experiment':'SHAFT-COMP-24','phase':'heldout','protocol':PROTOCOL,
            'torch':str(torch.__version__),'numpy':str(np.__version__),
            'calibration_sha256':freeze['freeze_sha256'],
            'per_goal':per_goal,'cells':[],'rows':[], 'rules':{}}
    all_rows=[]
    for family in FAMILIES:
        inferred=freeze['families'][family]['inference']['rule']
        other=freeze['families'][next(f for f in FAMILIES if f!=family)]['inference']['rule']
        truth_rule=true_trigger(family) # post-freeze audit only
        agrees=sum(inferred[p][a]==truth_rule[p][a] for p in range(V) for a in range(V))
        output['rules'][family]={'inferred':inferred,'oracle_entry_agreement':agrees,
            'oracle_entries':V*V,'oracle_for_posthoc_audit':[list(row) for row in truth_rule]}
        for seed in SEEDS[family]['heldout']:
            model,origin=load_frozen(root,family,seed,'heldout')
            for depth in DEPTHS:
                obj=sample(model,seed,depth,'heldout',per_goal)
                first,second,p=obj['first'],obj['second'],obj['previous']
                compare={
                    'learned':prediction(first,p,inferred),
                    'zero':prediction(first,p,np.zeros((V,V),dtype=np.uint8)),
                    'wrong_family':prediction(first,p,other),
                    'permuted_previous':prediction(first,(p+1)%V,inferred),
                    'oracle':prediction(first,p,truth_rule),
                }
                met={name:metrics(pred,second) for name,pred in compare.items()}
                output['cells'].append({'family':family,'seed':seed,'depth':depth,'source_model':origin,
                  'states':len(p),'previous_histogram':np.bincount(p,minlength=V).tolist(),
                  'source_seeds':{'shuffle':obj['sample_seed'],'tokens':obj['input_seed']},
                  'metrics':met})
                for i in range(len(p)):
                    row={'family':family,'seed':seed,'depth':depth,'index':i,
                         'observed_previous_token':int(p[i]),
                         'goal_for_generation_audit_only':int(obj['goals'][i])}
                    for name,pred in compare.items():
                        m=(pred[i]==second[i])
                        row[f'{name}_bits_correct']=int(m.sum())
                        row[f'{name}_exact']=int(m.all())
                    all_rows.append(row)
    output['rows']=all_rows
    output['primary']=bootstrap_difference(all_rows)
    output['registered_support']={
        'rules_exact_both':all(output['rules'][name]['oracle_entry_agreement']==36 for name in FAMILIES),
        'positive_bootstrap_lower_bound':output['primary']['bootstrap95'][0]>0,
        'four_model_improvements':all(v>0 for v in output['primary']['model_differences'].values()),
    }
    output['registered_support']['supported']=all(output['registered_support'].values())
    return output

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=('calibrate','evaluate'))
    ap.add_argument('--archive-root',type=Path,required=True)
    ap.add_argument('--per-goal',type=int,default=PER_GOAL)
    ap.add_argument('--freeze',type=Path)
    ap.add_argument('--freeze-sha256')
    ap.add_argument('--out',type=Path,required=True)
    opts=ap.parse_args()
    if not 1<=opts.per_goal<=32:raise ValueError('per-goal must lie in 1..32')
    if opts.phase=='calibrate':data=calibration(opts.archive_root,opts.per_goal)
    else:
        if not opts.freeze or not opts.freeze_sha256:raise ValueError('heldout requires frozen receipt hash')
        frozen_bytes=opts.freeze.read_bytes()
        if digest_bytes(frozen_bytes)!=opts.freeze_sha256:raise ValueError('frozen calibration SHA mismatch')
        data=json.loads(frozen_bytes)
        data['freeze_sha256']=opts.freeze_sha256
        data=evaluate(opts.archive_root,data,opts.per_goal)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'phase':opts.phase,'out':str(opts.out),
                     'families':list(data.get('families',data.get('rules',{}))),
                     'primary':data.get('primary'),'support':data.get('registered_support')}),flush=True)

if __name__=='__main__':main()
