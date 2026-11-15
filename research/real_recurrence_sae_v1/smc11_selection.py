"""SMC-11: sharp exact-rational SAE identity radii on pretrained recurrence.

Uses source-locked 44K teacher/64-puzzle roster, training-only stage-one SAE
with the unchanged SMC-07–10 recipe (seeds 71,89,113, 300 updates).

Each physical goal-token cell radius is an EXACT REAL claim about the affine
encoder defined by stored binary FP32 W,b,train mu/std,source h.
Only the selected top-k identities are certified, not FP32 rounded execution.

With --smc10-receipt, compose these guards with the PREVIOUSLY VERIFIED
SMC-10C local-Lipschitz finite-causal-effect ordering proof, checking the
full original receipt SHA256 and source provenance before accepting data.
"""
import argparse
import hashlib
import json
import math
from fractions import Fraction as Q
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'levin_sieve_v1'/'pretrained_trm'
for d in (HERE,PRE):
    sys.path.insert(0,str(d))

from selection_cells_exact import (
    affine_normalized_encoder, topk_selection_certificate, strict_guard
)
from smc07_real_sae import fit_sae,split_ids,EXPECTED_ROSTER
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster

EXPECTED_CHECKPOINT='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
EXPECTED_SMC10_JSON_SHA='176078f22967482420e08a20aa602b7c45a87074e2ab2b600848f2214d9e20aa'
SEEDS=(71,89,113)
RADII=('1/50','1/5')
BUDGETS=('0','1/1000000000000','1/10000000000','1/100000000','1/1000000')
CONTROLS=('isotropic','sae_inactive','jacobian_optimal')


def tensor_exact(t):
    """Canonical rational conversion of exactly stored FP32 binary numbers."""
    return [Q.from_float(float(v)) for v in t.detach().cpu().reshape(-1).tolist()]


def exact_fit_rows(sae,mu,std):
    w=sae.encoder.weight.detach().cpu().tolist()
    b=sae.encoder.bias.detach().cpu().tolist()
    return affine_normalized_encoder(w,b,
                                     mu.detach().cpu().tolist(),
                                     std.detach().cpu().tolist())


def descriptive_all_token_radii(sae,stage,mean,std,test_ids):
    """Fast FP64 exploratory survey. NO certification for these other tokens."""
    with torch.no_grad():
        anchors=stage[list(test_ids)].double().reshape(-1,48)
        scale=std.double()
        pre=torch.nn.functional.linear((anchors-mean.double())/scale,
                                        sae.encoder.weight.double(),
                                        sae.encoder.bias.double())
        order=torch.argsort(pre,dim=-1,descending=True,stable=True)
        chosen=order[:,:4];other=order[:,4:]
        W=sae.encoder.weight.double()/scale[None,:]
        pair_distance=torch.cdist(W,W).clamp_min(0)
        normal=torch.linalg.vector_norm(W,dim=1)
        low=torch.tensor(float('inf'),dtype=torch.float64)
        margin=pre.gather(1,chosen)[:,:,None]-pre.gather(1,other)[:,None,:]
        dist=pair_distance[chosen[:,:,None],other[:,None,:]]
        pair=torch.where(dist>0,margin/dist,low)
        chosen_s=pre.gather(1,chosen)
        active_norm=normal[chosen]
        positive=torch.where(active_norm>0,chosen_s/active_norm,low)
        rk=torch.minimum(pair.amin(dim=(1,2)),positive.amin(dim=1))
        t=order[:,0];competitors=order[:,1:]
        topmargin=pre.gather(1,t[:,None])-pre.gather(1,competitors)
        topdist=pair_distance[t[:,None],competitors]
        top_pairs=torch.where(topdist>0,topmargin/topdist,low)
        top_pos=torch.where(normal[t]>0,pre.gather(1,t[:,None]).squeeze(1)/normal[t],low)
        r1=torch.minimum(top_pairs.amin(dim=1),top_pos)
        invalid=(chosen_s[:,-1]<=0)|(chosen_s[:,-1]<=pre.gather(1,other[:,:1]).squeeze(1))
        invalid|=(pre.gather(1,t[:,None]).squeeze(1)<=pre.gather(1,order[:,1:2]).squeeze(1))
        def stat(z):
            good=z[~invalid]
            if not len(good):
                return {'valid_tokens':0,'quantiles':None}
            return {'valid_tokens':int(len(good)),
                    'min':float(good.min()),
                    'q05':float(torch.quantile(good,0.05)),
                    'q50':float(good.median()),
                    'q95':float(torch.quantile(good,0.95)),
                    'max':float(good.max()),
                    'below_1e-6':int((good<1e-6).sum())}
        return {'n_test_tokens':int(len(anchors)),
                'strict_nominal_tokens':int((~invalid).sum()),
                'method':'FP64 descriptive only; exact proof only for selected goal tokens',
                'top1':stat(r1),'top4':stat(rk)}


def goal_selection_cases(stage,inputs,sae,mu,std,test_ids,seed):
    rows,offsets=exact_fit_rows(sae,mu,std)
    pair_norm_cache={}
    out=[]
    for pid in test_ids:
        goal=int(torch.where(inputs[pid]==0)[0][0])
        anchor=stage[pid,goal,:].detach().cpu().tolist()
        row={'seed':seed,'puzzle_id':pid,'goal_token':goal}
        try:
            cert=topk_selection_certificate(rows,offsets,anchor,k=4,
                                            cache=pair_norm_cache)
            with torch.no_grad():
                pos=torch.relu(sae.encoder(((stage[pid,goal]-mu)/std)[None,:]))[0]
                ref_top=int(pos.argmax())
                ref_support=tuple(sorted(int(z) for z in pos.topk(4).indices.tolist()))
            same=(ref_top==cert['top_index'] and
                  ref_support==cert['topk_indices'])
            row.update({
                'status':'certified' if same else 'abstain_machine_nominal_disagreement',
                'top1_feature':cert['top_index'],
                'top4_features':list(cert['topk_indices']),
                'torch_same_top1_and_top4':same,
                'top1_radius_sq_exact':str(cert['top1']['radius_squared'])
                    if cert['top1']['radius_squared'] is not None else None,
                'top4_radius_sq_exact':str(cert['topk']['radius_squared'])
                    if cert['topk']['radius_squared'] is not None else None,
                'top1_radius_lower_dyadic':str(cert['top1']['dyadic_sqrt_lower'])
                    if cert['top1']['dyadic_sqrt_lower'] is not None else None,
                'top4_radius_lower_dyadic':str(cert['topk']['dyadic_sqrt_lower'])
                    if cert['topk']['dyadic_sqrt_lower'] is not None else None,
                'top1_radius_approx':math.sqrt(float(cert['top1']['radius_squared']))
                    if cert['top1']['radius_squared'] is not None else None,
                'top4_radius_approx':math.sqrt(float(cert['topk']['radius_squared']))
                    if cert['topk']['radius_squared'] is not None else None,
                'sharp_top1_boundary':cert['top1']['active_boundary'],
                'sharp_top4_boundary':cert['topk']['active_boundary'],
                'top1_positive_gap_exact':str(cert['top1_gap']),
                'top4_positive_gap_exact':str(cert['topk_gap']),
                'budget_guards':{
                    eta:{
                        'top1':bool(same and strict_guard(cert['top1']['radius_squared'],Q(eta))),
                        'top4':bool(same and strict_guard(cert['topk']['radius_squared'],Q(eta)))
                    } for eta in BUDGETS},
            })
        except (ValueError,ArithmeticError) as exc:
            row.update({'status':'abstain_'+type(exc).__name__,
                        'reason':str(exc)[:250],
                        'budget_guards':{eta:{'top1':False,'top4':False}
                                         for eta in BUDGETS}})
        out.append(row)
    return out


def load_smc10_receipt(path):
    raw=Path(path).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=EXPECTED_SMC10_JSON_SHA:
        raise ValueError('SMC10 original receipt SHA256 mismatch')
    d=json.loads(raw)
    if (d['schema']!='SMC-10C/verified-endpoint-Lipschitz-neighborhood-v1'
        or d['checkpoint_sha256']!=EXPECTED_CHECKPOINT
        or d['roster_sha256']!=EXPECTED_ROSTER
        or d['seeds']!=list(SEEDS)
        or d['sae_steps']!=300
        or len(d['cases'])!=480):
        raise ValueError('SMC10 experiment metadata mismatch')
    return d


def compose_with_previous(oracle,original_cases):
    index={(r['seed'],r['puzzle_id']):r for r in oracle}
    if len(index)!=len(oracle):
        raise ValueError('duplicate SAE selection case')
    detailed=[]
    for case in original_cases:
        key=(case['seed'],case['puzzle_id'])
        if key not in index:
            continue
        rad=case['radius'];eta=case['common_l2_uncertainty']
        if rad not in RADII or eta not in BUDGETS:
            raise ValueError('unexpected SMC10 radius or uncertainty')
        selection=index[key]
        strict=selection['status']=='certified'
        outcome={}
        for control in CONTROLS:
            original=case['comparisons'][control]
            guard_kind='top4' if control=='sae_inactive' else 'top1'
            safe=strict and selection['budget_guards'][eta][guard_kind]
            # For the Jacobian direction, the reference is the frozen
            # ORIGINAL singular vector; no reoptimization is certified.
            outcome[control]={
                'smc10_certified_original':original,
                'needed_selection_guard':guard_kind,
                'selection_guard_pass':safe,
                'composed_policy_certificate':original if safe else 'abstain_selection',
            }
        detailed.append({'seed':case['seed'],'puzzle_id':case['puzzle_id'],
                         'radius':rad,'l2_budget':eta,'comparisons':outcome})
    summary={}
    for rad in RADII:
        for eta in BUDGETS:
            for control in CONTROLS:
                data=[p['comparisons'][control]['composed_policy_certificate']
                      for p in detailed if p['radius']==rad and p['l2_budget']==eta]
                summary[f'{rad}|{eta}|{control}']={
                    'active_greater':data.count('sae_active_greater'),
                    'control_greater':data.count('control_greater'),
                    'inconclusive_original':data.count('inconclusive'),
                    'abstain_selection':data.count('abstain_selection'),
                    'total':len(data)}
    return {'cases':detailed,'summary':summary,
            'source_smc10_json_sha256':EXPECTED_SMC10_JSON_SHA,
            'interpretation':'Dynamic active SAE atom reselected after uncertain goal-token input; fixed isotropic and original Jacobian-optimal controls, seeded inactive control only if entire top4 support fixed.'}


def run(checkpoint=None,allow_fetch=False,smc10_receipt=None,
        max_test_puzzles=16,seeds=SEEDS,steps=300):
    if not 1<=max_test_puzzles<=16 or not seeds or steps<1:
        raise ValueError('invalid roster slice or SAE training budget')
    from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
    torch.set_num_threads(1)
    lock=json.loads((PRE/'source_lock.json').read_text())
    data,sha=locked_bytes(lock,checkpoint_path=checkpoint,allow_fetch=allow_fetch)
    if sha!=EXPECTED_CHECKPOINT:
        raise ValueError('source checkpoint mismatch')
    teacher=load_pretrained(lock,data).eval()
    (inputs,_,_),roster=prepare_roster(lock)
    if roster!=EXPECTED_ROSTER:
        raise ValueError('roster hash mismatch')
    split=split_ids(64)
    ids=split['test'][:max_test_puzzles]
    with torch.no_grad():
        stage=teacher.reason_states(inputs)[1].detach()
    per_seed=[]
    cases=[]
    for seed in seeds:
        sae,mu,std,fit=fit_sae(stage[split['train']],
                               stage[split['val']],seed=seed,steps=steps)
        goal=goal_selection_cases(stage,inputs,sae,mu,std,ids,seed)
        cases.extend(goal)
        per_seed.append({
            'seed':seed,
            'sae_fit':{'best_step':fit['best_step'],
                       'best_val_mse':fit['best_val_mse']},
            'all_tokens_descriptive':descriptive_all_token_radii(
                sae,stage,mu,std,ids),
            'exact_goal_certificates':len([r for r in goal if r['status']=='certified'])
        })
        print('SELECTION_SEED',seed,
              [(r['puzzle_id'],r['status'],
                r.get('top1_radius_approx'),r.get('top4_radius_approx'))
               for r in goal],flush=True)
    report={
      'schema':'SMC-11/exact-real-sae-selection-cells-v1',
      'source_commit':lock['revision'],'checkpoint_sha256':sha,
      'roster_sha256':roster,'sae_training_steps':steps,
      'train_validation_test_ids':split,'test_puzzle_ids':list(ids),
      'seeds':list(seeds),'k':4,'input_channels':48,'encoder_channels':96,
      'source_math_oracle':'MathCan OLM-11 selection_cells.py',
      'claim_scope':'exact-real selected SAE affine scores with exact stored binary weights, fitted train-only mu/std and FP32 anchors; Torch FP32 nominal identity cross-check, no machine-roundoff proof',
      'cases':cases,'per_seed':per_seed,
    }
    good=[r for r in cases if r['status']=='certified']
    report['summary']={
       'certified_goal_cases':len(good),'total_goal_cases':len(cases),
       'top1_min':min((r['top1_radius_approx'] for r in good),default=None),
       'top1_median':sorted(r['top1_radius_approx'] for r in good)[len(good)//2] if good else None,
       'top4_min':min((r['top4_radius_approx'] for r in good),default=None),
       'top4_median':sorted(r['top4_radius_approx'] for r in good)[len(good)//2] if good else None,
       'guard_counts':{
          eta:{k:sum(r['budget_guards'][eta][k] for r in cases)
               for k in ('top1','top4')} for eta in BUDGETS},
    }
    if smc10_receipt:
        if steps!=300 or set(seeds)!=set(SEEDS):
            raise ValueError('SMC10 composition requires identical SAE fit seeds and 300 steps')
        receipt=load_smc10_receipt(smc10_receipt)
        report['composed_smc10']=compose_with_previous(cases,receipt['cases'])
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--smc10-receipt',type=Path)
    p.add_argument('--max-test-puzzles',type=int,default=16)
    p.add_argument('--seeds',default='71,89,113')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--out',type=Path,default=Path('runs/smc11/full.json'))
    a=p.parse_args()
    result=run(a.checkpoint,a.fetch,a.smc10_receipt,a.max_test_puzzles,
               tuple(map(int,a.seeds.split(','))),a.steps)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(a.out),'checkpoint_sha256':result['checkpoint_sha256'],
                      'summary':result['summary'],
                      'composition_summary':result.get('composed_smc10',{}).get('summary')},
                     indent=2))


if __name__=='__main__':
    main()
