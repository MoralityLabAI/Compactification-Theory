"""SHAFT-COMP-16: dynamic lookup families, fixed source FPR and directional observability.

Local CPU PyTorch, intentionally isolated synthetic tasks. The experiment does
not use networks, devices, agents, or persistence behavior. See preregistration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from research.shaft_compactification.experiments.trained_recurrence import (
    TaskMemoryGRU, GOALS, TOKENS, BATCH, TRAIN_LENGTH, HIDDEN
)
from research.shaft_compactification.experiments.causal_probe import train_probe
from research.shaft_compactification.experiments.observability_prediction import model_digest, auc
from research.shaft_compactification.experiments.factorial_observability import (
    cell_rows, summary, contrasts, SCORES
)

FAMILY_SEEDS={
    'lag_rotate': {'calibration': (1101,1102), 'heldout': (1103,1104)},
    'lag_match': {'calibration': (1201,1202), 'heldout': (1203,1204)},
}
SOURCE_THRESHOLDS={'learned_direction': 6.854760490520719, 'learned_rank':2.5}
DEPTHS=(12,40)
TRAIN_STEPS=1200
TRAIN_LR=.01
PER_GOAL=16
PERMUTATIONS=1000
BOOTSTRAPS=1000


def family_targets(goals, tokens, family):
    """Targets depend on prior query x_(t-1), with x_-1 = 0."""
    if family not in FAMILY_SEEDS:
        raise ValueError(f'unknown family {family}')
    previous=torch.cat((torch.zeros_like(tokens[:,:1]),tokens[:,:-1]),dim=1)
    if family=='lag_rotate':
        index=(tokens+previous) % TOKENS
        return ((goals[:,None] >> index) & 1).float()
    return (((goals[:,None] >> tokens) & 1) ^ (tokens==previous).long()).float()


def fit_family(seed, family, steps=TRAIN_STEPS):
    torch.manual_seed(seed)
    generator=torch.Generator(device='cpu').manual_seed(10000+seed)
    model=TaskMemoryGRU()
    optimizer=torch.optim.Adam(model.parameters(),lr=TRAIN_LR)
    model.train()
    for _ in range(steps):
        goals=torch.randint(0,GOALS,(BATCH,),generator=generator)
        token=torch.randint(0,TOKENS,(BATCH,TRAIN_LENGTH),generator=generator)
        target=family_targets(goals,token,family)
        loss=F.binary_cross_entropy_with_logits(model(goals,token),target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        optimizer.step()
    model.eval()
    return model,float(loss.item())


@torch.no_grad()
def fresh_task_accuracy(model, seed, family, length=12, samples=1024):
    generator=torch.Generator(device='cpu').manual_seed(3000000+seed+length*43)
    goals=torch.randint(0,GOALS,(samples,),generator=generator)
    tokens=torch.randint(0,TOKENS,(samples,length),generator=generator)
    actual=model(goals,tokens)>=0
    truth=family_targets(goals,tokens,family)>=.5
    return round(float((actual==truth).float().mean()),6)


def operating_point(rows, feature, threshold):
    selected=[r for r in rows if r['eligible']]
    negative=[r for r in selected if not r['diverged_32']]
    positive=[r for r in selected if r['diverged_32']]
    fp=sum(r[feature]>threshold for r in negative)
    tp=sum(r[feature]>threshold for r in positive)
    return dict(threshold=float(threshold), negatives=len(negative),positives=len(positive),
                false_positives=fp,true_positives=tp,
                fpr=round(fp/len(negative),6) if negative else None,
                tpr=round(tp/len(positive),6) if positive else None)


def reference_threshold(rows, feature):
    negatives=np.array([r[feature] for r in rows if r['eligible'] and not r['diverged_32']],float)
    if len(negatives)==0:
        return None
    return float(np.quantile(negatives,.90,method='higher'))


def paired_inference(rows, permutations=PERMUTATIONS, bootstraps=BOOTSTRAPS):
    e=[r for r in rows if r['eligible'] and r['heldout']]
    if not e:
        return {'error':'no eligible holdout'}
    labels=np.array([r['diverged_32'] for r in e],int)
    direction=np.array([r['learned_direction'] for r in e],float)
    rank=np.array([r['learned_rank'] for r in e],float)
    primary_auc=auc(direction,labels)
    rank_auc=auc(rank,labels)
    if primary_auc is None or rank_auc is None:
        return {'error':'one-class heldout'}
    groups={}
    for i,r in enumerate(e):
        groups.setdefault((r['family'],r['seed'],r['depth']),[]).append(i)
    rng=np.random.default_rng(161616)
    null=[]
    for _ in range(permutations):
        y=labels.copy()
        for ids in groups.values():y[ids]=rng.permutation(y[ids])
        a,b=auc(direction,y),auc(rank,y)
        if a is not None and b is not None:null.append(a-b)
    delta=primary_auc-rank_auc
    center=float(np.mean(null)) if null else None
    p=(1+sum(abs(v-center)>=abs(delta-center)-1e-12 for v in null))/(len(null)+1) if null else None
    boot=[]
    for _ in range(bootstraps):
        idx=np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in groups.values()])
        a,b=auc(direction[idx],labels[idx]),auc(rank[idx],labels[idx])
        if a is not None and b is not None:boot.append(a-b)
    ci=np.quantile(boot,[.025,.975]).tolist() if boot else None
    return dict(primary_direction_auc=round(primary_auc,6),rank_auc=round(rank_auc,6),
                auc_difference=round(delta,6),bootstrap_95=[round(v,6) for v in ci] if ci else None,
                permutation_p_centered=round(p,6) if p is not None else None,
                permutation_null_mean=round(center,6) if center is not None else None,
                permutation_trials=len(null),bootstrap_trials_valid=len(boot))


def run(per_goal=PER_GOAL,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS,
        families=None,training_steps=TRAIN_STEPS):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    selected=list(FAMILY_SEEDS) if families is None else list(families)
    report=dict(experiment='SHAFT-COMP-16',protocol='SHAFT_COMP_16_PROTOCOL.md',
                torch=torch.__version__,numpy=np.__version__,calibration_source='COMP-15',
                source_thresholds=SOURCE_THRESHOLDS,training_steps=training_steps,
                training_lr=TRAIN_LR,training_sequence_length=TRAIN_LENGTH,
                per_goal=per_goal,diagnostic_steps=5,outcome_steps=32,
                model_hashes={},cells={},training={},families={},rows=[])
    for family in selected:
        cfg=FAMILY_SEEDS[family]
        rows_by_family=[]
        for part in ('calibration','heldout'):
            for seed in cfg[part]:
                model,train_loss=fit_family(seed,family,steps=training_steps)
                probe,probe_loss=train_probe(model,seed)
                report['model_hashes'][str(seed)]={
                    'family':family,'base':model_digest(model),'probe':model_digest(probe)}
                report['training'][str(seed)]={
                    'family':family,'train_loss_last':round(train_loss,7),
                    'probe_loss_last':round(probe_loss,7),
                    'accuracy_12_steps_fresh':fresh_task_accuracy(model,seed,family,12),
                    'accuracy_32_steps_fresh':fresh_task_accuracy(model,seed,family,32),
                }
                for depth in DEPTHS:
                    cell=cell_rows(model,probe,seed,depth,per_goal)
                    for r in cell:
                        r['family']=family
                        r['heldout']=part=='heldout'
                    report['cells'][f'{family}_{seed}_d{depth}']=summary(cell)
                    rows_by_family.extend(cell)
                    report['rows'].extend(cell)
        cal=[r for r in rows_by_family if not r['heldout']]
        test=[r for r in rows_by_family if r['heldout']]
        cells=dict(calibration=summary(cal),heldout=summary(test))
        cells['calibration']['contrasts']=contrasts(cells['calibration']['auc'])
        cells['heldout']['contrasts']=contrasts(cells['heldout']['auc'])
        cells['source_thresholds']={key:operating_point(test,key,value)
                                    for key,value in SOURCE_THRESHOLDS.items()}
        adaptive={}
        for name in SOURCE_THRESHOLDS:
            threshold=reference_threshold(cal,name)
            adaptive[name]=dict(calibration_threshold=threshold,
                                calibration=operating_point(cal,name,threshold),
                                heldout=operating_point(test,name,threshold)) if threshold is not None else None
        cells['family_calibrated_thresholds']=adaptive
        report['families'][family]=cells
    heldout=[r for r in report['rows'] if r['heldout']]
    report['heldout_pooled']=summary(heldout)
    report['heldout_pooled']['contrasts']=contrasts(report['heldout_pooled']['auc'])
    report['source_threshold_pooled']={key:operating_point(heldout,key,value)
                                       for key,value in SOURCE_THRESHOLDS.items()}
    if set(selected)==set(FAMILY_SEEDS):
        report['inference']=paired_inference(report['rows'],permutations,bootstraps)
        signs=[]
        for family,cfg in FAMILY_SEEDS.items():
            for seed in cfg['heldout']:
                d=summary([r for r in heldout if r['seed']==seed])
                gap=contrasts(d['auc'])['learned_direction_minus_rank']
                signs.append({'seed':seed,'family':family,'auc_gap':gap})
        report['seed_signs']=signs
        ci=report['inference'].get('bootstrap_95')
        report['decision']={
            'directional_prediction_transfer_supported':bool(
                ci and ci[0]>0 and all(v['heldout']['contrasts']['learned_direction_minus_rank']>0
                                     for v in report['families'].values())
                and sum(x['auc_gap'] is not None and x['auc_gap']>0 for x in signs)>=3),
            'source_threshold_fpr_transfer_supported':all(
                report['families'][f]['source_thresholds']['learned_direction']['fpr'] is not None
                and report['families'][f]['source_thresholds']['learned_direction']['fpr']<=.15
                for f in FAMILY_SEEDS),
        }
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--per-goal',type=int,default=PER_GOAL)
    ap.add_argument('--permutations',type=int,default=PERMUTATIONS)
    ap.add_argument('--bootstraps',type=int,default=BOOTSTRAPS)
    ap.add_argument('--training-steps',type=int,default=TRAIN_STEPS)
    ap.add_argument('--family',choices=tuple(FAMILY_SEEDS),action='append')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if not 1<=args.per_goal<=64 or args.training_steps<1:
        raise ValueError('invalid per-goal / training-steps')
    result=run(args.per_goal,args.permutations,args.bootstraps,args.family,args.training_steps)
    output=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.out:
        args.out.parent.mkdir(exist_ok=True,parents=True)
        args.out.write_text(output)
    else:print(output,end='')

if __name__=='__main__':main()
