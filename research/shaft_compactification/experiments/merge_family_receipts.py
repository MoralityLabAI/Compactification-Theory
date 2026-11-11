"""Compose separately executed SHAFT-COMP-16 family receipts without refitting.

The merge requires disjoint, registered seed sets; it does not modify any
feature or outcome label. Statistical inference is rerun on the union.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from research.shaft_compactification.experiments.family_transfer import (
    FAMILY_SEEDS,SOURCE_THRESHOLDS,summary,contrasts,operating_point,paired_inference,
    PERMUTATIONS,BOOTSTRAPS
)


def merge(paths,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    if len(paths)!=len(FAMILY_SEEDS):
        raise ValueError('expected exactly one receipt per registered task family')
    payloads=[json.loads(Path(path).read_text()) for path in paths]
    found=set()
    merged=dict(payloads[0])
    merged.update(families={},rows=[],cells={},model_hashes={},training={})
    merged['per_family_receipt_sha256']={}
    for path,payload in zip(paths,payloads):
        families=list(payload['families'])
        if len(families)!=1:raise ValueError('each receipt must contain one family')
        family=families[0]
        if family not in FAMILY_SEEDS or family in found:
            raise ValueError(f'unknown or duplicated family {family}')
        found.add(family)
        expected=set(FAMILY_SEEDS[family]['calibration']+FAMILY_SEEDS[family]['heldout'])
        if set(map(int,payload['model_hashes']))!=expected:
            raise ValueError('trained model seeds do not match preregistration')
        if payload['source_thresholds']!=SOURCE_THRESHOLDS:
            raise ValueError('source thresholds changed')
        if payload['training_steps']!=1200 or payload['per_goal']!=16:
            raise ValueError('training/sampling protocol drift')
        merged['families'][family]=payload['families'][family]
        merged['rows'].extend(payload['rows'])
        merged['cells'].update(payload['cells'])
        merged['model_hashes'].update(payload['model_hashes'])
        merged['training'].update(payload['training'])
        merged['per_family_receipt_sha256'][family]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if found!=set(FAMILY_SEEDS):raise ValueError('registered family missing')
    heldout=[r for r in merged['rows'] if r['heldout']]
    merged['heldout_pooled']=summary(heldout)
    merged['heldout_pooled']['contrasts']=contrasts(merged['heldout_pooled']['auc'])
    merged['source_threshold_pooled']={key:operating_point(heldout,key,threshold)
                                       for key,threshold in SOURCE_THRESHOLDS.items()}
    merged['inference']=paired_inference(merged['rows'],permutations,bootstraps)
    signs=[]
    for family,cfg in FAMILY_SEEDS.items():
        for seed in cfg['heldout']:
            cell=summary([r for r in heldout if r['seed']==seed])
            diff=contrasts(cell['auc'])['learned_direction_minus_rank']
            signs.append({'seed':seed,'family':family,'auc_gap':diff,'positives':cell['positives'],
                          'eligible':cell['eligible']})
    merged['seed_signs']=signs
    ci=merged['inference'].get('bootstrap_95')
    merged['decision']={
      'directional_prediction_transfer_supported':bool(
          ci and ci[0]>0 and all(
              v['heldout']['contrasts']['learned_direction_minus_rank'] is not None and
              v['heldout']['contrasts']['learned_direction_minus_rank']>0
              for v in merged['families'].values())
          and sum(x['auc_gap'] is not None and x['auc_gap']>0 for x in signs)>=3),
      'source_threshold_fpr_transfer_supported':all(
          merged['families'][f]['source_thresholds']['learned_direction']['fpr'] is not None and
          merged['families'][f]['source_thresholds']['learned_direction']['fpr']<=.15
          for f in FAMILY_SEEDS),
    }
    merged['execution_note']='per-family independent CPU training, deterministic union and inference'
    return merged


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family-receipt',type=Path,action='append',required=True)
    p.add_argument('--permutations',type=int,default=PERMUTATIONS)
    p.add_argument('--bootstraps',type=int,default=BOOTSTRAPS)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    payload=json.dumps(merge(a.family_receipt,a.permutations,a.bootstraps),sort_keys=True,indent=2)+'\n'
    if a.out:a.out.write_text(payload)
    else:print(payload,end='')
if __name__=='__main__':main()
