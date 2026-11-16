"""SMC-22: freeze exact original 64-puzzle residual calibration under LAD.

This phase is source-only, independent of test states. Verify SMC19 frozen
refits and full SMC21 original-calibration receipts before fitting two
weighted-median one-parameter corrections per original refit pair.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_smc21 import audit_calibration

PRECOMMIT='b208c1e7a63ac3e0d3008564d25bded5a2f80f52'
SMC21_CALIB_SHA='e7aa12a3da5b8eaa0132b5de45083ca1b2af5c38e556fef01e1f26d2b94f90af'
SMC19_SHA='612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b'
CLIP=2.
MIN_WEIGHT=1e-12


def sha(content):
    return hashlib.sha256(content).hexdigest()


def objective(raw,beta,source_key):
    return sum(abs(v['target_finite_residual']-beta*v[source_key])
               for row in raw for v in row['metrics'] if v is not None)


def weighted_median_fit(raw,source_key,reference_ridge):
    """Single parameter no intercept, fixed lower weighted median with clip."""
    n=0;puzzles=0;data=[]
    for row in raw:
        has=False
        for j,record in enumerate(row['metrics']):
            if record is None:continue
            has=True;n+=1
            x=float(record[source_key]);y=float(record['target_finite_residual'])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ArithmeticError('nonfinite source calibration residual')
            if x:
                data.append({'ratio':y/x,'weight':abs(x),
                             'puzzle_index':row['puzzle_index'],'goal_slot':j,
                             'source_residual':x,'target_residual':y})
        puzzles+=int(has)
    data.sort(key=lambda t:(t['ratio'],t['puzzle_index'],t['goal_slot']))
    W=sum(t['weight'] for t in data)
    degenerate=(W<=MIN_WEIGHT)
    idx=None;unclipped=0.;beta=0.
    if not degenerate:
        run=0.
        for i,item in enumerate(data):
            run+=item['weight']
            if 2.*run>=W:
                idx=i;unclipped=item['ratio'];break
        if idx is None:raise ArithmeticError('weighted median missing')
        beta=max(-CLIP,min(CLIP,unclipped))
    old=float(reference_ridge['coefficient'])
    return {
      'source_key':source_key,
      'coefficient':beta,'unclipped_median':unclipped,
      'winning_sorted_index':idx,
      'sorted_nonzero_ratio_weight_records':data,
      'total_weight':W,'source_nonzero_count':len(data),
      'calibration_eligible_goals':n,'calibration_eligible_puzzles':puzzles,
      'degenerate':degenerate,'identified':n>=24 and puzzles>=20 and not degenerate,
      'calibration_absolute_error_lad':objective(raw,beta,source_key),
      'calibration_absolute_error_original_ridge':objective(raw,old,source_key),
      'original_202621_source_ridge_coefficient':old,
      'clip':[-2.,2.]
    }


def freeze(original_smc21_calibration,frozen_smc19_dir,out):
    original_path=Path(original_smc21_calibration)
    raw=original_path.read_bytes()
    if sha(raw)!=SMC21_CALIB_SHA:
        raise ValueError('original SMC21 calibration source SHA256 mismatch')
    old=json.loads(raw)
    root=Path(frozen_smc19_dir)
    source_bytes=(root/'frozen.json').read_bytes()
    if sha(source_bytes)!=SMC19_SHA:
        raise ValueError('original SMC19 weights / calibration source hash changed')
    s19=json.loads(source_bytes)
    source_audit=audit_calibration(old,s19,root)
    if (old['pre_registration_commit']!='0f0af2dc4b34c6a85dc4e0ba5d5d1e08597bbf59'
        or len(old['pairs'])!=4 or len(source_audit['comparisons'])!=4):
        raise ValueError('original calibration source protocol incomplete')
    pairs=[]
    for i,prior in enumerate(old['pairs']):
        rows=prior['original_64_puzzle_calibration']
        fit_a=weighted_median_fit(rows,'source_finite_residual',prior['beta_source'])
        fit_o=weighted_median_fit(rows,'other_finite_residual',prior['beta_other'])
        if fit_a['calibration_eligible_goals']!=fit_o['calibration_eligible_goals']:
            raise ValueError('unequal original source and other calibration budgets')
        pair={
         'source':prior['source'],'target':prior['target'],
         'frozen_SMC19_mapping_hashes':prior['frozen_original_mapping_hashes'],
         'lad_source':fit_a,'lad_other':fit_o,
         'original_ridge_source':prior['beta_source'],
         'original_ridge_other':prior['beta_other']}
        pairs.append(pair)
        print('SMC22_SOURCE_ONLY_LAD_FROZEN',pair['target'],{
          'source_beta':fit_a['coefficient'],'other_beta':fit_o['coefficient'],
          'original_ridge_beta':prior['beta_source']['coefficient'],
          'source_calibration_MAE':fit_a['calibration_absolute_error_lad'],
          'original_source_ridge_calibration_MAE':fit_a['calibration_absolute_error_original_ridge']},flush=True)
    receipt={
     'schema':'SMC-22/original-source-only-LAD-freeze-v1',
     'pre_registration_commit':PRECOMMIT,
     'original_smc21_calibration_json_sha256':sha(raw),
     'original_smc19_frozen_refit_json_sha256':sha(source_bytes),
     'original_SMC21_calibration_independent_audit':source_audit,
     'reference_calibration_dataset':'SMC13 original feature_calibration',
     'calibration_puzzles':64,
     'one_parameter_rule':'lower weighted median of target/source residual ratio weighted by absolute source residual',
     'clip':[-2.,2.],'weight_tolerance':MIN_WEIGHT,
     'source_only_calibration_frozen_before_new_tests':True,
     'pairs':pairs,
     'scope':'Only original source-locked SMC21 64-puzzle calibration; no test inputs, gradients, erasures or labels scored.'
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print('SMC22_LAD_CALIBRATION_SOURCE_HASH',sha(out.read_bytes()),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smc21-calibration',type=Path,required=True)
    p.add_argument('--frozen-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/smc22/calibration/lad_calibration.json'))
    a=p.parse_args()
    freeze(a.smc21_calibration,a.frozen_dir,a.out)
