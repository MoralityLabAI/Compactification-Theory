"""SHAFT-COMP-19: merge phase-gated four-cell exact receipts and AUC tests.

Only consumes frozen calibration/heldout JSON. Never retrains, changes source
thresholds, or searches thresholds on heldout labels. All sources are local.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path
import numpy as np
from research.shaft_compactification.experiments import capacity_history as cap

FIELDS=('instant','duplicate','history','instant_norm','duplicate_norm','history_norm','distance')
CELLS=tuple(cap.SEEDS)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def auc(scores,labels):return cap.auc(scores,labels)

def count_auc(rows,field):
    valid=[r for r in rows if r['eligible']]
    positives=[r[field] for r in valid if r['future_any_8_16_32']==1]
    negatives=sorted(r[field] for r in valid if r['future_any_8_16_32']==0)
    if not positives or not negatives:
        return {'positive':len(positives),'negative':len(negatives),
                'strict_wins':None,'tied_pairs':None,'auc':None}
    strict=sum(bisect_left(negatives,p) for p in positives)
    ties=sum(bisect_right(negatives,p)-bisect_left(negatives,p) for p in positives)
    val=(strict+ties/2)/(len(positives)*len(negatives))
    value=auc([r[field] for r in valid],[r['future_any_8_16_32'] for r in valid])
    assert abs(value-val)<1e-12,(field,val,value)
    return {'positive':len(positives),'negative':len(negatives),
            'strict_wins':strict,'tied_pairs':ties,'auc':round(val,6)}

def summarise(rows):
    valid=[r for r in rows if r['eligible']]
    out={'total':len(rows),'eligible':len(valid),
         'positives':sum(r['future_any_8_16_32'] for r in valid),
         'auc':{f:count_auc(rows,f)['auc'] for f in FIELDS}}
    out['primary_delta']=None if out['auc']['history'] is None or out['auc']['duplicate'] is None else round(out['auc']['history']-out['auc']['duplicate'],6)
    return out

def operating(rows,field,threshold):
    valid=[r for r in rows if r['eligible']]
    neg=[r for r in valid if r['future_any_8_16_32']==0]
    pos=[r for r in valid if r['future_any_8_16_32']==1]
    fp=sum(r[field]>threshold for r in neg);tp=sum(r[field]>threshold for r in pos)
    return {'threshold':threshold,'negatives':len(neg),'positives':len(pos),
            'false_positives':fp,'true_positives':tp,
            'fpr':round(fp/len(neg),6) if neg else None,
            'tpr':round(tp/len(pos),6) if pos else None}

def inference(rows,nperm=1000,nboot=1000):
    valid=[r for r in rows if r['eligible']]
    ys=np.asarray([r['future_any_8_16_32'] for r in valid],dtype=np.int64)
    h=np.asarray([r['history'] for r in valid],dtype=float)
    d=np.asarray([r['duplicate'] for r in valid],dtype=float)
    indices={}
    for i,r in enumerate(valid): indices.setdefault((r['cell'],r['depth']),[]).append(i)
    observed=auc(h,ys)-auc(d,ys)
    rng=np.random.default_rng(190019)
    permutations=[]
    for _ in range(nperm):
        yp=ys.copy()
        for ids in indices.values():yp[ids]=rng.permutation(yp[ids])
        ah,ad=auc(h,yp),auc(d,yp)
        if ah is not None and ad is not None:permutations.append(ah-ad)
    center=float(np.mean(permutations))
    p=(1+sum(abs(x-center)>=abs(observed-center)-1e-12 for x in permutations))/(len(permutations)+1)
    bootstrap=[]
    for _ in range(nboot):
        idx=np.concatenate([rng.choice(v,len(v),replace=True) for v in indices.values()])
        ah,ad=auc(h[idx],ys[idx]),auc(d[idx],ys[idx])
        if ah is not None and ad is not None:bootstrap.append((ah,ad,ah-ad))
    boot=np.asarray(bootstrap)
    q=lambda col:[round(float(x),6) for x in np.quantile(boot[:,col],[.025,.975])]
    return {'observed_delta':round(observed,6),
            'permutation_null_mean':round(center,6),
            'permutation_two_sided_p_centered':round(p,6),
            'permutation_trials':len(permutations),'bootstrap_trials':len(boot),
            'history_bootstrap95':q(0),'duplicate_bootstrap95':q(1),
            'primary_delta_bootstrap95':q(2),
            'note':'within-four-fixed-models by model×depth; no between-model population CI'}

def merge(root,threshold_path,nperm=1000,nboot=1000):
    root=Path(root);threshold_path=Path(threshold_path)
    frozen=json.loads(threshold_path.read_text())
    assert frozen['registration_commit']==cap.REGISTRATION_COMMIT
    assert frozen['phase']=='calibration_thresholds'
    assert sha(Path(cap.__file__))==frozen['source_code_sha256'], 'code drift after calibration'
    original={};rows={};records={};model_rows={}
    for phase in ('calibration','heldout'):
        allrows=[];per_cell={};per_model={};sources={}
        for cell in CELLS:
            path=root/f'{"cal" if phase=="calibration" else "test"}_{cell}.json'
            data=json.loads(path.read_text())
            expected_seed=cap.SEEDS[cell][phase]
            assert data['seed']==expected_seed and data['phase']==phase and data['cell']==cell
            assert data['train_steps']==cap.TRAIN_STEPS and data['per_goal']==cap.PER_GOAL
            assert data['registration_commit']==cap.REGISTRATION_COMMIT
            assert data['probes']['history']['outputs']==36
            assert data['probes']['duplicate']['outputs']==36
            assert data['probes']['instant']['outputs']==6
            if phase=='calibration':
                assert frozen['calibration_models'][cell]['model_sha']==data['model']['model_sha256']
                assert frozen['calibration_models'][cell]['receipt_sha']==sha(path)
            for r in data['rows']:
                r['cell']=cell
                assert r['seed']==expected_seed
            assert len(data['rows'])==2048
            allrows+=data['rows']
            per_cell[cell]=summarise(data['rows'])
            per_model[str(expected_seed)]=per_cell[cell]
            sources[cell]={'sha256':sha(path),'model_sha256':data['model']['model_sha256'],
                'training_accuracy_d12':data['model']['accuracy_d12'],
                'training_accuracy_d32':data['model']['accuracy_d32'],
                'probe_sha256':{k:v['sha256'] for k,v in data['probes'].items()},
                'probe_bit_accuracy_by_start_depth':{d:m['probe_bit_agreement'] for d,m in data['depth_checks'].items()}}
        rows[phase]=allrows
        records[phase]={'aggregate':summarise(allrows),'per_cell':per_cell,
                        'models':sources,'per_model':per_model,
                        'auc_pairs':{f:count_auc(allrows,f) for f in FIELDS}}
    cal_valid=[r for r in rows['calibration'] if r['eligible'] and not r['future_any_8_16_32']]
    assert len(cal_valid)==frozen['calibration_counts']['negatives']
    for feature,th in frozen['pooled_thresholds'].items():
        measured=float(np.quantile([r[feature] for r in cal_valid],.9,method='higher'))
        assert measured==th,(feature,measured,th)
    test=rows['heldout']
    ops={cell:{key:operating([r for r in test if r['cell']==cell],key,t)
               for key,t in frozen['pooled_thresholds'].items()} for cell in CELLS}
    aggregate_ops={key:operating(test,key,t) for key,t in frozen['pooled_thresholds'].items()}
    adapted={cell:{key:operating([r for r in test if r['cell']==cell],key,t)
                   for key,t in frozen['cell_thresholds'][cell].items()} for cell in CELLS}
    infer=inference(test,nperm,nboot)
    signs=sum(1 for s in records['heldout']['per_cell'].values() if s['primary_delta'] is not None and s['primary_delta']>0)
    primary_supported=bool(infer['primary_delta_bootstrap95'][0]>0 and signs>=3)
    fpr_transfer=all(ops[cell]['history']['fpr'] is not None and ops[cell]['history']['fpr']<=0.15 for cell in CELLS)
    summary={'experiment':'SHAFT-COMP-19',
             'registration_commit':cap.REGISTRATION_COMMIT,
             'calibration_freeze_commit':'671a60e3f5502b9c11c66292ea900af58632384e',
             'frozen_threshold_sha256':sha(threshold_path),
             'source_code_sha256':sha(Path(cap.__file__)),
             'task_families':['lag_rotate','lag_match'],
             'model_architectures':['gru','lstm'],
             'vocab':6,'state_depths':list(cap.STATE_DEPTHS),
             'endpoints':{'number_of_futures':2,'steps':[8,16,32],'outcome':'any delayed query disagreement'},
             'training_steps':cap.TRAIN_STEPS,'pairs_per_cell_per_depth':1024,
             'calibration':records['calibration'],'heldout':records['heldout'],
             'inference':infer,
             'fixed_source_thresholds':frozen['pooled_thresholds'],
             'global_operating_points':aggregate_ops,
             'cell_fixed_source_operating_points':ops,
             'cell_adapted_calibration_operating_points':adapted,
             'positive_model_count':signs,
             'registered_primary_supported':primary_supported,
             'registered_fpr_transfer_pass':fpr_transfer,
             'evidence_scope':'toy 64-goal six-symbol GRU/LSTM, same-goal natural state pairs, intact weights; one heldout seed per cell'}
    full={'summary':summary,'calibration_rows':rows['calibration'],'heldout_rows':rows['heldout']}
    return summary,full

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/mnt/data/shaft19_work'))
    p.add_argument('--thresholds',type=Path,default=Path('/mnt/data/shaft19_work/comp19_frozen_thresholds.json'))
    p.add_argument('--permutations',type=int,default=1000)
    p.add_argument('--bootstraps',type=int,default=1000)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--full',type=Path,required=True)
    a=p.parse_args()
    summary,full=merge(a.root,a.thresholds,a.permutations,a.bootstraps)
    for path,data in ((a.summary,summary),(a.full,full)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'calibration':summary['calibration']['aggregate'],
                      'heldout':summary['heldout']['aggregate'],
                      'inference':summary['inference'],
                      'models_positive':summary['positive_model_count'],
                      'primary_supported':summary['registered_primary_supported'],
                      'fixed_fpr_transfer':summary['registered_fpr_transfer_pass']},indent=2))

if __name__=='__main__':main()
