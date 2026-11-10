"""OIC-15 standard-library frozen-receipt verifier.

The verifier independently reconstructs all *finite oracle* traces, validates
probability metrics from archived model outputs, and checks hashes. It does
not certify stochastic gradient optimization. `oic15_analyze.py` additionally
replays the stored trained parameter snapshots with PyTorch.
"""
from __future__ import annotations
import base64
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from oic15_math import (raw_codebook,codebook_min_error_correct,finite_exact_bayes_accuracy,
    rotating_pulse_classified,residual_phase_entropy)

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'


def require(condition:bool, message:str):
    if not condition:raise ValueError(message)


def sha256(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def load_archive(path:Path,receipt_info:dict)->dict:
    b64=path.read_text().strip().encode('ascii')
    raw_compressed=base64.b64decode(b64,validate=True)
    require(sha256(raw_compressed)==receipt_info['gzip_sha256'],path.name+' compressed SHA mismatch')
    raw=gzip.decompress(raw_compressed)
    require(sha256(raw)==receipt_info['raw_sha256'],path.name+' JSON SHA mismatch')
    return json.loads(raw)


def independent_oracle(n:int,f:int,z:int,word:tuple[int,...],marker:int=0)->tuple[int,...]:
    x=0; trace=[x]
    for t,u in enumerate(word):
        x=1-x if u==0 or (z+t*f)%n!=marker else x
        trace.append(x)
    return tuple(trace)


def validate_panels(trace_archive:dict,analysis:dict)->dict:
    protocol=trace_archive['protocol']
    require(protocol['observer_receives_phase'] is False,'unexpected privileged phase interface')
    require(protocol['observer_receives_n'] is True,'n interface unspecified')
    runs=trace_archive['runs']
    require(len(runs)==30,'30 fits expected')
    require(len({(r['architecture'],r['split'],r['seed']) for r in runs})==30,'duplicate runs')
    require({r['seed'] for r in runs}==set(range(5)),'seed coverage')
    require({r['architecture'] for r in runs}=={'gru','cyclic','raw'},'architecture coverage')
    require({r['split'] for r in runs}=={'full','heldout'},'split coverage')
    expected_ns={4,5,6,7,8}
    schedules={'passive','full_active','latch','extended_latch','alternating','random_budget'}
    checked=0
    for run in runs:
        require(set(map(int,run['panels']))==expected_ns,'phase-count coverage')
        require(run['training']['train_steps']==500,'training steps changed')
        require(run['training']['train_ns']==[4,5,6],'training n changed')
        require(run['training']['parameters']==(3873 if run['architecture']=='gru' else 1),'parameter count mismatch')
        for n in sorted(expected_ns):
            panels=run['panels'][str(n)]
            require(set(panels)==schedules,'evaluation panel coverage')
            # Every passive readout from a uniformly unknown phase is identical across
            # families, so any deterministic classifier must output the same value.
            passive=panels['passive']['predicted_probabilities']
            require(len(passive)==2*n,'passive sample count')
            require(max(passive)-min(passive)<1e-6,'passive phase/family data leak')
            # Check finite exact task's all-input family identifiability (no learned claim).
            a=raw_codebook(n,0,(1,)*n)
            b=raw_codebook(n,1,(1,)*n)
            require(not set(a)&set(b),'exact n-probe family observation collision')
            require(len(set(raw_codebook(n,0,(0,)*(2*n))))==1,'passive oracle mismatch')
            for name,record in panels.items():
                p=record['predicted_probabilities']
                require(len(p)==2*n,'panel probability count')
                require(all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in p), 'invalid probability')
                labels=[0]*n+[1]*n
                preds=[int(v>0.5) for v in p]
                accuracy=sum(int(p0==y) for p0,y in zip(preds,labels))/(2*n)
                brier=sum((v-y)**2 for v,y in zip(p,labels))/(2*n)
                mean_prob=sum(p)/(2*n)
                deviation=sum(abs(v-.5) for v in p)/(2*n)
                require(abs(accuracy-record['accuracy'])<1e-8,'accuracy tampering')
                require(abs(brier-record['brier'])<2e-6,'Brier score tampering')
                require(abs(mean_prob-record['mean_probability'])<2e-6,'probability mean tampering')
                require(abs(deviation-record['passive_deviation'])<2e-6,'deviation tampering')
                require(abs(sum(int(preds[i]==0) for i in range(n))/n -record['correct_by_family']['fixed'])<1e-8,'fixed accuracy mismatch')
                require(abs(sum(int(preds[i]==1) for i in range(n,2*n))/n -record['correct_by_family']['rotating'])<1e-8,'rotating accuracy mismatch')
                checked+=1
    # Recompute published aggregate scores from archived probabilities.
    metrics=analysis['metrics']
    require(len(metrics)==120,'expected 120 summary panels')
    for m in metrics:
        grp=sorted([r for r in runs if r['split']==m['split'] and r['architecture']==m['architecture']],key=lambda r:r['seed'])
        require(len(grp)==5,'five seeds required')
        panels=[r['panels'][str(m['n'])][m['schedule']] for r in grp]
        accuracies=[p['accuracy'] for p in panels]
        require(all(abs(a-b)<1e-8 for a,b in zip(accuracies,m['per_seed_accuracies'])),'per-seed summary mismatch')
        for input_key,output_key in [('accuracy','mean_accuracy'),('brier','mean_brier'),('passive_deviation','mean_passive_deviation')]:
            mean=sum(p[input_key] for p in panels)/5
            require(abs(mean-m[output_key])<1e-8,'aggregate score mismatch '+output_key)
    # Independently verify the theoretical raw error/erasure guarantee.
    for n in (4,6,8):
        for kind,tail,e,r in [('flip',2,1,0),('erasure',1,0,1)]:
            word=(1,)*n+(0,)*tail
            require(codebook_min_error_correct(n,word,e,r),'raw oracle distance theorem failed')
            codes={f:raw_codebook(n,f,word) for f in (0,1)}
            for f in (0,1):
                for trace in codes[f]:
                    for t in range(len(trace)):
                        rcv=list(trace)
                        rcv[t]=(rcv[t]^1 if kind=='flip' else -1)
                        score={ff:min(sum(a!=-1 and a!=b for a,b in zip(rcv,c)) for c in codes[ff]) for ff in (0,1)}
                        require(score[f]<score[1-f],'exact decoder failed on corrupted raw state')
    for n in expected_ns:
        require(abs(analysis['observability_controls'][str(n)]['fixed_remaining_phase_entropy_bits'] - residual_phase_entropy(n))<1e-12,'entropy control changed')
        require(analysis['structural_controls']['unseen_marker_relabelings'][str(n)] is True,'cyclic relabel control')
    for f in analysis['raw_fault_experiments']:
        require(0 <= f['mean_accuracy'] <= 1 and 0 <= f['mean_worst_case_phase_success'] <= 1,'fault rates invalid')
        require(0<=f['successful_seeds']<=5,'fault seed count invalid')
        require(len(f['per_seed_accuracy'])==5,'fault seed count mismatch')
        require(abs(sum(f['per_seed_accuracy'])/5-f['mean_accuracy'])<1e-9,'fault aggregate mismatch')
        if f['architecture']=='raw':
            require(f['successful_seeds']==5,'reported raw-likelihood failure')
    return {'fit_count':len(runs),'verified_panels':checked,
            'verified_aggregates':len(metrics),'fault_summary_groups':len(analysis['raw_fault_experiments']),
            'frozen_source_claim':'pure standard library checks oracle and reported scores, not training'}


def verify() -> dict:
    manifest=json.loads((RESULTS/'oic15_manifest.json').read_text())
    require(manifest['tag']=='OIC-15-CPU-01','manifest tag mismatch')
    for filename,digest in manifest['source_sha256'].items():
        require(sha256((HERE/filename).read_bytes())==digest,'source file modified: '+filename)
    analysis_raw=(RESULTS/'oic15_analysis.json').read_bytes()
    require(sha256(analysis_raw)==manifest['analysis_sha256'],'analysis SHA mismatch')
    analysis=json.loads(analysis_raw)
    trace=load_archive(RESULTS/manifest['files']['traces']['path'],manifest['files']['traces'])
    weights=load_archive(RESULTS/manifest['files']['weights']['path'],manifest['files']['weights'])
    require(len(weights['runs'])==len(trace['runs']),'weight-trace fit-count mismatch')
    a=sorted((r['architecture'],r['split'],r['seed']) for r in trace['runs'])
    b=sorted((r['architecture'],r['split'],r['seed']) for r in weights['runs'])
    require(a==b,'weight-to-trace keys mismatch')
    result=validate_panels(trace,analysis)
    return {**result,'source_integrity':'pass','receipt_integrity':'pass',
            'independent_oracle':'pass','raw_fault_oracle':'pass'}


def main():
    print(json.dumps(verify(),indent=2,sort_keys=True))

if __name__=='__main__':main()
