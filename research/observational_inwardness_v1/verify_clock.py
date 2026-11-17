"""OIC-13: standard-library verification of archived structural-clock results.

Recompute ground-truth finite trajectories, train/test split metadata,
whole-trace metrics and cross-class Hamming distances from saved outputs.
The verifier does not independently reproduce PyTorch optimizer weights.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SUMMARY=HERE/'results'/'oic13_summary.json'
ARCHIVE=HERE/'results'/'oic13_full.json.gz.b64'
NS=(4,5,6)
SPLITS=('full','heldout')
FAMILIES=('fixed','rotating')
ARCHITECTURES=('reflection_tied','oriented')
SEEDS=tuple(range(5))


def check(condition,message):
    if not condition:raise ValueError(message)


def oracle(n,family,phase,word):
    check(family in FAMILIES,'unknown family')
    x=0
    z=phase
    trace=[x]
    for bit in word:
        check(bit in '01','bad action bit')
        x^=1^(int(bit=='1')*int(z==0))
        z=(z+int(family=='rotating'))%n
        trace.append(x)
    return trace


def distance(a,b):
    check(len(a)==len(b),'mismatched trace lengths')
    return sum(x!=y for x,y in zip(a,b))


def min_cross(a,b):
    return min(distance(u,v) for u in a for v in b)


def schedules(n):
    # Only deterministic schedules known analytically are checked here.
    return {
        'latch':'1'*n+'0'*n,
        'latch_long':'1'*n+'0'*(2*n),
        'always_probe':'1'*(2*n),
        'never_probe':'0'*(2*n),
        'alternating':''.join('1' if i%2==0 else '0' for i in range(2*n))
    }


def decode_archive(raw:bytes):
    compressed=base64.b64decode(raw.strip(),validate=True)
    plain=gzip.decompress(compressed)
    return json.loads(plain),plain,compressed


def validate_payload(payload):
    check(payload['protocol']['tag']=='OIC-13-STRUCTURAL-CLOCK-01','protocol mismatch')
    check(set(payload['protocol']['architectures'])==set(ARCHITECTURES),'architecture mismatch')
    runs=payload['runs']
    check(len(runs)==60,'expected 60 paired runs')
    ids={(r['n'],r['architecture'],r['training_split'],r['seed']) for r in runs}
    check(len(ids)==len(runs),'duplicate runs')
    check(ids=={(n,a,s,seed) for n in NS for a in ARCHITECTURES for s in SPLITS for seed in SEEDS},'incomplete coverage')
    for run in runs:
        n=run['n']; arch=run['architecture']; split=run['training_split']
        check(set(run['evaluations'])=={'latch','latch_long','always_probe','never_probe','alternating','random_budget'},'missing eval schedules')
        for family in FAMILIES:
            train=run['training'][family]
            expected=list(range(n)) if split=='full' else [p for p in range(n) if p not in (1,n-2)]
            check(train['train_phase_indices']==expected,'training phase split wrong')
            check(train['heldout_phase_indices']==[1,n-2],'heldout phase split wrong')
            check(train['train_steps']==350 and train['train_batch_size']==96,'training budget drift')
            check(train['train_rollout_length']==2*n+6,'training trajectory length drift')
            check(train['trainable_parameters']==(2 if arch=='reflection_tied' else 3),'parameter count mismatch')
            check(train['final_teacher_loss']>=0,'invalid loss')
            names=set(train['learned_values'])
            check(names==({'w_cos','bias'} if arch=='reflection_tied' else {'w_cos','w_sin','bias'}),'unexpected weight names')
        for name,evaluation in run['evaluations'].items():
            word=evaluation['word'];check(len(word)>0 and set(word)<={'0','1'},'bad action word')
            if name in schedules(n):check(word==schedules(n)[name],'schedule mismatch')
            if name=='random_budget':check(len(word)==2*n and word.count('1')==n,'budget mismatch')
            for family in FAMILIES:
                data=evaluation['families'][family]
                predictions=data['predicted_state_traces']
                check(len(predictions)==n,'wrong phase count')
                check(all(len(t)==len(word)+1 and t[0]==0 and set(t)<={0,1} for t in predictions),'wrong prediction traces')
                reference=[oracle(n,family,z,word) for z in range(n)]
                check(data['exact_state_traces']==reference,'oracle mismatch')
                flags=[int(p==r) for p,r in zip(predictions,reference)]
                check(data['whole_trace_exact_by_phase']==flags,'whole-trace flags mismatch')
                wrong=sum(distance(p,r) for p,r in zip(predictions,reference))
                check(data['total_wrong_raw_state_samples']==wrong,'wrong bit error count')
            fixed=evaluation['families']['fixed']
            rotated=evaluation['families']['rotating']
            check(evaluation['learned_code_distance']==min_cross(fixed['predicted_state_traces'],rotated['predicted_state_traces']),'empirical code distance mismatch')
            check(evaluation['oracle_code_distance']==min_cross(fixed['exact_state_traces'],rotated['exact_state_traces']),'oracle code distance mismatch')
            for key,detail in [('predicted_state_traces','learned'),('exact_state_traces','oracle')]:
                overlap=bool({row[-1] for row in fixed[key]}&{row[-1] for row in rotated[key]})
                check(evaluation[f'{detail}_terminal_overlap']==overlap,'terminal overlap mismatch')
            if name=='latch' and n%2==0:
                check(evaluation['oracle_code_distance']==n+1,'even oracle latch mismatch')
    summaries=[]
    for n in NS:
        for split in SPLITS:
            for arch in ARCHITECTURES:
                group=sorted([r for r in runs if (r['n'],r['training_split'],r['architecture'])==(n,split,arch)],key=lambda r:r['seed'])
                latch=[r['evaluations']['latch'] for r in group]
                distances=[r['learned_code_distance'] for r in latch]
                row={
                  'n':n,'split':split,'architecture':arch,
                  'models':10,'seeds':list(SEEDS),
                  'trainable_parameters_per_model':2 if arch=='reflection_tied' else 3,
                  'latch_exact_fixed':sum(sum(r['families']['fixed']['whole_trace_exact_by_phase']) for r in latch)/(5*n),
                  'latch_exact_rotating':sum(sum(r['families']['rotating']['whole_trace_exact_by_phase']) for r in latch)/(5*n),
                  'latch_unseen_fixed':sum(sum(bit for z,bit in enumerate(r['families']['fixed']['whole_trace_exact_by_phase']) if z in (1,n-2)) for r in latch)/10,
                  'latch_unseen_rotating':sum(sum(bit for z,bit in enumerate(r['families']['rotating']['whole_trace_exact_by_phase']) if z in (1,n-2)) for r in latch)/10,
                  'latch_code_distances':distances,
                  'oracle_distances':[r['oracle_code_distance'] for r in latch],
                  'noise_free_identifiable_runs':sum(d>=1 for d in distances),
                  'one_bit_error_robust_runs':sum(d>=3 for d in distances),
                  'all_phase_faithful_runs':sum(all(sum(r['families'][family]['whole_trace_exact_by_phase'])==n for family in FAMILIES) for r in latch),
                }
                summaries.append(row)
    return summaries


def verify(manifest_path=SUMMARY,archive_path=ARCHIVE,source_path=None):
    summary=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    payload,plain,compressed=decode_archive(Path(archive_path).read_bytes())
    check(hashlib.sha256(plain).hexdigest()==summary['decoded_sha256'],'decoded receipt digest mismatch')
    check(hashlib.sha256(compressed).hexdigest()==summary['gzip_sha256'],'compressed digest mismatch')
    check(summary['model_fits']==120 and summary['pair_runs']==60,'fit count mismatch')
    if source_path is None:source_path=HERE/'clock_pilot.py'
    check(hashlib.sha256(Path(source_path).read_bytes()).hexdigest()==summary['source_sha256'],'source changed since run')
    actual=validate_payload(payload)
    check(actual==summary['summary'],'aggregate summary mismatch')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    result=verify()
    print('OIC-13 verified:',result['model_fits'],'trained models,',result['pair_runs'],'paired runs')
    for row in result['summary']:
        print(f"n={row['n']} {row['split']} {row['architecture']}: "
              f"accuracy={row['latch_exact_fixed']:.1%}/{row['latch_exact_rotating']:.1%} "
              f"heldout={row['latch_unseen_fixed']:.1%}/{row['latch_unseen_rotating']:.1%} "
              f"d={row['latch_code_distances']}")

if __name__=='__main__':main()
