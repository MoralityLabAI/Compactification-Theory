"""OIC-16 independent, torch-free receipt verification and finite-state reference.

Recomputes model-family likelihoods in double precision from archived fitted
weights and re-simulates each observable history, including raw substitutions,
marked erasures, gauge-shift controls and the passive prior gate.

This verifies predictions and metrics from *weights*, not optimizer provenance.
The optional retraining command requires PyTorch/NumPy separately.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

from oic16_projection import projected_steps, projected_marker, projected_sets, oracle_sets, panel as projection_panel

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE/'results'/'oic16_full.json.gz.b64'
MANIFEST = HERE/'results'/'oic16_summary.json'
SHIFTS = (-2,-1,0,1,2)
TRAIN_NS = (4,5,6)
OOD_NS = (7,8)
SEEDS = range(5)
SPLITS = ('full','heldout')
AUGMENTATIONS = ('clean','raw_fault_aug')
SCHEDULES = ('passive','full_active','latch','extended_latch','alternating','random_budget')
# Fixed PCG64 OIC-15 schedule words, checked against the original NumPy generator.
RANDOM_WORDS = {4:'01100011', 5:'1011010010', 6:'100010101101',
                7:'10000010011111', 8:'0001111001010101'}


def check(cond, msg):
    if not cond:
        raise ValueError(msg)


def stable_softmax(xs):
    peak=max(xs)
    exp=[math.exp(x-peak) for x in xs]
    den=sum(exp)
    return [x/den for x in exp]


def sigmoid(x):
    if x>=0:
        return 1/(1+math.exp(-x))
    a=math.exp(x)
    return a/(1+a)


def oracle(n, f, z, word, marker=0):
    x=0
    trace=[x]
    for ch in word:
        u=int(ch)
        x ^= 1 ^ (u & int(z == marker))
        z=(z+f)%n
        trace.append(x)
    return trace


def word_for(n,name):
    if name=='passive':return '0'*(2*n)
    if name=='full_active':return '1'*n
    if name=='latch':return '1'*n+'0'*n
    if name=='extended_latch':return '1'*n+'0'*(n+2)
    if name=='alternating':return ''.join(str(int(i%2==0)) for i in range(2*n))
    if name=='random_budget':
        return RANDOM_WORDS[n]
    raise ValueError(name)


def filter_episode(n,word,raw,weights):
    """Reference implementation of the learned 2n-by-2-state belief update."""
    check(len(raw)==len(word)+1,'bad raw trace length')
    check(set(word)<=set('01'),'bad probes')
    check(set(raw)<={-1,0,1},'bad raw readout')
    logits=weights['kernel_logits']
    coeff=weights['emission_coeff']
    noise=float(weights['noise_logit'])
    check(len(logits)==2 and all(len(row)==5 for row in logits),'bad kernel shape')
    check(len(coeff)==3,'bad emission')
    kernels=[stable_softmax(row) for row in logits]
    phase_emission=[sigmoid(coeff[0]+coeff[1]*math.cos(2*math.pi*z/n)
                            +coeff[2]*math.cos(4*math.pi*z/n)) for z in range(n)]
    pobs=sigmoid(noise)*.98
    mass=[[[1/(2*n),0.] for _ in range(n)] for _ in (0,1)]
    def condition(m,observed):
        if observed==-1:return m
        for fam in range(2):
            for z in range(n):
                m[fam][z][0]*=1-pobs if observed==0 else pobs
                m[fam][z][1]*=1-pobs if observed==1 else pobs
        den=sum(m[fam][z][v] for fam in range(2) for z in range(n) for v in (0,1))
        check(den>0,'underflow or zero raw likelihood')
        return [[[v/den for v in m[fam][z]] for z in range(n)] for fam in range(2)]
    mass=condition(mass,raw[0])
    for t,ch in enumerate(word):
        u=int(ch)
        evolved=[[[0.,0.] for _ in range(n)] for _ in range(2)]
        for fam in range(2):
            for z in range(n):
                pulse=phase_emission[z]*u
                pre0=mass[fam][z][0]*pulse+mass[fam][z][1]*(1-pulse)
                pre1=mass[fam][z][1]*pulse+mass[fam][z][0]*(1-pulse)
                for i,shift in enumerate(SHIFTS):
                    dest=(z+shift)%n
                    evolved[fam][dest][0]+=kernels[fam][i]*pre0
                    evolved[fam][dest][1]+=kernels[fam][i]*pre1
        mass=condition(evolved,raw[t+1])
    family_post=sum(sum(phase) for phase in mass[1])
    return family_post,mass


def report_panel(n,word,weights,marker=0,flip=None,erase=None):
    ps=[]; entr=[]; phase_mass=[]
    rot_left=weights['kernel_logits'][1][1] > weights['kernel_logits'][1][3]
    for f in (0,1):
        for z in range(n):
            raw=oracle(n,f,z,word,marker)
            if flip is not None:raw[flip]^=1
            if erase is not None:raw[erase]=-1
            p,posterior=filter_episode(n,word,raw,weights)
            ps.append(p)
            ent=-sum(v*math.log2(v) for mat in posterior for phase in mat for v in phase if v>0)
            entr.append(ent)
            if f==1:
                index=(-z if rot_left else z)%n
                family_mass=sum(sum(phase) for phase in posterior[1])
                phase_mass.append(sum(posterior[1][index])/max(family_mass,1e-30))
    y=[f for f in (0,1) for _ in range(n)]
    return {
        'probabilities':ps,
        'accuracy':sum(int((p>.5)==(f==1)) for p,f in zip(ps,y))/len(y),
        'brier':sum((p-f)**2 for p,f in zip(ps,y))/len(y),
        'nll':sum(-math.log(max(1e-40,p if f else 1-p)) for p,f in zip(ps,y))/len(y),
        'mean_passive_deviation':sum(abs(p-.5) for p in ps)/len(ps),
        'mean_joint_state_entropy':sum(entr)/len(entr),
        'rotating_true_phase_mass':sum(phase_mass)/len(phase_mass),
    }


def verify_panel(got,expected,ctx):
    for k in ('accuracy','brier','nll','mean_passive_deviation','mean_joint_state_entropy','rotating_true_phase_mass'):
        check(abs(float(got[k])-float(expected[k]))<1e-4, f'{ctx}: {k} disagrees ({got[k]} != {expected[k]})')
    check(len(got['probabilities'])==len(expected['probabilities']),f'{ctx}: prob count')
    for i,(a,b) in enumerate(zip(got['probabilities'],expected['probabilities'])):
        check(abs(a-b)<1e-5, f'{ctx}: prob[{i}] mismatch ({a} vs {b})')


def validate_run(run):
    n_keys={str(n) for n in TRAIN_NS+OOD_NS}
    check(set(run['panels'])==n_keys,'phase-count coverage')
    weights=run['weights']
    check(set(weights)=={'kernel_logits','emission_coeff','noise_logit'},'weight keys')
    check(run['training']['parameters']==14,'parameter count')
    check(run['training']['steps']==500 and run['training']['batch']==128,'training budget')
    check(run['training']['split']==run['split'],'training split')
    check(run['training']['noise_augmentation']==run['augment'],'augmentation metadata')
    check(projected_steps(weights)[0]==0 and abs(projected_steps(weights)[1])==1,'learned generator projection')
    for n in TRAIN_NS+OOD_NS:
        suites=run['panels'][str(n)]
        check(set(suites)==set(SCHEDULES)|{'relabel_control'}|({'one_flip_by_position','one_erase_by_position'} if n%2==0 else set()),'panel coverage')
        for name in SCHEDULES:
            word=word_for(n,name)
            if name=='random_budget':
                check(len(word)==2*n and word.count('1')==n, 'random probe budget drift')
            actual=report_panel(n,word,weights)
            verify_panel(suites[name],actual,f"seed={run['seed']} n={n} {name}")
            if name=='passive':
                check(actual['accuracy']==.5,'passive classification leak')
                check(actual['mean_passive_deviation']<1e-4,'passive posterior leak')
            if name=='full_active':
                check(actual['accuracy']==1.,'unexpected clean identification failure')
        changed=report_panel(n,'1'*n,weights,marker=2%n)
        standard=suites['full_active']
        gap=max(abs(a-b) for f in (0,1)
                for a,b in zip(sorted(changed['probabilities'][f*n:(f+1)*n]),sorted(standard['probabilities'][f*n:(f+1)*n])))
        control=suites['relabel_control']
        check(abs(control['prob_set_gap']-gap)<1e-5 and abs(control['accuracy']-changed['accuracy'])<1e-5, 'phase relabel control mismatch')
        if n%2==0:
            word='1'*n+'00'
            for fault in ('flip','erase'):
                expected=suites['one_flip_by_position' if fault=='flip' else 'one_erase_by_position']
                check(len(expected)==n+3,'fault-position coverage')
                for j in range(n+3):
                    actual=report_panel(n,word,weights,**{fault:j})['accuracy']
                    check(abs(expected[j]-actual)<1e-5,f'fault panel mismatch n={n} pos={j} kind={fault}')
    return True


def load_receipt(archive=ARCHIVE, manifest=MANIFEST):
    meta=json.loads(manifest.read_text())
    compressed=base64.b64decode(archive.read_text().strip(),validate=True)
    check(hashlib.sha256(compressed).hexdigest()==meta['archive_sha256'],'archive compressed hash mismatch')
    payload=gzip.decompress(compressed)
    check(hashlib.sha256(payload).hexdigest()==meta['receipt_sha256'],'decoded payload hash mismatch')
    data=json.loads(payload)
    check(len(data['runs'])==20,'expected twenty learned observers')
    combo={(x['augment'],x['split'],x['seed']) for x in data['runs']}
    check(combo=={(a,s,k) for a in AUGMENTATIONS for s in SPLITS for k in SEEDS},'incomplete seed/fault split matrix')
    for name, digest in meta['source_sha256'].items():
        current=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        check(current==digest,f'{name} source hash mismatch')
    return meta,data


def validate_summary(meta, data):
    for augment in AUGMENTATIONS:
        for split in SPLITS:
            sub=[r for r in data['runs'] if r['augment']==augment and r['split']==split]
            for n in TRAIN_NS+OOD_NS:
                for schedule in SCHEDULES:
                    mean=sum(r['panels'][str(n)][schedule]['accuracy'] for r in sub)/len(sub)
                    expected=meta['summary'][augment][split][str(n)][schedule]['accuracy']
                    check(abs(mean-expected)<1e-10,'summary accuracy mismatch')
                if n%2==0:
                    for key in ('one_flip_by_position','one_erase_by_position'):
                        mean=sum(sum(r['panels'][str(n)][key])/len(r['panels'][str(n)][key]) for r in sub)/len(sub)
                        actual=meta['summary'][augment][split][str(n)][key]['accuracy']
                        check(abs(mean-actual)<1e-10,'fault summary mismatch')
    return True


def verify():
    meta,data=load_receipt()
    for run in data['runs']:
        validate_run(run)
    validate_summary(meta,data)
    for run in data['runs']:
        for n in (4,6,8,10,12):
            weights=run['weights']
            check(projected_marker(weights,n)==0, 'projected emission gauge')
            word=(1,)*n+(0,)*2
            check(projected_sets(n,word,weights)==oracle_sets(n,word),'projected codebook differs from oracle')
            for j in range(n+3):
                check(projection_panel(n,word,weights,flip_position=j)['accuracy']==1.,'projected flip failure')
                check(projection_panel(n,word,weights,erase_position=j)['accuracy']==1.,'projected erase failure')
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    meta=verify()
    print(f"OIC-16 independent reference OK: {meta['run_count']} trained fits, "
          f"{meta['full_panel_count']} clean panels, {meta['fault_case_count']} raw fault-position panels")


if __name__=='__main__':main()
