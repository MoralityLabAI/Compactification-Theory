"""OIC-18 independent CPU receipt validator (Python standard library only).

Reconstructs selected posterior probabilities from fitted weights with a separate
implementation of the learned stochastic hidden-state filter; checks all
recorded accuracy/Brier/passive gates and source/archive hashes.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json,math
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'results/oic18_summary.json'
ARCHS=('temperature','von_mises','harmonic3')
SPLITS=('full','heldout')
AUGS=('clean','raw_fault_aug')
SEEDS=range(5)
NS=(4,5,6,7,8,12,16,24,32,64)
SHIFTS=(-2,-1,0,1,2)


def require(c,msg):
    if not c:raise ValueError(msg)


def sigmoid(x):
    if x>=0:return 1/(1+math.exp(-x))
    y=math.exp(x)
    return y/(1+y)


def softplus(x):return math.log1p(math.exp(-abs(x)))+max(0,x)


def emission(n,arch,w):
    require(arch in ARCHS and n>=2,'invalid emission arch/n')
    if arch=='von_mises':
        a=float(w['marker_logit']); k=softplus(float(w['concentration_raw']))*(n/5)**2
        return [sigmoid(a-k*(1-math.cos(2*math.pi*z/n))) for z in range(n)]
    c=w['emission_coeff']
    if arch=='temperature':
        s=sigmoid(float(w['temperature_raw']))
        tau=1+s*((n/5)**2-1)
        return [sigmoid(c[0]+tau*(c[1]*(math.cos(2*math.pi*z/n)-1)
                      +c[2]*(math.cos(4*math.pi*z/n)-1))) for z in range(n)]
    return [sigmoid(c[0]+c[1]*math.cos(2*math.pi*z/n)+c[2]*math.cos(6*math.pi*z/n)) for z in range(n)]


def softmax(vals):
    vmax=max(vals);exps=[math.exp(x-vmax) for x in vals];den=sum(exps)
    return [v/den for v in exps]


def oracle(n,f,z,word,marker=0):
    x=0;raw=[0]
    for a in word:
        require(a in '01','invalid probe word')
        x ^= 1 ^ (int(a)&int(z==marker))
        z=(z+f)%n
        raw.append(x)
    return raw


def filter_posterior(n,word,raw,arch,w):
    """Standalone scalar Bayesian filter; no torch/NumPy imports."""
    require(len(raw)==len(word)+1,'incompatible raw channel')
    k=[softmax(row) for row in w['kernel_logits']]
    g=emission(n,arch,w)
    pobs=sigmoid(float(w['noise_logit']))*.98
    m=[[[1/(2*n),0.] for z in range(n)] for f in (0,1)]
    def condition(current,observed):
        if observed==-1:return current
        for f in (0,1):
            for z in range(n):
                current[f][z][0]*=(1-pobs if observed==0 else pobs)
                current[f][z][1]*=(1-pobs if observed==1 else pobs)
        den=sum(sum(pair) for family in current for pair in family)
        require(den>0,'underflow in independent filter')
        return [[[v/den for v in pair] for pair in family] for family in current]
    m=condition(m,raw[0])
    for t,ch in enumerate(word):
        u=int(ch)
        next_mass=[[[0.,0.] for _ in range(n)] for _ in (0,1)]
        for f in (0,1):
            for z in range(n):
                pulse=g[z]*u
                v0=m[f][z][0]*pulse+m[f][z][1]*(1-pulse)
                v1=m[f][z][1]*pulse+m[f][z][0]*(1-pulse)
                for j,step in enumerate(SHIFTS):
                    dst=(z+step)%n
                    next_mass[f][dst][0]+=k[f][j]*v0
                    next_mass[f][dst][1]+=k[f][j]*v1
        m=condition(next_mass,raw[t+1])
    return sum(sum(pair) for pair in m[1])


def word_for(n,name):
    if name=='full_active':return '1'*n
    if name=='passive':return '0'*(2*n)
    if name=='latch':return '1'*n+'0'*n
    if name=='alternate':return ''.join(str(int(t%2==0)) for t in range(2*n))
    raise ValueError(name)


def visible_permutation_invariant(n,word,shift,permutation):
    """Relabel *both* transition and marker, then check every public trace.

    This is a genuine hidden-coordinate conjugacy; any arbitrary permutation
    preserves the observation law when every piece of the hidden model is
    consistently transformed. A non-dihedral permutation is included.
    """
    require(sorted(permutation)==list(range(n)),'permutation of n states required')
    inverse=[0]*n
    for i,j in enumerate(permutation):inverse[j]=i
    for f in (0,1):
        for z in range(n):
            x=0;trace=[0]
            zprime=permutation[z]
            for c in word:
                pulse=int(c=='1' and zprime==permutation[shift%n])
                x ^= 1 ^ pulse
                idx=inverse[zprime]
                zprime=permutation[(idx+f)%n]
                trace.append(x)
            expected=oracle(n,f,z,word,marker=shift%n)
            if trace!=expected:return False
    return True


def load_all(manifest=MANIFEST):
    m=json.loads(manifest.read_text())
    for name,sha in m['source_sha256'].items():
        require(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha, f'source drift {name}')
    records=[]
    for entry in m['archives']:
        raw=(HERE/entry['file']).read_text().strip()
        packed=base64.b64decode(raw,validate=True)
        require(hashlib.sha256(packed).hexdigest()==entry['gzip_sha256'],'compressed digest mismatch')
        decoded=gzip.decompress(packed)
        require(hashlib.sha256(decoded).hexdigest()==entry['decoded_sha256'],'decoded digest mismatch')
        piece=json.loads(decoded)
        require(len(piece['runs'])==15,'expected 15 fits per condition')
        require(piece['protocol']['train_budget_steps_per_fit']==500,'training budget change')
        records.extend(piece['runs'])
    require(len(records)==60,'expected 60 fitted new models')
    combos={(r['architecture'],r['augmentation'],r['split'],r['seed']) for r in records}
    require(combos=={(a,g,s,k) for a in ARCHS for g in AUGS for s in SPLITS for k in SEEDS},'incomplete sample lattice')
    return m,records


def verify_record_fast(r):
    require(r['training']['steps']==500 and r['training']['batch']==128,'training budget')
    require(r['training']['parameters']=={'temperature':15,'von_mises':13,'harmonic3':14}[r['architecture']], 'parameter count')
    require(set(r['panels'])=={str(n) for n in NS},'eval phase count coverage')
    for n in NS:
        suite=r['panels'][str(n)]
        require(set(suite)=={'full_active','passive','latch','alternate'},'suite mismatch')
        for label,entry in suite.items():
            require(len(entry['probabilities'])==2*n,'phase/family coverage')
            probabilities=entry['probabilities']
            require(all(0<=p<=1 for p in probabilities),'invalid probs')
            labels=[0]*n+[1]*n
            accuracy=sum(int((p>0.5)==y) for p,y in zip(probabilities,labels))/(2*n)
            brier=sum((p-y)**2 for p,y in zip(probabilities,labels))/(2*n)
            deviation=sum(abs(p-.5) for p in probabilities)/(2*n)
            require(abs(accuracy-entry['accuracy'])<1e-7,'recorded accuracy mismatch')
            require(abs(brier-entry['brier'])<1e-7,'recorded Brier mismatch')
            require(abs(deviation-entry['mean_passive_deviation'])<1e-6,'recorded uncertainty mismatch')
            if label=='passive':
                require(abs(accuracy-.5)<1e-10,'passive label leakage')
                require(deviation<2e-4,'passive posterior prior leak')
    return True


def verify_posterior_samples(r,ns=(4,8,24,64)):
    arch,w=r['architecture'],r['weights']
    for n in ns:
        # Fixed selected phases for predictable CI and numerical coverage.
        phases={0,1,n-2,n-1}
        for name in ('full_active','passive'):
            word=word_for(n,name)
            predictions=r['panels'][str(n)][name]['probabilities']
            for f in (0,1):
                for z in sorted(phases):
                    observed=filter_posterior(n,word,oracle(n,f,z,word),arch,w)
                    stated=predictions[f*n+z]
                    require(abs(stated-observed)<2e-4,
                            f'independent posterior mismatch {arch} n={n} family={f} z={z}: {stated} vs {observed}')
    return True



def verify_fault_receipts(manifest,models):
    """Check every archived fault matrix and replay a stratified exact subset."""
    expected_n=(4,6,8,12,16,24,32,64)
    bykey={(r['architecture'],r['augmentation'],r['split'],r['seed']):r for r in models}
    count=0;total_episodes=0;replayed=0
    for item in manifest['fault_archives']:
        packed=base64.b64decode((HERE/item['file']).read_text().strip(),validate=True)
        require(hashlib.sha256(packed).hexdigest()==item['gzip_sha256'],'fault gzip digest')
        decoded=gzip.decompress(packed)
        require(hashlib.sha256(decoded).hexdigest()==item['decoded_sha256'],'fault receipt decoded digest')
        payload=json.loads(decoded)
        require(payload['phase_counts']==list(expected_n),'fault phase count coverage')
        require(len(payload['runs'])==15,'fault fits per group')
        for r in payload['runs']:
            key=r['architecture'],r['augmentation'],r['split'],r['seed']
            require(key in bykey,'fault fit unmatched to saved weights')
            count+=1
            require(set(r['panels'])=={str(n) for n in expected_n},'fault panel coverage')
            for n in expected_n:
                for kind in ('flip','erase'):
                    v=r['panels'][str(n)][kind]
                    require(v['n']==n and v['type']==kind and v['horizon']==n+2,'fault horizon drift')
                    require(v['fault_positions']==n+3,'fault position count drift')
                    require(v['all_episodes']==2*n*(n+3),'fault episode count drift')
                    require(len(v['position_accuracies'])==n+3,'missing position')
                    require([x['position'] for x in v['position_accuracies']]==list(range(n+3)),'position order')
                    correct=v['correct_episodes']
                    require(correct==sum(v['per_family_phase_correct_counts']),'fault phase count consistency')
                    require(abs(v['accuracy']-correct/v['all_episodes'])<1e-7,'wrong fault mean')
                    require(abs(v['worst_fault_position_accuracy']-min(x['accuracy'] for x in v['position_accuracies']))<1e-7,'fault min pos')
                    require(abs(v['worst_family_phase_accuracy']-min(v['per_family_phase_correct_counts'])/(n+3))<1e-7,'fault min phase')
                    require(abs(sum(x['accuracy'] for x in v['position_accuracies'])/(n+3)-v['accuracy'])<1e-7,'fault positions mean')
                    total_episodes+=v['all_episodes']
                    # Independent exact oracle + posterior replay for n=8 across
                    # one seed/architecture/augmentation (both training splits).
                    if n==8 and r['seed']==0 and kind=='flip':
                        w=bykey[key]['weights'];arch=r['architecture']
                        for pos in range(n+3):
                            expected_acc=0
                            for f in (0,1):
                                for z in range(n):
                                    raw=oracle(n,f,z,'1'*n+'00')
                                    raw[pos]^=1
                                    p=filter_posterior(n,'1'*n+'00',raw,arch,w)
                                    expected_acc+=int((p>.5)==(f==1))
                            require(abs(expected_acc/(2*n)-v['position_accuracies'][pos]['accuracy'])<1e-7,
                                    f'independent fault replay n8 pos={pos} {key}')
                        replayed+=1
    require(count==60,'fault number fits mismatch')
    return {'fault_models':count,'fault_episodes':total_episodes,'fault_panels_replayed':replayed}

def verify():
    m,records=load_all()
    for r in records:verify_record_fast(r)
    # Reconstruct 1/5 fits per group across all architectures and both conditions;
    # every other fit still has every stored prediction and score validated.
    for r in records:
        if r['seed']==0:verify_posterior_samples(r)
    for n in (4,5,6,7,8,12):
        perm=list(range(n)); perm[1],perm[2]=perm[2],perm[1]
        require(visible_permutation_invariant(n,'1'*n,0,perm),'non-dihedral relabel gauge violated')
        require(visible_permutation_invariant(n,'1'*n,1,list(reversed(range(n)))),'reflection gauge violated')
    fault_verified=verify_fault_receipts(m,records)
    return {'fits':len(records),**fault_verified,'replayed_fits':sum(r['seed']==0 for r in records),
            'verified_panel_probabilities':sum(len(panel['probabilities']) for r in records for suite in r['panels'].values() for panel in suite.values()),
            'status':'oic18_evidence_verified'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    args=p.parse_args();print(json.dumps(verify(),sort_keys=True))
