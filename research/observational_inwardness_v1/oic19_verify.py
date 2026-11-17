"""OIC-19 independent receipt verifier, using only the standard library.

Verifies all archived family classifications, Brier scores, passive-prior
controls, kernel metrics, error-position aggregates, and source hashes.
Reconstructs selected predictions through OIC-18's independent scalar filter
instead of the training-side PyTorch implementation.
"""
from __future__ import annotations
import base64,gzip,hashlib,json,math
from pathlib import Path
from collections import Counter
from oic18_verify import filter_posterior,oracle
from oic19_identifiability import first_n_pulse_probability,distributions

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
MANIFEST=RESULTS/'oic19_summary.json'
FULL=RESULTS/'oic19_full.json.gz.b64'
FAULTS=RESULTS/'oic19_faults_predictive.json'


def require(v,msg):
    if not v:raise ValueError(msg)


def load_records(check_sources=True):
    m=json.loads(MANIFEST.read_text())
    if check_sources:
        for name,digest in m['source_sha256'].items():
            p=HERE/name
            require(p.is_file(),'missing source '+name)
            require(hashlib.sha256(p.read_bytes()).hexdigest()==digest,'source drift '+name)
    packed=base64.b64decode(FULL.read_text().strip(),validate=True)
    require(hashlib.sha256(packed).hexdigest()==m['full_archive_gzip_sha256'],'compressed receipt hash drift')
    decompressed=gzip.decompress(packed)
    require(hashlib.sha256(decompressed).hexdigest()==m['full_archive_decoded_sha256'],'decoded receipt hash drift')
    data=json.loads(decompressed)
    require(len(data['runs'])==40,'must have ten fits per objective')
    groups=Counter((r['variant'],r['split']) for r in data['runs'])
    require(set(groups.values())=={5} and len(groups)==8,'incomplete 8 experiment groups')
    for group in groups:
        require({r['seed'] for r in data['runs'] if (r['variant'],r['split'])==group}==set(range(5)),
                'seed coverage invalid')
    return m,data['runs']


def check_record(r):
    require(r['variant'] in ('baseline_family_bce','entropy','predictive','phase_supervised'),'variant invalid')
    require(r['seed'] in range(5) and r['split'] in ('full','heldout'),'seed/split invalid')
    require(set(r['weights'])=={'kernel_logits','marker_logit','concentration_raw','noise_logit'},'weights missing')
    k=r['weights']['kernel_logits']
    require(len(k)==2 and all(len(row)==5 for row in k),'transition kernel shape')
    kk=[]
    for row in k:
        scale=max(row);p=[math.exp(v-scale) for v in row];s=sum(p);kk.append([x/s for x in p])
    require(abs(r['kernels']['rotating_step_mass']-max(kk[1][1],kk[1][3]))<1e-5,'wrong rotating generator mass')
    require(abs(r['kernels']['fixed_step_mass']-kk[0][2])<1e-5,'wrong fixed generator mass')
    for n in (4,5,6,7,8,12,16,24,32,64):
        panel=r['panels'][str(n)]['full_active']
        probabilities=panel['probabilities'];require(len(probabilities)==2*n,'phase count missing')
        accuracy=sum((float(p)>.5)==(i>=n) for i,p in enumerate(probabilities))/(2*n)
        brier=sum((float(p)-int(i>=n))**2 for i,p in enumerate(probabilities))/(2*n)
        require(abs(accuracy-panel['accuracy'])<1e-6,'wrong family accuracy')
        require(abs(brier-panel['brier'])<1e-6,'wrong Brier')
        require(0<=panel['rotating_phase_MAP_gauge_accuracy']<=1,'phase MAP out of range')
    for n in (8,24,64):
        passive=r['panels'][str(n)]['passive']
        require(passive['accuracy']==.5,'passive family leakage')
        require(passive['mean_passive_deviation']<1e-4,'passive posterior deviates from prior')
    if r['variant']!='baseline_family_bce':
        require(r['training']['steps']==500 and r['training']['batch']==128,'training budget mismatch')
        require(r['training']['parameters']==13,'parameter count wrong')
        require(r['training']['supplied_hidden_phase_in_training']==(r['variant']=='phase_supervised'),
                'privileged phase label metadata wrong')
    return True


def check_independent_replays(records):
    replayed=0
    for r in records:
        if r['seed']!=0:continue
        for n in (8,24,64):
            for f,z in ((0,0),(0,n-2),(1,0),(1,n-2)):
                word='1'*n
                observed=filter_posterior(n,word,oracle(n,f,z,word),'von_mises',r['weights'])
                stated=r['panels'][str(n)]['full_active']['probabilities'][f*n+z]
                require(abs(observed-stated)<2e-4,
                        f'independent raw posterior mismatch {r["variant"]} {r["split"]} n={n}, f={f}, z={z}: {observed}/{stated}')
                replayed+=1
    return replayed


def fault_audit():
    obj=json.loads(FAULTS.read_text())
    require(obj['phase_counts']==[8,24,64],'fault phase-count drift')
    require(len(obj['runs'])==10,'ten predictive fault fits required')
    keys={(r['variant'],r['split'],r['seed']) for r in obj['runs']}
    require(keys=={('predictive',s,k) for s in ('full','heldout') for k in range(5)},'fault fit missing')
    total=0
    for r in obj['runs']:
        for n in (8,24,64):
            suite=r['panels'][str(n)]
            for kind in ('flip','erase'):
                p=suite[kind];B=2*n;P=n+3
                require(p['n']==n and p['type']==kind,'fault panel metadata changed')
                require(p['horizon']==n+2 and p['fault_positions']==P,'fault horizon changed')
                require(p['all_episodes']==B*P,'fault denominator changed')
                require(p['correct_episodes']==sum(p['per_family_phase_correct_counts']),'phase fault sum wrong')
                require(abs(p['correct_episodes']/p['all_episodes']-p['accuracy'])<1e-7,'fault accuracy wrong')
                require(len(p['position_accuracies'])==P,'fault positions missing')
                require([x['position'] for x in p['position_accuracies']]==list(range(P)),'fault positions unordered')
                require(abs(sum(x['accuracy'] for x in p['position_accuracies'])/P-p['accuracy'])<1e-7,'position means incorrect')
                total+=p['all_episodes']
    return total


def independent_fault_replay(records):
    """Recompute selected full fault-position accuracies through the scalar oracle."""
    fault=json.loads(FAULTS.read_text())
    bykey={(r['variant'],r['split'],r['seed']):r for r in fault['runs']}
    good=[]
    count=0
    for split in ('full','heldout'):
        model=next(r for r in records if (r['variant'],r['split'],r['seed'])==('predictive',split,0))
        original=bykey[('predictive',split,0)]
        for n in (8,24,64):
            word='1'*n+'00'
            # Complete re-enumeration at each chosen fault position, both
            # families and all hidden phases. Includes hardest boundary faults.
            positions=(1,n-2) if n<64 else (1,)
            for pos in positions:
                success=0
                for f in (0,1):
                    for z in range(n):
                        raw=oracle(n,f,z,word)
                        raw[pos]^=1
                        p=filter_posterior(n,word,raw,'von_mises',model['weights'])
                        success+=int((p>.5)==bool(f))
                        count+=1
                stored=original['panels'][str(n)]['flip']['position_accuracies'][pos]['accuracy']
                require(abs(success/(2*n)-stored)<1e-7,
                        f'fault position replay mismatch split={split} n={n} pos={pos}')
                good.append((split,n,pos))
    return {'fault_receipts_independently_replayed':len(good),
            'fault_episodes_independently_replayed':count}


def verify():
    m,records=load_records()
    for r in records:check_record(r)
    independent=check_independent_replays(records)
    total=fault_audit()
    fault_replay=independent_fault_replay(records)
    # Do not attach phase-label interpretability claims to the invariant observables.
    for n in (5,8,12):
        all_units=[d for d in range(n) if math.gcd(n,d)==1]
        common=distributions(n,1,(1,)*n)
        require(all(distributions(n,d,(1,)*n)==common for d in all_units),'unit-step observational gauge')
        require(first_n_pulse_probability(n,1)==1,'pulse probability for unit step')
    return {'status':'oic19_verified','fits':len(records),
      'independent_scalar_posteriors_replayed':independent,
      'fault_episodes_audited':total,**fault_replay,
      'lambda_selected':m['lambda_validation']['chosen_lambda'],
      'passed_clean_joint_gates':{f"{r['variant']}:{r['split']}":r['joint_gate_pass_count'] for r in m['aggregates']}}


if __name__=='__main__':print(json.dumps(verify(),indent=2,sort_keys=True))
