"""OIC-14 independent pure-stdlib verification of archived neural pilot.

Recomputes finite oracle trajectories and claimed output metrics, checks fit
parameters and emitted monitor traces, and verifies a condition sufficient
for exact all-horizon dynamics after discrete-group projection. No retraining
occurs in CI and the float64 numerical replica is a check of the saved
parameters, not a formal floating-point certificate.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
SUMMARY=HERE/'results'/'oic14_summary.json'
RECEIPT=HERE/'results'/'oic14_compact_full.json.gz.b64'
NS=(4,5,6)
SEEDS=tuple(range(5))
SPLITS=('full','heldout')
ARCHES=('orthogonal_local','orthogonal_grid','unconstrained')
FAMILIES=('fixed','rotating')
WORDS=('latch','latch_long','always_probe','never_probe','alternating','random_budget')


def require(condition:bool,message:str)->None:
    if not condition:raise ValueError(message)


def exact_trace(n:int,family:str,z0:int,word:str)->list[int]:
    require(family in FAMILIES,'unregistered family')
    z=z0;x=0;answer=[x]
    for ch in word:
        require(ch in '01','nonbinary probe word')
        x ^= 1 ^ (int(ch) & int(z==0))
        z=(z+int(family=='rotating'))%n
        answer.append(x)
    return answer


def predicted_trace(n:int,arch:str,fit:dict,z0:int,word:str)->list[int]:
    vals=fit['learned_values']
    gain=vals['gain'][0];bias=vals['bias'][0]
    theta0=2*math.pi*z0/n
    if arch=='unconstrained':
        m=fit['transition_matrix']
        xh,yh=math.cos(theta0),math.sin(theta0)
    else:
        omega=vals['omega'][0]
    x=0;outputs=[x]
    for t,ch in enumerate(word):
        if arch=='unconstrained':
            projected=xh
        else:
            projected=math.cos(theta0+t*omega)
        event=int(gain*projected+bias>=0)
        x ^= 1 ^ (int(ch) & event)
        outputs.append(x)
        if arch=='unconstrained':
            xh,yh=m[0][0]*xh+m[0][1]*yh,m[1][0]*xh+m[1][1]*yh
    return outputs


def cross_distance(a:list[list[int]],b:list[list[int]])->int:
    require(bool(a) and bool(b),'nonempty codebooks required')
    return min(sum(x!=y for x,y in zip(left,right)) for left in a for right in b)


def nearest_step(n:int,omega:float)->int:
    return round(omega*n/(2*math.pi))%n

def projected_exact(n:int,family:str,fit:dict)->bool:
    vals=fit['learned_values'];gain=vals['gain'][0];bias=vals['bias'][0]
    step=nearest_step(n,vals['omega'][0])
    detector=tuple(gain*math.cos(2*math.pi*z/n)+bias>=0 for z in range(n))
    return step==int(family=='rotating') and detector==(True,)+(False,)*(n-1)


def inspect(payload:dict)->dict:
    require(payload.get('protocol',{}).get('tag')=='OIC-14-LEARNABLE-PHASE-01','incorrect experiment tag')
    runs=payload['runs']
    require(len(runs)==90 and payload.get('model_fits')==180 and payload.get('run_count')==90,'run count drift')
    keys=[(r['n'],r['training_split'],r['architecture'],r['seed']) for r in runs]
    expected={(n,split,arch,seed) for n in NS for split in SPLITS for arch in ARCHES for seed in SEEDS}
    require(set(keys)==expected and len(set(keys))==len(keys),'duplicate or missing runs')
    collected={}
    projected_ok=0;projected_grid_ok=0
    for row in runs:
        n=row['n'];split=row['training_split'];arch=row['architecture']
        require(set(row['training'])==set(FAMILIES),'model-family training mismatch')
        require(set(row['evaluations'])==set(WORDS),'evaluation schedule drift')
        for fam in FAMILIES:
            fit=row['training'][fam]
            phases=list(range(n)) if split=='full' else [z for z in range(n) if z not in (1,n-2)]
            require(fit['train_phase_indices']==phases,'training phase split mismatch')
            require(fit['heldout_phase_indices']==[1,n-2],'heldout phases mismatch')
            require(fit['train_steps']==350 and fit['batch_size']==96 and fit['train_rollout_length']==2*n+6,'training budget drift')
            require(fit['final_teacher_bce']>=0 and fit['initial_teacher_bce']>=0,'negative BCE')
            require(fit['trainable_parameters']==(6 if arch=='unconstrained' else 3),'wrong architecture size')
            require(fit['warmstart_extra_candidate_evaluations']==(4*n if arch=='orthogonal_grid' else 0),'grid candidate accounting mismatch')
            if arch=='orthogonal_grid':
                q=fit['warmstart_omega']
                require(q is not None,'missing warmstart step')
                require(min(abs(q-(2*math.pi*j/(4*n))) for j in range(4*n))<1e-5,'warmstart outside grid')
            elif arch=='orthogonal_local':
                require(fit['warmstart_omega'] is None,'unexpected warmstart')
            require(fit['true_omega']==round(2*math.pi/n if fam=='rotating' else 0.0,8),'oracle angular metadata drift')
            if arch=='unconstrained':
                require(set(fit['learned_values'])=={'matrix','gain','bias'},'dense parameter schema mismatch')
                require(len(fit['learned_values']['matrix'])==4,'matrix element count')
                require(fit['learned_omega'] is None,'dense arch mislabelled')
            else:
                require(set(fit['learned_values'])=={'omega','gain','bias'},'orthogonal parameter schema mismatch')
                require(len(fit['learned_values']['omega'])==1,'omega scalar expected')
                require(abs(fit['learned_omega']-fit['learned_values']['omega'][0])<1e-7,'omega mismatch')
                require(fit['rotation_commutator_defect']==0 and fit['orthogonality_defect']==0,'SO2 invariant broken')
                is_exact=projected_exact(n,fam,fit)
                projected_ok+=int(is_exact)
                if arch=='orthogonal_grid':projected_grid_ok+=int(is_exact)
        entry={}
        for name,evaluation in row['evaluations'].items():
            word=evaluation['word'];require(len(word)>0 and all(ch in '01' for ch in word),'invalid word')
            if name=='latch':require(word=='1'*n+'0'*n,'wrong latch schedule')
            if name=='latch_long':require(word=='1'*n+'0'*(2*n),'wrong long latch')
            if name=='always_probe':require(word=='1'*(2*n),'wrong all-probe schedule')
            if name=='never_probe':require(word=='0'*(2*n),'wrong passive schedule')
            if name=='alternating':require(word==''.join('1' if t%2==0 else '0' for t in range(2*n)),'wrong alternating schedule')
            if name=='random_budget':require(len(word)==2*n and word.count('1')==n,'wrong random-budget schedule')
            require(set(evaluation['predicted'])==set(FAMILIES),'missing family traces')
            results={}
            for fam in FAMILIES:
                traces=evaluation['predicted'][fam]
                require(len(traces)==n,'wrong initial phase coverage')
                for z,saved in enumerate(traces):
                    require(len(saved)==len(word)+1 and saved[0]==0 and set(saved)<={0,1},'nonbinary or incomplete trace')
                    computed=predicted_trace(n,arch,row['training'][fam],z,word)
                    require(saved==computed,'archived emitted trace differs from stored fitted parameters')
                correct=[exact_trace(n,fam,z,word) for z in range(n)]
                results[fam]={'traces':traces,'accuracy':sum(t==o for t,o in zip(traces,correct))/n,
                              'oracle':correct}
            entry[name]={'learned_distance':cross_distance(results['fixed']['traces'],results['rotating']['traces']),
                         'oracle_distance':cross_distance(results['fixed']['oracle'],results['rotating']['oracle']),
                         'accuracy_fixed':results['fixed']['accuracy'],
                         'accuracy_rotating':results['rotating']['accuracy']}
        collected[(n,split,arch,row['seed'])]=entry
    summaries=[]
    for n in NS:
      for split in SPLITS:
       for arch in ARCHES:
        selections=[r for r in runs if r['n']==n and r['training_split']==split and r['architecture']==arch]
        selections.sort(key=lambda r:r['seed'])
        accuracy_f=[];accuracy_r=[];withheld={fam:[] for fam in FAMILIES}
        distances=[];oracle_d=[];omegaR=[];omegaF=[];gridR=[];ortho=[];equiv=[]
        for r in selections:
            row=r['evaluations']['latch'];word=row['word']
            pair={}
            for fam in FAMILIES:
                saved=row['predicted'][fam]
                truth=[exact_trace(n,fam,z,word) for z in range(n)]
                p=[int(a==b) for a,b in zip(saved,truth)]
                pair[fam]=p
                withheld[fam].extend(p[z] for z in (1,n-2))
            accuracy_f.extend(pair['fixed']);accuracy_r.extend(pair['rotating'])
            d=collected[(n,split,arch,r['seed'])]['latch']
            distances.append(d['learned_distance']);oracle_d.append(d['oracle_distance'])
            omegaR.append(r['training']['rotating']['learned_omega']);omegaF.append(r['training']['fixed']['learned_omega'])
            gridR.append(r['training']['rotating']['warmstart_omega'])
            ortho.append(r['training']['rotating']['orthogonality_defect'])
            equiv.append(r['training']['rotating']['rotation_commutator_defect'])
        summaries.append({
            'n':n,'training_split':split,'architecture':arch,'seeds':list(SEEDS),
            'whole_trace_accuracy_fixed':sum(accuracy_f)/len(accuracy_f),
            'whole_trace_accuracy_rotating':sum(accuracy_r)/len(accuracy_r),
            'phase_indices_1_and_n_minus_2_accuracy':{k:sum(v)/len(v) for k,v in withheld.items()},
            'latch_min_distances':distances,'latch_oracle_min_distances':oracle_d,
            'learned_rotation_rad_rotating':omegaR,'learned_rotation_rad_fixed':omegaF,
            'warmstart_grid_rotating_radians':gridR,
            'max_equivariance_residual_rotating':max(equiv),
            'max_orthogonality_defect_rotating':max(ortho),
            'zero_error_identifiable_runs':sum(d>0 for d in distances),
            'one_raw_bit_fault_tolerant_runs':sum(d>=3 for d in distances),
            'exact_oracle_code_distance_runs':sum(a==b for a,b in zip(distances,oracle_d)),
            'trainable_params_fixed':selections[0]['training']['fixed']['trainable_parameters'],
            'trainable_params_rotating':selections[0]['training']['rotating']['trainable_parameters'],
            'extra_scan_candidate_evals_per_family':selections[0]['training']['rotating']['warmstart_extra_candidate_evaluations'],
        })
    return {'summary':summaries,'projected_correct_fit_count':projected_ok,
            'projected_correct_grid_fit_count':projected_grid_ok}


def verify(summary:Path=SUMMARY,archive:Path=RECEIPT,source_hash_check:bool=True)->dict:
    manifest=json.loads(summary.read_text(encoding='utf8'))
    zipped=base64.b64decode(archive.read_text(encoding='ascii').strip(),validate=True)
    raw=gzip.decompress(zipped)
    require(hashlib.sha256(raw).hexdigest()==manifest['full_receipt_sha256'],'uncompressed receipt hash mismatch')
    require(hashlib.sha256(zipped).hexdigest()==manifest['compressed_receipt_sha256'],'compressed receipt hash mismatch')
    if source_hash_check:
        require(hashlib.sha256((HERE/'learnable_clock.py').read_bytes()).hexdigest()==manifest['training_source_sha256'],'training source modified')
        require(hashlib.sha256((HERE/'projection.py').read_bytes()).hexdigest()==manifest['projection_source_sha256'],'projection source modified')
    payload=json.loads(raw)
    stats=inspect(payload)
    require(stats['summary']==manifest['summary'],'summary does not match archived trace reconstruction')
    require(stats['projected_correct_fit_count']==manifest['full_trace_projected_correct_fit_count'],'projection evidence mismatch')
    require(stats['projected_correct_grid_fit_count']==manifest['full_trace_projected_correct_grid_fit_count'],'grid projection evidence mismatch')
    require(manifest['model_fits']==180 and manifest['run_count']==90,'manifest count mismatch')
    return {'manifest':manifest,'stats':stats}


if __name__=='__main__':
    result=verify()
    print('OIC-14 verified:',result['manifest']['model_fits'],'fits /',result['manifest']['run_count'],'matched pairs')
    print('projected correct:',result['stats']['projected_correct_fit_count'],
          '(grid:',result['stats']['projected_correct_grid_fit_count'],')')
    for r in result['stats']['summary']:
        print(f"n={r['n']} split={r['training_split']} arch={r['architecture']} "
              f"accuracy={r['whole_trace_accuracy_fixed']:.3f}/{r['whole_trace_accuracy_rotating']:.3f} "
              f"d={r['latch_min_distances']}")
