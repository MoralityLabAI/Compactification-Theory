"""SCM-22 reproducible CPU intervention assay. Purely synthetic trained neural evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from model import Case, exact, input_features, make_pairs, sample_cases, train
from geometry import frozen_states,fit_sae,make_frames,matched_patch

HERE=Path(__file__).resolve().parent
SEEDS=(191,1191,2191)
METHODS=('sham','full_difference','difference_pca4','state_pca4','sae_fiber4','random_fiber4')

def sign_test(a_only:int,b_only:int)->float:
    n=a_only+b_only
    if n==0:return 1.0
    tail=sum(math.comb(n,k) for k in range(min(a_only,b_only)+1))
    return float(min(1.,2.*tail/2**n))

@torch.no_grad()
def evaluate(model,pairs,visit,frames,factor=1.0):
    inp_a=input_features([a for a,_ in pairs]);inp_b=input_features([b for _,b in pairs]);
    la,sa=model(inp_a,record=True);lb,sb=model(inp_b,record=True)
    ya=torch.tensor([exact(a) for a,_ in pairs]);yb=torch.tensor([exact(b) for _,b in pairs]);
    eligible=(la.argmax(-1)==ya)&(lb.argmax(-1)==yb)&(ya!=yb)
    choices=eligible.nonzero().flatten();ifew=len(choices)
    if ifew<64:raise ValueError(f'clean eligible pairs below 64: {ifew}')
    d=(sb[visit]-sa[visit])[choices];x=inp_a[choices];target=yb[choices]
    n=len(choices);results={};norm_errors={};energy_ratios={};fallbacks={}
    outcomes={};bysecond={}
    for method in METHODS:
        if method=='sham':patch=torch.zeros_like(d);ratio=torch.zeros(n);fallback=0
        elif method=='full_difference':patch=d*factor;ratio=torch.ones(n);fallback=0
        else:patch,ratio,fallback=matched_patch(d,frames[method],factor)
        norm_error=float((patch.norm(dim=1)-d.norm(dim=1)*factor).abs().max()) if method!='sham' else 0.
        if method!='sham' and norm_error>1e-5:raise ValueError(f'norm mismatch {method}: {norm_error}')
        outs=[]
        for start in range(0,n,128):
            pr,_=model(x[start:start+128],patch=(visit,patch[start:start+128]));outs.append(pr.argmax(-1))
        correct=torch.cat(outs)==target
        outcomes[method]=correct.cpu().tolist()
        results[method]=int(correct.sum());norm_errors[method]=norm_error
        energy_ratios[method]=float(ratio.square().mean())
        fallbacks[method]=fallback
        counts={}
        for i in range(n):
            op=pairs[int(choices[i])][0].second
            row=counts.setdefault(str(op),[0,0]);row[0]+=int(correct[i]);row[1]+=1
        bysecond[method]=counts
    a=outcomes['difference_pca4'];b=outcomes['sae_fiber4'];c=outcomes['random_fiber4']
    return {'heldout_pairs':len(pairs),'eligible':n,
            'clean_recipient_correct':int((la.argmax(-1)==ya).sum()),
            'clean_donor_correct':int((lb.argmax(-1)==yb).sum()),
            'factor':factor,'visit':visit,'successes':results,'norm_error_max':norm_errors,
            'projection_energy_fraction':energy_ratios,'fallback_counts':fallbacks,
            'by_second_operator':bysecond,
            'diff_vs_sae':{'diff_only':sum(x and not y for x,y in zip(a,b)),
                           'sae_only':sum(y and not x for x,y in zip(a,b)),
                           'both':sum(x and y for x,y in zip(a,b)),
                           'neither':sum(not x and not y for x,y in zip(a,b)),
                           'paired_sign_p':sign_test(sum(x and not y for x,y in zip(a,b)),sum(y and not x for x,y in zip(a,b)))},
            'diff_vs_random':{'diff_only':sum(x and not y for x,y in zip(a,c)),
                              'random_only':sum(y and not x for x,y in zip(a,c))}}


def run(seeds=SEEDS,updates=400,sae_steps=250,calibration_size=192,test_size=256):
    torch.set_num_threads(2);start=time.perf_counter()
    root={'schema':'scm22.cpu_normmatched_fiber.v1','evidence_class':'fresh trained synthetic 4-visit neural model with F11 exact oracle; no SCM20 private weights','num_threads':2,
          'registered_protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
          'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'model.py',HERE/'geometry.py',HERE/'run.py']},
          'seeds':list(seeds),'trained':[],'scientific_gate':{}}
    for seed in seeds:
        model,training=train(seed,updates=updates)
        states=frozen_states(model,sample_cases(2048,seed+80001))
        sae,sae_info=fit_sae(model,states,seed,sae_steps)
        calibration=make_pairs(calibration_size,seed+90001)
        frames,frame_info=make_frames(model,sae,sae_info,calibration,seed)
        # This marks completion of source/calibration phase; test pairs generated below.
        frozen_digest=training['checkpoint_sha256'];
        heldout=make_pairs(test_size,20261117+seed)
        primary=evaluate(model,heldout,1,frames,1.0)
        half=evaluate(model,heldout,1,frames,.5)
        # Terminal test of the linear-head kernel, explicitly non-exact after LayerNorm.
        terminal=evaluate(model,heldout,3,frames,1.0)
        root['trained'].append({'seed':seed,'training':training,'sae':{k:v for k,v in sae_info.items() if k!='active_frequencies'},
                                'frame':frame_info,'primary':primary,'half_norm':half,
                                'terminal_diagnostic':{k:terminal[k] for k in ('eligible','successes')},
                                'frozen_checkpoint_sha256':frozen_digest})
    ok=[]
    for row in root['trained']:
        n=row['primary']['eligible'];c=row['primary']['successes'];diff=c['difference_pca4'];sae=c['sae_fiber4'];random=c['random_fiber4']
        ok.append(diff/n>=.5 and (diff-sae)/n>=.1 and (diff-random)/n>=.1)
    root['scientific_gate']={'thresholds':'diff-pca >= .5 and >= .1 above both rank-4 matched-norm fibers in each seed',
                             'per_seed_pass':ok,'passed':all(ok),'elapsed_seconds':time.perf_counter()-start,
                             'model_provenance':'deterministic local CPU PyTorch seed training; frozen weights excluded from Github',
                             'limitations':'same finite semantic combinations may recur across training/calibration/final, nuisance samples differ; not independent mathematical family generalization'}
    return root

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'results'/'scm22_cpu_report.json')
    p.add_argument('--check',action='store_true');p.add_argument('--updates',type=int,default=400)
    a=p.parse_args();result=run(updates=a.updates)
    # Strip timing from the canonical comparison; timings recorded separately.
    for row in result['trained']:row['training'].pop('training_seconds',None)
    result['scientific_gate'].pop('elapsed_seconds',None)
    output=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.check:
        if not a.out.is_file() or a.out.read_text()!=output:raise SystemExit('SCM22 deterministic replay mismatch')
        print('SCM22 replay PASS')
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(output);print('Wrote',a.out)
    for row in result['trained']:
        print('seed',row['seed'],'clean',row['primary']['eligible'],'/',row['primary']['heldout_pairs'],
              'primary',row['primary']['successes'],'discordance',row['primary']['diff_vs_sae'])
    print('registered gate',result['scientific_gate']['passed'])

if __name__=='__main__':main()
