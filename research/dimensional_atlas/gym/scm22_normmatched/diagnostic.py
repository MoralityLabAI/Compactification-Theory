"""SCM-22 POST-HOC diagnostic: matched rank-four PCA inside linear head fiber.

This diagnostic was specified after the registered primary outcome was seen.
It must not be counted toward SCM-22's preregistered success criterion.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import torch
from model import exact,input_features,make_pairs,sample_cases,train
from geometry import readout_fiber,fit_sae,frozen_states,make_frames,matched_patch

HERE=Path(__file__).resolve().parent

def fiber_pca_frame(model,calibration,visit=1,mode='fiber'):
    with torch.no_grad():
        aa=frozen_states(model,[a for a,b in calibration])[:,visit]
        bb=frozen_states(model,[b for a,b in calibration])[:,visit]
        delta=bb-aa
        fiber=readout_fiber(model).double()
        if mode=='fiber': sub=fiber
        elif mode=='visible':
            W=model.head.weight.detach().double()
            u,s,v=torch.linalg.svd(W,full_matrices=False)
            sub=v.T
        else:raise ValueError(mode)
        x=delta.double()@sub
        x=x-x.mean(0)
        _,sing,vt=torch.linalg.svd(x,full_matrices=False)
        if sing[3]<1e-8:raise ValueError('projected calibration singular')
        q=(sub@vt[:4].T).float()
        return q,{'mode':mode,'top4_variance_fraction':float((sing[:4]**2).sum()/(sing**2).sum()),
                  'total_projected_variance':float((sing**2).sum())}

@torch.no_grad()
def score_basis(model,pairs,visit,basis,scale=1.0):
    x=input_features([a for a,b in pairs]);donor=input_features([b for a,b in pairs])
    aa,a_states=model(x,record=True);bb,b_states=model(donor,record=True)
    y_a=torch.tensor([exact(a) for a,b in pairs]);y_b=torch.tensor([exact(b) for a,b in pairs])
    ids=((aa.argmax(-1)==y_a)&(bb.argmax(-1)==y_b)&(y_a!=y_b)).nonzero().flatten()
    d=(b_states[visit]-a_states[visit])[ids]
    patch,energy,_=matched_patch(d,basis,scale)
    outs=[]
    for start in range(0,len(ids),128):
        z,_=model(x[ids][start:start+128],patch=(visit,patch[start:start+128]));outs.append(z.argmax(-1))
    return {'eligible':len(ids),'correct':int((torch.cat(outs)==y_b[ids]).sum()),
            'energy_fraction':float(energy.square().mean()),
            'norm_error_max':float((patch.norm(dim=1)-d.norm(dim=1)*scale).abs().max())}

def run():
    torch.set_num_threads(2)
    rows=[]
    for seed in (191,1191,2191):
        model,meta=train(seed,400)
        states=frozen_states(model,sample_cases(2048,seed+80001))
        sae,info=fit_sae(model,states,seed,250)
        cal=make_pairs(192,seed+90001)
        old,oldinfo=make_frames(model,sae,info,cal,seed,visit=1)
        fiber,ff=fiber_pca_frame(model,cal,1,'fiber')
        visible,vf=fiber_pca_frame(model,cal,1,'visible')
        pairs=make_pairs(256,20261117+seed)
        results={}
        for name,q in [('difference_pca4',old['difference_pca4']),('sae_fiber4',old['sae_fiber4']),
                       ('random_fiber4',old['random_fiber4']),('fiber_diff_pca4',fiber),('visible_diff_pca4',visible)]:
            results[name]=score_basis(model,pairs,1,q)
        sweep=[]
        for visit in range(4):
            q_f,meta_f=fiber_pca_frame(model,cal,visit,'fiber')
            q_v,meta_v=fiber_pca_frame(model,cal,visit,'visible')
            sweep.append({'visit':visit,'fiber_diff_pca4':score_basis(model,pairs,visit,q_f),
                          'visible_diff_pca4':score_basis(model,pairs,visit,q_v)})
        rows.append({'seed':seed,'checkpoint_sha256':meta['checkpoint_sha256'],
                     'fiber_variance':ff,'visible_variance':vf,'methods':results,'visit_sweep':sweep})
    return {'schema':'scm22.posthoc_withinfiber_diagnostic.v1','scope':'post-hoc synthetic, not registered gate and not trained SCM20 models',
            'model_and_pair_seeds_identical_to_registered_scm22':True,'rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'results'/'scm22_posthoc_diagnostic.json');p.add_argument('--check',action='store_true');a=p.parse_args()
    report=run();encoded=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if a.check:
        if a.out.read_text()!=encoded:raise SystemExit('posthoc replay FAIL')
        print('posthoc replay PASS')
    else:a.out.write_text(encoded)
    for row in report['rows']:print('seed',row['seed'],{k:v['correct'] for k,v in row['methods'].items()},flush=True)
