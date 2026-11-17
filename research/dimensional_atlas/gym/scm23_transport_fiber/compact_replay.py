"""SCM-23 compact independent CPU replay of the registered fiber transport assay.

Full source, independent receipt verifier and checkpoint-host adapter live in the
SCM-23 research archive; this executable reproduces the frozen numeric endpoint.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scm22_normmatched'))
from model import exact,input_features,make_pairs,sample_cases,train
from geometry import frozen_states,fit_sae,make_frames,readout_fiber,matched_patch

EXPECTED={
 191:{'full':254,'visible':167,'gradient':36,'diff':20,'sae':1,'random':3},
 1191:{'full':256,'visible':185,'gradient':20,'diff':5,'sae':2,'random':1},
 2191:{'full':256,'visible':186,'gradient':46,'diff':10,'sae':4,'random':0},
}

def fitted(model,pairs,kind):
    with torch.no_grad():
        a=frozen_states(model,[x for x,y in pairs])[:,1]
        b=frozen_states(model,[y for x,y in pairs])[:,1]
        delta=b-a
        if kind=='visible':
            _,_,vh=torch.linalg.svd(model.head.weight.detach().double(),full_matrices=False)
            basis=vh.T
        else:basis=readout_fiber(model).double()
    if kind=='gradient':
        grads=[]
        for start in range(0,len(pairs),32):
            batch=pairs[start:start+32];n=len(batch)
            zero=torch.zeros(n,model.width,requires_grad=True)
            logits,_=model(input_features([a for a,b in batch]),patch=(1,zero))
            ar=torch.arange(n);ta=torch.tensor([exact(a) for a,b in batch]);tb=torch.tensor([exact(b) for a,b in batch])
            grads.append(torch.autograd.grad((logits[ar,tb]-logits[ar,ta]).sum(),zero)[0].detach())
        vectors=torch.cat(grads).double()
    else:vectors=delta.double()
    x=vectors@basis;x=x-x.mean(0)
    _,s,vh=torch.linalg.svd(x,full_matrices=False)
    if s[3]<1e-8:raise RuntimeError('calibration rank below four')
    q=(basis@vh[:4].T).float()
    if (q.T@q-torch.eye(4)).abs().max()>2e-5:raise RuntimeError('bad frame')
    if kind!='visible' and (model.head.weight.detach()@q).abs().max()>2e-5:raise RuntimeError('fiber leak')
    return q

@torch.no_grad()
def assay(model,pairs,frames,factor=1.):
    x=input_features([a for a,b in pairs]);y=input_features([b for a,b in pairs])
    a,sa=model(x,record=True);b,sb=model(y,record=True)
    ya=torch.tensor([exact(a) for a,b in pairs]);yb=torch.tensor([exact(b) for a,b in pairs])
    eligible=(a.argmax(-1)==ya)&(b.argmax(-1)==yb)&(ya!=yb)
    ids=eligible.nonzero().flatten()
    if len(ids)!=256:raise RuntimeError(f'expected 256 eligible, found {len(ids)}')
    d=(sb[1]-sa[1])[ids];cases=x[ids];target=yb[ids]
    counts={}
    for name in ('full','visible','gradient','diff','sae','random'):
        if name=='full':patch=d*factor
        else:patch,_,fallback=matched_patch(d,frames[name],factor)
        if name!='full' and fallback:raise RuntimeError('unexpected zero projection')
        if (patch.norm(-1)-d.norm(-1)*factor).abs().max()>1e-5:
            raise RuntimeError('norm mismatch')
        pred,_=model(cases,patch=(1,patch))
        counts[name]=int((pred.argmax(-1)==target).sum())
    return counts

def main():
    torch.set_num_threads(2)
    actual={}
    for seed in (191,1191,2191):
        model,_=train(seed,updates=400)
        states=frozen_states(model,sample_cases(2048,seed+80001))
        sae,info=fit_sae(model,states,seed,250)
        cal=make_pairs(192,seed+90001)
        orig,_=make_frames(model,sae,info,cal,seed,visit=1)
        frames={'sae':orig['sae_fiber4'],'random':orig['random_fiber4']}
        for kind,key in [('visible','visible'),('gradient','gradient'),('fiber','diff')]:
            frames[key]=fitted(model,cal,kind)
        heldout=make_pairs(256,20261201+seed)
        counts=assay(model,heldout,frames)
        if counts!=EXPECTED[seed]:
            same_other=(seed==2191 and all(counts[k]==EXPECTED[seed][k] for k in counts if k!='gradient'))
            if not (same_other and counts['gradient'] in (44,46)):
                raise SystemExit(f'REPLAY FAIL seed {seed}: {counts}')
            print('SCM23 DECLARED CROSS-MACHINE FLOAT VARIATION: seed 2191 gradient 44/46',flush=True)
        actual[seed]=counts
        print('SCM23',seed,counts,flush=True)
    assert [actual[x]['gradient']>=26 for x in (191,1191,2191)] == [True,False,True]
    print('SCM23 CPU REPLAY PASS, registered gate FAIL',flush=True)

if __name__=='__main__':main()
