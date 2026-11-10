"""RRC-20: calibration on RRC-19 development labels, with frozen experts.

The calibrator never constructs a RRC-20 test roster or adapts a base model.
A separate shuffled-label gate is the negative control.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import family20 as f, gate20 as gate, learn18 as l, model18 as m, bench19
except ImportError:
    import family20 as f
    import gate20 as gate
    import learn18 as l
    import model18 as m
    import bench19

HERE=Path(__file__).resolve().parent
SEED=f.SEED
STEPS=500
LR=.035
ROLES=('constant','adaptive','shuffled')
SOURCES=('family20.py','gate20.py','train20.py','family19.py','model18.py','learn18.py')


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCES}


def prepare(cases,classes,base):
    import torch
    logits=[];xs=[];labels=[];top=[]
    for case in cases:
        values={key:m.logits(case,base[key]) for key in gate.EXPERTS[:3]}
        logits.append(gate.expert_distributions(case.topology,values,classes))
        xs.append(gate.source_features(case))
        rep=classes[case.topology]['mapping'][case.hidden_graph]
        labels.append(classes[case.topology]['representatives'].index(rep))
        top.append(case.topology)
    # Pad to 33 classes, zero on all absent representatives.
    probs=torch.tensor([[list(p)+[0.]*(33-len(p)) for p in case] for case in logits],dtype=torch.float64)
    xx=torch.tensor(xs,dtype=torch.float64)
    y=torch.tensor(labels,dtype=torch.long)
    return xx,probs,y,top


def train(*,steps=STEPS):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases=f.calibration()
    if len(cases)!=420 or any(fam not in ('signed','product','sparse') for fam in (c.identifier.split('-')[2] for c in cases)):
        raise AssertionError('invalid calibration roster')
    models,_,digests=bench19.checkpoints()
    cls=l.class_map();x,probs,y,top=prepare(cases,cls,models)
    checkpoint={}
    shuffled=y.clone()
    rng=random.Random(SEED+130)
    for topology in l.TOPOLOGIES:
        indices=[i for i,z in enumerate(top) if z==topology]
        vals=[int(y[i]) for i in indices];rng.shuffle(vals)
        for i,val in zip(indices,vals):shuffled[i]=val
    for kind in ROLES:
        torch.manual_seed(SEED+170)
        matrix=torch.nn.Parameter(torch.zeros((4,gate.FEATURES),dtype=torch.float64))
        bias=torch.nn.Parameter(torch.zeros((4,),dtype=torch.float64))
        optimizer=torch.optim.Adam([bias] if kind=='constant' else [bias,matrix],lr=LR)
        labels=shuffled if kind=='shuffled' else y
        for step in range(steps):
            z=bias[None,:] if kind=='constant' else bias[None,:]+x@matrix.T
            w=torch.softmax(z,dim=1)
            mixed=torch.einsum('nk,nkc->nc',w,probs)
            target=mixed[torch.arange(len(labels)),labels].clamp_min(1e-20)
            loss=-torch.log(target).mean()
            if kind!='constant':loss=loss+0.002*(matrix**2).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            z=bias[None,:] if kind=='constant' else bias[None,:]+x@matrix.T
            mixed=torch.einsum('nk,nkc->nc',torch.softmax(z,dim=1),probs)
            rank=mixed.argsort(dim=1,descending=True)
            top12=int((rank[:,:12]==y[:,None]).any(1).sum())
            nll=float(-torch.log(mixed[torch.arange(len(y)),y].clamp_min(1e-20)).mean())
        checkpoint[kind]={
            'schema':'rrc20.gate.v1','kind':kind,'calibration_examples':len(cases),
            'calibration_digest':f.old.digest(cases),'seed':SEED,
            'steps':steps,'learning_rate':LR,'features':gate.FEATURES,
            'experts':list(gate.EXPERTS),'parameter_count':4 if kind=='constant' else 4*(gate.FEATURES+1),
            'bias':[float(v) for v in bias.detach().tolist()],
            'matrix':None if kind=='constant' else [[float(z) for z in row] for row in matrix.detach().tolist()],
            'calibration_canonical_top12':top12,'calibration_nll':nll,
            'test_families_used':False,'base_models_frozen':True,'classifier_supervision_available_at_calibration':True,
            'checkpoint_sha256':digests,'source_sha256':source_hashes(),
            'torch_version':torch.__version__,
        }
    return checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=HERE/'results')
    args=parser.parse_args();args.out_dir.mkdir(exist_ok=True,parents=True)
    for kind,record in train().items():
        p=args.out_dir/f'rrc20_{kind}.json'
        p.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
        print(kind,'calibration top12',record['calibration_canonical_top12'],
              'NLL',round(record['calibration_nll'],4),'->',p,flush=True)

if __name__=='__main__':main()
