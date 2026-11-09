"""RRC-07 CPU-only training of raw-coefficient iterative refinement (test-blind)."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

try:
    from .rrc07 import SEED,PREFIX,new_roster,roster_hash,raw_features,euler_exponents
    from .student07 import (HIDDEN,FF_HIDDEN,DEFAULT_LOOPS,parameter_count,
                             portable_forward)
    from .train06 import build_torch
except ImportError:
    from rrc07 import SEED,PREFIX,new_roster,roster_hash,raw_features,euler_exponents
    from student07 import (HIDDEN,FF_HIDDEN,DEFAULT_LOOPS,parameter_count,
                           portable_forward)
    from train06 import build_torch

HERE=Path(__file__).resolve().parent
STEPS=240
BATCH=64
LR=.003
KINDS=('recurrent','feedforward','shuffled_recurrent')


def digest(path:Path)->str:return sha256(path.read_bytes()).hexdigest()


def load_frozen():
    import torch
    ckpt_path=HERE/'results'/'rrc06_recurrent.json'
    ckpt=json.loads(ckpt_path.read_text())
    if ckpt['schema']!='rrc06.raw_coefficient_student.v1' or ckpt['kind']!='recurrent':
        raise AssertionError('invalid RRC-06 frozen checkpoint')
    if ckpt['input_algebraic_preprocessing']!='none' or ckpt['training_depth']!=2:
        raise AssertionError('frozen feature contract changed')
    net=build_torch('recurrent')
    net.load_state_dict({k:torch.tensor(v,dtype=torch.float32)
                         for k,v in ckpt['weights'].items()},strict=True)
    net.eval()
    for p in net.parameters():p.requires_grad_(False)
    return net,digest(ckpt_path)


def prepare(cases,base_net):
    import torch
    xs=torch.tensor([raw_features(c.prefix) for c in cases],dtype=torch.float32)
    target=[]
    for c in cases:
        a=euler_exponents(c.prefix)   # teacher available *only* in training
        if any(v not in (-1,0,1) for v in a[1:]):
            raise AssertionError('training target out of ternary range')
        target.append([v+1 for v in a])
    ys=torch.tensor(target,dtype=torch.long)
    with torch.no_grad():
        base=torch.cat([base_net(xs[k:k+64]) for k in range(0,len(cases),64)],dim=0)
    return xs,base,ys


def build(kind):
    import torch
    from torch import nn
    if kind in ('recurrent','shuffled_recurrent'):
        class R(nn.Module):
            def __init__(self):
                super().__init__();self.gru=nn.GRU(9,HIDDEN,batch_first=True)
                self.head=nn.Linear(HIDDEN,3)
            def forward(self,x,base,passes=DEFAULT_LOOPS):
                if passes<0:raise ValueError('negative recurrence')
                base=base[:,:,:3]
                if passes==0:return [base]
                base_p=torch.softmax(base,dim=-1)
                h=None; cur=base; stages=[]
                for _ in range(passes):
                    z=torch.cat((x,base_p,torch.softmax(cur,dim=-1)),dim=-1)
                    v,h=self.gru(z,h)
                    cur=base+self.head(v)
                    stages.append(cur)
                return stages
        return R()
    if kind=='feedforward':
        class F(nn.Module):
            def __init__(self):
                super().__init__();self.l1=nn.Linear((PREFIX+1)*6,FF_HIDDEN)
                self.l2=nn.Linear(FF_HIDDEN,(PREFIX+1)*3)
            def forward(self,x,base):
                a=base[:,:,:3]
                z=torch.cat((x,torch.softmax(a,dim=-1)),dim=-1).flatten(1)
                return [a+self.l2(torch.tanh(self.l1(z))).reshape(-1,PREFIX+1,3)]
        return F()
    raise ValueError(kind)


def training_loss(stages,targets):
    import torch
    import torch.nn.functional as F
    weights=torch.tensor([1.4,.7,1.15],dtype=torch.float32)
    losses=[F.cross_entropy(z[:,1:].reshape(-1,3),targets[:,1:].reshape(-1),weight=weights)
            for z in stages]
    return losses[-1]+(.15*losses[0] if len(losses)>1 else 0.)


def train(*,steps:int=STEPS,batch_size:int=BATCH,out_dir:Path|None=None,
          kinds=KINDS):
    import torch
    if steps<1 or batch_size<1:raise ValueError('invalid optimizer budget')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    out_dir=out_dir or HERE/'results';out_dir.mkdir(parents=True,exist_ok=True)
    frozen,frozen_hash=load_frozen()
    # No creation of held-out test splits is reachable in training.
    rosters=new_roster(include_test=False)
    train_data=prepare(rosters['train'],frozen)
    dev_data=prepare(rosters['dev'],frozen)
    source_names=('rrc07.py','student07.py','train07.py')
    fingerprints={p:digest(HERE/p) for p in source_names}
    report={}
    for idx,kind in enumerate(KINDS):
        if kind not in kinds:continue
        torch.manual_seed(SEED+200+idx)
        net=build(kind)
        params=sum(p.numel() for p in net.parameters())
        if params!=parameter_count(kind):raise AssertionError('parameter budget drift')
        optim=torch.optim.Adam(net.parameters(),lr=LR)
        rng=torch.Generator().manual_seed(SEED+300+idx)
        labels=train_data[2]
        if kind=='shuffled_recurrent':
            perm=torch.randperm(len(rosters['train']),generator=torch.Generator().manual_seed(SEED+400))
            labels=labels[perm].clone()
        history=[]
        started=time.monotonic()
        for step in range(steps):
            pick=torch.randint(len(rosters['train']),(batch_size,),generator=rng)
            stages=net(train_data[0][pick],train_data[1][pick])
            loss=training_loss(stages,labels[pick])
            optim.zero_grad(set_to_none=True);loss.backward();optim.step()
            if (step+1)%60==0 or step==steps-1:
                with torch.no_grad():
                    z=net(dev_data[0],dev_data[1])[-1]
                    dev_loss=training_loss([z],dev_data[2])
                    acc=(z[:,1:].argmax(-1)==dev_data[2][:,1:]).float().mean()
                    history.append({'step':step+1,'dev_loss':round(float(dev_loss),6),
                                    'dev_a_accuracy':round(float(acc),6)})
        state={k:v.detach().cpu().tolist() for k,v in net.state_dict().items()}
        ckpt={
            'schema':'rrc07.raw_iterative_refinement.v1',
            'kind':kind,'parameter_count':params,'fp32_weight_bits':32*params,
            'fp32_live_hidden':HIDDEN if kind!='feedforward' else 0,
            'training_passes':DEFAULT_LOOPS if kind!='feedforward' else 1,
            'feature_contract':'raw log1p counts plus frozen RRC-06 class logits; iterative own probabilities',
            'exact_residual_at_inference':False,
            'test_generated_in_training':False,
            'train_roster_sha256':roster_hash(rosters['train']),
            'dev_roster_sha256':roster_hash(rosters['dev']),
            'frozen_base_sha256':frozen_hash,
            'source_sha256':fingerprints,'weights':state,
            'training':{'steps':steps,'batch_size':batch_size,'lr':LR,
                        'optimizer':'Adam','seed':SEED+200+idx,
                        'sampler_seed':SEED+300+idx,
                        'label_shuffle':kind=='shuffled_recurrent',
                        'teacher_used_for_loss_only':True},
            'dev_history':history,'torch_version':torch.__version__}
        with torch.no_grad():
            for sample in (0,len(rosters['dev'])//2,len(rosters['dev'])-1):
                ref=net(dev_data[0][sample:sample+1],dev_data[1][sample:sample+1])[-1][0].tolist()
                p=portable_forward(raw_features(rosters['dev'][sample].prefix),
                                   dev_data[1][sample].tolist(),ckpt)
                maxerr=max(abs(z-y) for a,b in zip(ref,p) for z,y in zip(a,b))
                if maxerr>5e-5:raise AssertionError(('portable parity',kind,maxerr))
        out=out_dir/f'rrc07_{kind}.json'
        out.write_text(json.dumps(ckpt,indent=2,sort_keys=True)+'\n')
        report[kind]={'parameters':params,'dev_history':history,
                      'sha256':digest(out),'elapsed_local_s':round(time.monotonic()-started,2)}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--batch-size',type=int,default=BATCH)
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    p.add_argument('--kinds',nargs='+',choices=KINDS,default=list(KINDS))
    args=p.parse_args()
    print(json.dumps(train(steps=args.steps,batch_size=args.batch_size,
                           out_dir=args.out_dir,kinds=tuple(args.kinds)),indent=2,sort_keys=True))

if __name__=='__main__':main()
