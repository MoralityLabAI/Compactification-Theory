"""OIC-18 preregistered CPU architecture sweep, 500 steps * 128 examples.

Registered train phase counts 4/5/6; heldout phase indices 1,n-2; OOD
7/8/12/16/24/32/64. Identical seeded examples across emission architectures,
with OIC-16 original Fourier models reused as frozen baseline fit receipts.

Output contains fitted weights, training histories and complete clean/posterior
predictions (up to all 128 OOD initial phase and family pairs).
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from oic15_pilot import make_train_batch, balanced_panel
from oic18_model import ARCHITECTURES, PhaseEmissionBelief

TRAIN_NS=(4,5,6)
EVAL_NS=(4,5,6,7,8,12,16,24,32,64)
SEEDS=(0,1,2,3,4)
SPLITS=('full','heldout')
AUGMENTS=('clean','raw_fault_aug')
STEPS=500
BATCH=128
LR=0.035
ARCHS=('temperature','von_mises','harmonic3')


def fix_seed(seed:int):
    torch.manual_seed(160016+seed)


def train_one(arch:str, split:str, seed:int, augment:str, steps:int=STEPS):
    if arch not in ARCHS or split not in SPLITS or augment not in AUGMENTS:
        raise ValueError('invalid train condition')
    fix_seed(seed)
    rng=np.random.default_rng(151100+seed*19)
    fault_rng=np.random.default_rng(161600+seed*211)
    model=PhaseEmissionBelief(arch)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    checkpoint=[]
    start=time.perf_counter()
    for i in range(steps):
        n=TRAIN_NS[i%len(TRAIN_NS)]
        fam,u,x=make_train_batch(n,BATCH,split,rng)
        if augment=='raw_fault_aug':
            x=x.clone()
            flip=torch.from_numpy(fault_rng.random(x.shape)<.04)
            erase=torch.from_numpy(fault_rng.random(x.shape)<.04)
            x[flip]^=1
            x[erase]=-1
        logits=model(n,u,x)
        loss=F.binary_cross_entropy_with_logits(logits,fam[:,None].float().expand_as(logits))
        kernel=model.kernel()
        regularizer=.008*(1-(kernel*kernel).sum(-1)).sum()
        objective=loss+regularizer
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        opt.step()
        if i in (0,9,49,99,199,349,steps-1):
            checkpoint.append({'step':i+1,'loss':round(float(loss.item()),8),
                               'regularizer':round(float(regularizer.item()),8)})
    return model.eval(),{
        'parameters':sum(p.numel() for p in model.parameters()),
        'steps':steps,'batch':BATCH,'lr':LR,'split':split,'augmentation':augment,
        'optimizer':'Adam','checkpoint':checkpoint,
        'wall_seconds':round(time.perf_counter()-start,5),
        'matched_OIC16_training_sample_rng':True,
    }


@torch.no_grad()
def panel(model,n:int,word:tuple[int,...],marker:int=0,flip_pos=None,erase_pos=None,
          report_phase:bool=True)->dict:
    fam,phase,u,x=balanced_panel(n,word,marker=marker)
    if flip_pos is not None or erase_pos is not None:
        x=x.clone()
        if flip_pos is not None:x[:,flip_pos]^=1
        if erase_pos is not None:x[:,erase_pos]=-1
    logits,posterior=model(n,u,x,return_state=True)
    scores=torch.sigmoid(logits[:,-1]).cpu().numpy().astype('float64')
    labels=fam.numpy()
    output={
        'accuracy':round(float(np.mean((scores>0.5)==labels)),8),
        'brier':round(float(np.mean((scores-labels)**2)),8),
        'mean_passive_deviation':round(float(np.mean(abs(scores-0.5))),10),
        'probabilities':[round(float(p),8) for p in scores],
    }
    if report_phase:
        bp=posterior.detach().cpu().numpy()
        # Marginalize family/phase over visible state, then condition on the
        # correct model family to avoid conflating identification with phase inference.
        pcond=bp.sum(-1)
        true_mass=[]
        agreement=[]
        rot_weights=model.kernel()[1].detach().cpu().numpy()
        orientation=-1 if rot_weights[1]>rot_weights[3] else 1
        for f in (0,1):
            for z in range(n):
                m=pcond[f*n+z,f]
                m=m/(m.sum()+1e-38)
                target=(((-z if orientation<0 else z) + (-(len(word)) if orientation<0 else len(word)))%n) if f else z
                # For fixed family nonzero phases form an unidentifiable fibre;
                # report its class mass, rather than using arbitrary phase labels.
                if f==0:
                    mass=float(m[0]) if z==0 else float(m[1:].sum())
                else:
                    mass=float(m[target])
                true_mass.append(mass)
                agreement.append(int(int(m.argmax())==int(target)))
        output['mean_equivalence_class_phase_mass']=round(float(np.mean(true_mass)),8)
        output['rotating_phase_MAP_gauge_accuracy']=round(float(np.mean(agreement[n:])),8)
        output['phase_entropy_bits']=round(float(np.mean([-(a[a>0]*np.log2(a[a>0])).sum()
                     for a in pcond.reshape(2*n,2,n).sum(1)])),8)
    return output


def action(n:int,label:str)->tuple[int,...]:
    if label=='full_active':return (1,)*n
    if label=='passive':return (0,)*(2*n)
    if label=='latch':return (1,)*n+(0,)*n
    if label=='alternate':return tuple(int(i%2==0) for i in range(2*n))
    raise ValueError(label)


def eval_one(model,ns=EVAL_NS):
    all_panels={}
    for n in ns:
        p={}
        for name in ('full_active','passive','latch','alternate'):
            p[name]=panel(model,n,action(n,name),report_phase=(name=='full_active'))
        all_panels[str(n)]=p
    return all_panels


def run(architectures=ARCHS,split_conditions=SPLITS,augments=AUGMENTS,seeds=SEEDS,
        steps=STEPS,out=Path('oic18_raw_sweep.json')):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    runs=[]
    for aug in augments:
        for split in split_conditions:
            for arch in architectures:
                for seed in seeds:
                    model,train=train_one(arch,split,seed,aug,steps)
                    panels=eval_one(model)
                    r={'architecture':arch,'split':split,'augmentation':aug,'seed':seed,
                       'weights':{k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},
                       'training':train,'resource_ledger':model.resource_ledger(64),
                       'emission_parameters':model.emission_parameters(),
                       'panels':panels}
                    runs.append(r)
                    msg=f"arch={arch} aug={aug} split={split} seed={seed} loss={train['checkpoint'][-1]['loss']:.5f}"
                    msg+=f"  n7/8/12/16/24/32/64="+str([round(panels[str(n)]['full_active']['accuracy'],3) for n in (7,8,12,16,24,32,64)])
                    print(msg,flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    record={'protocol':{
        'name':'OIC-18-CPU-01','pre_registered_protocol':'OIC18_PROTOCOL.md',
        'architecture_choices_frozen_before_outcome_inspection':True,
        'architectures':list(architectures),'augmentations':list(augments),
        'splits':list(split_conditions),'seeds':list(seeds),
        'train_phase_counts':list(TRAIN_NS),'test_phase_counts':list(EVAL_NS),
        'train_budget_steps_per_fit':steps,'train_batch':BATCH,'optimizer_lr':LR,
        'training_rng':'151100+seed*19 identical OIC16',
        'augmentation_rng':'161600+seed*211 exactly OIC16',
        'kernel_regularizer':.008,'received_true_phase_at_inference':False,
        'received_family_label_at_inference':False,
        'known_n':True,'torch':torch.__version__,'numpy':np.__version__,
        'python':platform.python_version(),
        'monitor':'learned stochastic phase belief and raw-state likelihood',
        'baseline':'OIC16 clean/fault archived frozen fits',
    },'runs':runs,'wall_seconds':round(time.perf_counter()-started,5)}
    out.write_text(json.dumps(record,indent=1,sort_keys=True)+'\n')
    print('OIC18 COMPLETE',len(runs),'fits',round(record['wall_seconds'],2),'seconds',flush=True)
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--architecture',choices=ARCHS,nargs='+',default=list(ARCHS))
    p.add_argument('--augment',choices=AUGMENTS,nargs='+',default=list(AUGMENTS))
    p.add_argument('--split',choices=SPLITS,nargs='+',default=list(SPLITS))
    p.add_argument('--steps',type=int,default=STEPS)
    a=p.parse_args()
    run(tuple(a.architecture),tuple(a.split),tuple(a.augment),tuple(a.seed),a.steps,a.out)
