"""OIC-19: finite synthetic phase-family models with alternative objectives.

The phase-supervised ceiling uses simulator phase labels during training only.
All variants see x,u,n at inference; public-history variants use family labels
for training and never receive true hidden phase.
"""
from __future__ import annotations
import argparse, json, math, time, platform
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from oic15_pilot import make_train_batch, balanced_panel
from oic18_model import PhaseEmissionBelief
from oic18_train import panel, action, LR, BATCH, STEPS, TRAIN_NS, EVAL_NS, SEEDS, SPLITS

MODELS=('entropy','predictive','phase_supervised')
FIXED_REG=.008
PREDICTIVE_WEIGHT=.35
PHASE_WEIGHT=.35


def forward_aux(model: PhaseEmissionBelief,n:int,u:torch.Tensor,x:torch.Tensor):
    """Return prefix family logits, conditional raw NLL ingredients, final state.

    Reproduces OIC-18 ordinary forward without changing the inference model.
    """
    b,t_max=u.shape
    if x.shape!=(b,t_max+1): raise ValueError('bad raw state shape')
    k=model.kernel();g=model.phase_emission(n).view(1,1,n,1)
    noise=torch.sigmoid(model.noise_logit)*.98
    mass=torch.zeros(b,2,n,2,device=u.device,dtype=k.dtype)
    mass[:,:,:,0]=1./(2*n)
    state_ids=torch.arange(2,device=u.device).view(1,1,1,2)
    def observation_likelihood(raw):
        valid=(raw>=0).to(k.dtype).view(b,1,1,1)
        obs=raw.clamp(min=0).view(b,1,1,1)
        like=torch.where(state_ids==obs,1.-noise,noise)
        return valid*like+(1.-valid)
    mass=mass*observation_likelihood(x[:,0])
    mass=mass/mass.sum((1,2,3),keepdim=True).clamp_min(1.e-38)
    logits=[];pred=[]
    for t in range(t_max):
        pulse=g*u[:,t].reshape(b,1,1,1).to(g.dtype)
        before=mass*pulse+mass.flip(-1)*(1-pulse)
        evolved=[]
        for family in range(2):
            v=sum(k[family,j]*torch.roll(before[:,family,:,:],shifts=s,dims=1)
                  for j,s in enumerate((-2,-1,0,1,2)))
            evolved.append(v)
        mass=torch.stack(evolved,dim=1)
        pre_f=mass.sum((2,3)).clamp_min(1e-38)
        like=observation_likelihood(x[:,t+1]);mass=mass*like
        post_f=mass.sum((2,3)).clamp_min(1e-38)
        pred.append((post_f/pre_f).clamp_min(1e-38).log())
        mass=mass/mass.sum((1,2,3),keepdim=True).clamp_min(1e-38)
        by_f=mass.sum((2,3)).clamp_min(1e-38)
        logits.append(by_f[:,1].log()-by_f[:,0].log())
    return torch.stack(logits,dim=1),torch.stack(pred,dim=1),mass


def supervised_phase_loss(n, fam, true_phase, mass,steps):
    """Extra training-only simulator phase supervision; orientation gauge minimized."""
    b=fam.shape[0]; p=mass.sum(-1)
    idx=torch.arange(b,device=fam.device)
    correct=p[idx,fam] / p[idx,fam].sum(-1,keepdim=True).clamp_min(1e-30)
    is_rot=(fam==1)
    # Fixed nonzero phases share an exact observational equivalence class.
    fixed_target=torch.where(true_phase==0,correct[:,0],1.-correct[:,0]).clamp_min(1e-30)
    rot_phase=(true_phase+steps)%n
    plus=correct[idx,rot_phase].clamp_min(1e-30)
    minus=correct[idx,(-rot_phase)%n].clamp_min(1e-30)
    if is_rot.any():
        forward_nll=-plus[is_rot].log().mean()
        reverse_nll=-minus[is_rot].log().mean()
        rotloss=torch.minimum(forward_nll,reverse_nll)
    else:rotloss=correct.sum()*0
    if (~is_rot).any():fixedloss=-fixed_target[~is_rot].log().mean()
    else:fixedloss=correct.sum()*0
    return .5*(fixedloss+rotloss)


def batch_training(n,batch,split,rng):
    """Same seeded family-blind batches as OIC-16/OIC-18, plus private phase labels.

    Independently replay the generator's exact RNG calls; phase labels are only
    consumed by the privileged ceiling.
    """
    fam=rng.integers(0,2,size=batch,dtype=np.int64)
    allowed=np.array([i for i in range(n) if split=='full' or i not in (1,n-2)])
    phase=rng.choice(allowed,size=batch)
    T=2*n
    acts=np.zeros((batch,T),dtype=np.int64)
    for i in range(batch):
        strategy=int(rng.integers(0,5))
        if strategy==0:acts[i,:]=0
        elif strategy==1:acts[i,:]=1
        elif strategy==2:acts[i,:]=(rng.random(T)<.3)
        elif strategy==3:acts[i,:]=(rng.random(T)<.7)
        else:acts[i,:n]=1
    from oic15_pilot import oracle_batch
    fm=torch.from_numpy(fam);ph=torch.from_numpy(phase);u=torch.from_numpy(acts)
    x=oracle_batch(n,fm,ph,u)
    return fm,ph,u,x


def train_one(variant,split,seed,lam=.06,steps=STEPS):
    if variant not in MODELS or split not in SPLITS:raise ValueError('invalid variant/split')
    torch.manual_seed(160016+seed)
    rng=np.random.default_rng(151100+seed*19)
    model=PhaseEmissionBelief('von_mises')
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    checkpoints=[];t0=time.perf_counter()
    for i in range(steps):
        n=TRAIN_NS[i%len(TRAIN_NS)]
        fam,phase,u,x=batch_training(n,BATCH,split,rng)
        logits,pred,mass=forward_aux(model,n,u,x)
        cls=F.binary_cross_entropy_with_logits(logits,fam[:,None].float().expand_as(logits))
        kernel=model.kernel()
        if variant=='entropy':
            # H(p) rather than the OIC-18 quadratic Gini impurity.
            reg=lam*(-kernel.clamp_min(1e-38).log()*kernel).sum()
        else:
            reg=FIXED_REG*(1-(kernel*kernel).sum(-1)).sum()
        if variant=='predictive':
            batch_idx=torch.arange(BATCH)
            aux=-pred[batch_idx,:,fam].mean()
            objective=cls+reg+PREDICTIVE_WEIGHT*aux
        elif variant=='phase_supervised':
            aux=supervised_phase_loss(n,fam,phase,mass,u.shape[1])
            objective=cls+reg+PHASE_WEIGHT*aux
        else:
            aux=cls*0
            objective=cls+reg
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        optimizer.step()
        if i in (0,9,49,99,199,349,steps-1):
            checkpoints.append({'step':i+1,'family_bce':round(float(cls.item()),7),
                'penalty':round(float(reg.item()),7),
                'aux':round(float(aux.item()),7)})
    return model.eval(),{'variant':variant,'split':split,'seed':seed,'steps':steps,
        'batch':BATCH,'lr':LR,'parameters':sum(x.numel() for x in model.parameters()),
        'entropy_coefficient':lam if variant=='entropy' else None,
        'predictive_coefficient':PREDICTIVE_WEIGHT if variant=='predictive' else None,
        'privileged_phase_coefficient':PHASE_WEIGHT if variant=='phase_supervised' else None,
        'supplied_hidden_phase_in_training':variant=='phase_supervised',
        'train_counts':list(TRAIN_NS),'training_rng':'151100+seed*19',
        'checkpoint':checkpoints,'wall_seconds':round(time.perf_counter()-t0,4)}


def allowed_val(n,split):
    allowed=[z for z in range(n) if split=='full' or z not in (1,n-2)]
    return [f*n+z for f in (0,1) for z in allowed]


def validation_logloss(model,split):
    scores=[]
    for n in TRAIN_NS:
        word=action(n,'full_active')
        full=panel(model,n,word,report_phase=False)['probabilities']
        ix=allowed_val(n,split)
        for i in ix:
            p=min(max(float(full[i]),1e-8),1-1e-8)
            y=int(i>=n)
            scores.append(-math.log(p if y else (1-p)))
    return sum(scores)/len(scores)


def kernel_metrics(model):
    probs=model.kernel().detach().cpu().numpy()
    fixed=float(probs[0,2]); rotating=max(float(probs[1,1]),float(probs[1,3]))
    return {'fixed_step_mass':round(fixed,8),'rotating_step_mass':round(rotating,8),
            'rotating_orientation':(-1 if probs[1,1]>=probs[1,3] else 1),
            'rotating_kernel_entropy_bits':round(float(-(probs[1]*np.log2(np.maximum(probs[1],1e-38))).sum()),8),
            'fixed_kernel':probs[0].tolist(),'rotating_kernel':probs[1].tolist()}


def eval_one(model,variant,split,seed,train,ns=EVAL_NS):
    panels={}
    for n in ns:
        panels[str(n)]={}
        for mode in (('full_active','passive','latch') if n in (8,24,64) else (('full_active','passive') if n in (7,) else ('full_active',))):
            panels[str(n)][mode]=panel(model,n,action(n,mode),report_phase=(mode=='full_active'))
    return {'variant':variant,'split':split,'seed':seed,'training':train,
            'kernels':kernel_metrics(model),
            'weights':{k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},
            'panels':panels}


def run(variants,split_conditions,seeds,out,lam=.06,steps=STEPS):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    if out.exists():
        rec=json.loads(out.read_text(encoding='utf-8'))
        if rec['protocol']['name']!='OIC-19-CPU-01' or rec['protocol']['entropy_lambda']!=lam or rec['protocol']['steps']!=steps:
            raise ValueError('resume contract mismatch')
        runs=rec['runs']
    else:
        runs=[]
        rec={'protocol':{'name':'OIC-19-CPU-01','source':'OIC19_PROTOCOL.md',
                     'seeds':list(seeds),'splits':list(split_conditions),
                     'variants':list(variants),'training_n':list(TRAIN_NS),
                     'evaluation_n':list(EVAL_NS),'steps':steps,'batch':BATCH,
                     'lr':LR,'entropy_lambda':lam,'predictive_weight':PREDICTIVE_WEIGHT,
                     'phase_supervision_weight':PHASE_WEIGHT,
                     'phase_supervised_is_privileged_ceiling':True,
                     'torch':torch.__version__,'numpy':np.__version__,
                     'python':platform.python_version()},'runs':runs}
    completed={(r['variant'],r['split'],r['seed']) for r in runs}
    for split in split_conditions:
        for variant in variants:
            for seed in seeds:
                if (variant,split,seed) in completed:continue
                m,train=train_one(variant,split,seed,lam,steps)
                r=eval_one(m,variant,split,seed,train)
                runs.append(r)
                rec['wall_seconds_this_invocation']=round(time.perf_counter()-start,5)
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_text(json.dumps(rec,indent=1,sort_keys=True)+'\n',encoding='utf-8')
                diag=r['panels']['64']['full_active']
                print(f'OIC19 variant={variant} split={split} seed={seed} '
                      f'n24/64_acc={r["panels"]["24"]["full_active"]["accuracy"]:.3f}/{diag["accuracy"]:.3f} '
                      f'n64_phaseMAP={diag["rotating_phase_MAP_gauge_accuracy"]:.3f} '
                      f'mass={r["kernels"]["rotating_step_mass"]:.3f}',flush=True)
    print('OIC19 checkpoint',len(runs),'fits',round(time.perf_counter()-start,2),'seconds',flush=True)
    return rec


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--split',nargs='+',choices=SPLITS,default=list(SPLITS))
    p.add_argument('--seed',nargs='+',type=int,default=list(SEEDS))
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--entropy-lambda',type=float,default=.06)
    args=p.parse_args();run(tuple(args.variant),tuple(args.split),tuple(args.seed),args.out,args.entropy_lambda,args.steps)
