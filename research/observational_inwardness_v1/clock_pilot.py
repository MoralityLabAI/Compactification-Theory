"""OIC-13: symmetry-tied clock/controller test for unseen initial phases.

The n-phase clock and phase transition are hard-coded structural priors.
Only the zero-phase event readout is learned; outputs use the known XOR
transition schema. This is a privileged-input inductive-bias control,
not evidence that a GRU inferred the underlying state machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import trajectory

SEEDS=tuple(range(5))
NS=(4,5,6)
SPLITS=('full','heldout')
FAMILIES=('fixed','rotating')
ARCHITECTURES=('reflection_tied','oriented')
STEPS=350
BATCH=96
LR=0.014
PROBE_PROB=0.70


def holdout(n:int) -> tuple[int,int]:
    if n<4:raise ValueError('n>=4 required')
    return (1,n-2)


def schedules(n:int)->dict[str,tuple[int,...]]:
    rng=np.random.default_rng(79013+n)
    random_budget=[0]*(2*n)
    indices=rng.choice(2*n,size=n,replace=False)
    for i in indices:random_budget[int(i)]=1
    return {'latch':(1,)*n+(0,)*n,
            'latch_long':(1,)*n+(0,)*(2*n),
            'always_probe':(1,)*(2*n),
            'never_probe':(0,)*(2*n),
            'alternating':tuple(int(i%2==0) for i in range(2*n)),
            'random_budget':tuple(random_budget)}

class PhaseClock(nn.Module):
    """An exact Z_n recurrent clock with a learned event detector.

    Hidden dynamics are a fixed cyclic permutation. The marker at z=0
    breaks the cyclic symmetry of the output while retaining the exact
    group action on hidden state. Reflection-tied readout depends on cos.
    """
    def __init__(self,architecture:str):
        super().__init__()
        if architecture not in ARCHITECTURES:raise ValueError(architecture)
        self.architecture=architecture
        self.w_cos=nn.Parameter(torch.tensor(0.08))
        self.bias=nn.Parameter(torch.tensor(-0.03))
        if architecture=='oriented':
            self.w_sin=nn.Parameter(torch.tensor(0.0))

    def detector(self,phase:torch.Tensor,n:int)->torch.Tensor:
        theta=(2*math.pi/n)*phase.float()
        logit=self.w_cos*torch.cos(theta)+self.bias
        if self.architecture=='oriented':
            logit=logit+self.w_sin*torch.sin(theta)
        return torch.sigmoid(logit)

    def probability(self,phase:torch.Tensor,n:int,initial_visible:torch.Tensor,
                    probes:torch.Tensor,family:str,teacher:torch.Tensor)->torch.Tensor:
        """Teacher-forced output probabilities, based on learned event detector."""
        visible=torch.cat([initial_visible[:,None],teacher[:,:-1]],dim=1)
        timeline=torch.arange(probes.shape[1])
        phases=phase[:,None]+timeline[None,:]*int(family=='rotating')
        p=self.detector(phases%n,n)
        # Structural XOR: output toggles except when u=1 and marker reports z=0.
        predicted=(1-visible)*(1-probes*p)+visible*probes*p
        return predicted.clamp(1e-6,1-1e-6)

    @torch.no_grad()
    def rollout(self,phase:torch.Tensor,n:int,initial_visible:torch.Tensor,
                probes:torch.Tensor,family:str)->torch.Tensor:
        visible=initial_visible.clone()
        phases=phase.clone()
        result=[]
        for t in range(probes.shape[1]):
            pulse=(self.detector(phases,n)>=0.5).long()*probes[:,t].long()
            visible=(visible.long()^1^pulse).float()
            result.append(visible)
            if family=='rotating':phases=(phases+1)%n
        return torch.stack(result,dim=1)


def oracle_batch(n:int,family:str,phase:torch.Tensor,initial_visible:torch.Tensor,
                 probes:torch.Tensor)->torch.Tensor:
    hidden=phase.clone()
    visible=initial_visible.clone()
    result=[]
    for t in range(probes.shape[1]):
        pulse=probes[:,t]*(hidden==0).float()
        visible=(1-visible)*(1-pulse)+visible*pulse
        result.append(visible)
        hidden=(hidden+int(family=='rotating'))%n
    return torch.stack(result,dim=1)


def fit(n:int,family:str,arch:str,seed:int,split:str)->tuple[PhaseClock,dict]:
    torch.manual_seed(seed)
    rng=np.random.default_rng(100000+seed*1117+n*37)
    train_phases=list(range(n)) if split=='full' else [z for z in range(n) if z not in holdout(n)]
    model=PhaseClock(arch)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    t_len=2*n+6
    start=0
    final=0
    for k in range(STEPS):
        z=torch.from_numpy(rng.choice(train_phases,size=BATCH).astype('int64'))
        x=torch.from_numpy(rng.integers(2,size=BATCH).astype('float32'))
        u=torch.from_numpy((rng.random((BATCH,t_len))<PROBE_PROB).astype('float32'))
        label=oracle_batch(n,family,z,x,u)
        predicted=model.probability(z,n,x,u,family,label)
        loss=F.binary_cross_entropy(predicted,label)
        if k==0:start=round(float(loss.item()),8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final=round(float(loss.item()),8)
    return model,{
      'train_phase_indices':train_phases,
      'heldout_phase_indices':list(holdout(n)),
      'initial_teacher_loss':start,
      'final_teacher_loss':final,
      'train_steps':STEPS,
      'train_batch_size':BATCH,
      'train_rollout_length':t_len,
      'trainable_parameters':sum(p.numel() for p in model.parameters()),
      'learned_values':{name:round(float(param.detach().item()),9) for name,param in model.named_parameters()},
      'fixed_prior':'exact_modular_clock_and_xor_schema',
    }


def evaluate(n:int,family:str,model:PhaseClock,word:tuple[int,...])->dict:
    initial=torch.arange(n,dtype=torch.long)
    x=torch.zeros(n)
    u=torch.tensor([word]*n,dtype=torch.float32)
    emitted=[[0]+[int(v) for v in row] for row in model.rollout(initial,n,x,u,family).tolist()]
    oracle=[list(trajectory(n,family,z,word)) for z in range(n)]
    return {'predicted_state_traces':emitted,
            'exact_state_traces':oracle,
            'whole_trace_exact_by_phase':[int(a==b) for a,b in zip(emitted,oracle)],
            'total_wrong_raw_state_samples':sum(sum(i!=j for i,j in zip(a,b)) for a,b in zip(emitted,oracle))}


def cross_distance(a:list[list[int]],b:list[list[int]])->int:
    return min(sum(x!=y for x,y in zip(left,right)) for left in a for right in b)


def one_pair(n:int,arch:str,seed:int,split:str)->dict:
    models={}
    trainings={}
    for family in FAMILIES:models[family],trainings[family]=fit(n,family,arch,seed,split)
    evals={}
    for name,word in schedules(n).items():
        outcome={fam:evaluate(n,fam,models[fam],word) for fam in FAMILIES}
        f,r=outcome['fixed'],outcome['rotating']
        evals[name]={
            'word':''.join(map(str,word)),
            'families':outcome,
            'learned_code_distance':cross_distance(f['predicted_state_traces'],r['predicted_state_traces']),
            'oracle_code_distance':cross_distance(f['exact_state_traces'],r['exact_state_traces']),
            'learned_terminal_overlap':bool({t[-1] for t in f['predicted_state_traces']} & {t[-1] for t in r['predicted_state_traces']}),
            'oracle_terminal_overlap':bool({t[-1] for t in f['exact_state_traces']} & {t[-1] for t in r['exact_state_traces']}),
        }
    return {'n':n,'architecture':arch,'seed':seed,'training_split':split,'training':trainings,'evaluations':evals}


def protocol()->dict:
    return {
        'tag':'OIC-13-STRUCTURAL-CLOCK-01',
        'task':'OIC-12 actions, initial-phase encoding, frozen schedules and heldout split',
        'n':list(NS),'seeds':list(SEEDS),'splits':list(SPLITS),
        'architectures':list(ARCHITECTURES),'families':list(FAMILIES),
        'phase_transition':'exact modular Z_n shift for rotating and identity for fixed',
        'observation':'raw x_t only; internal phase supplied to model at t=0',
        'readout_prior':'known XOR-visible update and distinguished z=0 event',
        'reflection_tied_readout':'sigmoid(w_cos cos(2pi z/n)+bias)',
        'oriented_readout':'sigmoid(w_cos cos(2pi z/n)+w_sin sin(2pi z/n)+bias)',
        'optimizer':'Adam','lr':LR,'train_steps':STEPS,'batch':BATCH,
        'train_action_probability':PROBE_PROB,'torch':torch.__version__,
        'numpy':np.__version__,'python':platform.python_version(),
        'interpretation':'symbolic-clock inductive-bias control, not an emergent clock',
    }


def run(ns=NS,arches=ARCHITECTURES,seeds=SEEDS,splits=SPLITS):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    rows=[]
    for n in ns:
      for split in splits:
       for arch in arches:
        for seed in seeds:
            row=one_pair(n,arch,seed,split)
            rows.append(row)
            latch=row['evaluations']['latch']
            accuracy=[sum(latch['families'][fam]['whole_trace_exact_by_phase'])/n for fam in FAMILIES]
            print(f'n={n} split={split} arch={arch} seed={seed} accuracy={accuracy} distance={latch["learned_code_distance"]}/{latch["oracle_code_distance"]}',flush=True)
    return {'protocol':protocol(),'runs':rows,'wall_seconds':round(time.perf_counter()-t0,3)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--n',nargs='+',type=int,default=list(NS))
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    parser.add_argument('--arches',nargs='+',choices=ARCHITECTURES,default=list(ARCHITECTURES))
    parser.add_argument('--splits',nargs='+',choices=SPLITS,default=list(SPLITS))
    args=parser.parse_args()
    if any(n<4 for n in args.n):raise SystemExit('n must be at least 4')
    output=run(tuple(args.n),tuple(args.arches),tuple(args.seeds),tuple(args.splits))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print('saved',args.out,'seconds',output['wall_seconds'],flush=True)

if __name__=='__main__':main()
