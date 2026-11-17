"""OIC-14: learn the recurrent phase transition from observable x traces.

Both architectures receive the *initial* phase encoding and the known XOR
observation rule. Neither receives the true phase update. Training minimizes
only visible-output BCE with teacher-forced x_t; evaluation uses hard binary
closed-loop rollouts. Orthogonal rotation has exact SO(2) equivariance,
while the dense matrix has no structural constraint.

Dependencies: torch and numpy. Pure-stdlib receipt verifier is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import trajectory

NS = (4, 5, 6)
SEEDS = tuple(range(5))
SPLITS = ('full', 'heldout')
ARCHITECTURES = ('orthogonal_local', 'orthogonal_grid', 'unconstrained')
FAMILIES = ('fixed', 'rotating')
STEPS = 350
BATCH = 96
LR = 0.014
PROBE_P = 0.7
WIDTH = 2


def holdout(n: int) -> tuple[int, int]:
    if n < 4:
        raise ValueError('n >=4 required')
    return (1, n - 2)


def schedules(n: int) -> dict[str, tuple[int, ...]]:
    rng = np.random.default_rng(79013 + n)
    loc = rng.choice(2*n, size=n, replace=False)
    random_budget = [0] * (2*n)
    for j in loc: random_budget[int(j)] = 1
    return {
        'latch': (1,) * n + (0,) * n,
        'latch_long': (1,) * n + (0,) * (2*n),
        'always_probe': (1,) * (2*n),
        'never_probe': (0,) * (2*n),
        'alternating': tuple(int(i % 2 == 0) for i in range(2*n)),
        'random_budget': tuple(random_budget),
    }


class LearnableClock(nn.Module):
    def __init__(self, n: int, architecture: str):
        super().__init__()
        if architecture not in ARCHITECTURES or n < 4:
            raise ValueError('invalid architecture or phase count')
        self.n = n
        self.architecture = architecture
        # Same phase midpoint prior for *both* families; learner has to infer update.
        starting_omega = math.pi/n
        if architecture.startswith('orthogonal'):
            self.omega = nn.Parameter(torch.tensor(starting_omega))
        else:
            self.matrix = nn.Parameter(torch.tensor([
                [math.cos(starting_omega), -math.sin(starting_omega)],
                [math.sin(starting_omega), math.cos(starting_omega)],
            ], dtype=torch.float32))
        # Same reflection-symmetric detector in both architectures.
        self.gain = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(-6.0))

    def phases(self, z: torch.Tensor, horizon: int) -> torch.Tensor:
        theta0 = (2*math.pi/self.n)*z.float()
        if self.architecture.startswith('orthogonal'):
            angle = theta0[:, None] + torch.arange(horizon, dtype=theta0.dtype, device=z.device)[None, :] * self.omega
            return torch.cos(angle)
        v = torch.stack((torch.cos(theta0), torch.sin(theta0)), -1)
        xs = []
        for _ in range(horizon):
            xs.append(v[:, 0])
            v = v @ self.matrix.T
        return torch.stack(xs, dim=1) if xs else z.new_zeros((len(z), 0), dtype=torch.float32)

    def readout(self, z: torch.Tensor, horizon: int) -> torch.Tensor:
        return torch.sigmoid(self.gain * self.phases(z, horizon) + self.bias)

    def teacher_probability(self, z: torch.Tensor, x0: torch.Tensor, u: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prev = torch.cat((x0[:, None], target[:, :-1]), dim=1)
        pulse = self.readout(z, u.shape[1])*u
        return ((1-prev)*(1-pulse) + prev*pulse).clamp(1e-6, 1-1e-6)

    @torch.no_grad()
    def rollout(self, z: torch.Tensor, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        events = self.readout(z,u.shape[1]) >= 0.5
        x = x0.long()
        outputs=[]
        for t in range(u.shape[1]):
            x = x ^ 1 ^ (u[:,t].long() * events[:,t].long())
            outputs.append(x)
        return torch.stack(outputs,dim=1) if outputs else u.long()


def oracle_batch(n: int, family: str, phase: torch.Tensor, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    z=phase.clone(); x=x0.clone(); observed=[]
    for t in range(u.shape[1]):
        pulse=u[:,t]*(z==0).float()
        x=(1-x)*(1-pulse)+x*pulse
        observed.append(x)
        z=(z+int(family=='rotating'))%n
    return torch.stack(observed,dim=1)


def fit(n:int,family:str,arch:str,seed:int,split:str) -> tuple[LearnableClock,dict]:
    torch.manual_seed(seed)
    rng=np.random.default_rng(100000+seed*1117+n*37)
    train_phases = list(range(n)) if split=='full' else [z for z in range(n) if z not in holdout(n)]
    model=LearnableClock(n,arch)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    length=2*n+6
    initial_loss=0; last_loss=0
    warmstart_omega=None
    warmstart_scans=0
    for it in range(STEPS):
        z=torch.from_numpy(rng.choice(train_phases,size=BATCH).astype('int64'))
        x=torch.from_numpy(rng.integers(2,size=BATCH).astype('float32'))
        u=torch.from_numpy((rng.random((BATCH,length))<PROBE_P).astype('float32'))
        target=oracle_batch(n,family,z,x,u)
        if it == 0 and arch == 'orthogonal_grid':
            # Oracle-free action/sensor likelihood screen over a prespecified grid.
            # Uses only the first training minibatch and the same task detector;
            # this is extra inference compute and is charged in the receipt.
            with torch.no_grad():
                grid=torch.arange(4*n,dtype=torch.float32)*(2*math.pi/(4*n))
                theta=(2*math.pi/n)*z.float()
                times=torch.arange(u.shape[1],dtype=torch.float32)
                angles=theta[None,:,None]+grid[:,None,None]*times[None,None,:]
                detector=torch.sigmoid(model.gain*torch.cos(angles)+model.bias)
                previous=torch.cat((x[:,None],target[:,:-1]),dim=1)
                observed=(1-previous)[None,:,:]*(1-u[None,:,:]*detector)+previous[None,:,:]*u[None,:,:]*detector
                pointwise=F.binary_cross_entropy(observed.clamp(1e-6,1-1e-6),target[None,:,:].expand_as(observed),reduction='none')
                idx=int(pointwise.mean(dim=(1,2)).argmin())
                model.omega.copy_(grid[idx])
                warmstart_omega=float(grid[idx])
                warmstart_scans=int(grid.numel())
        p=model.teacher_probability(z,x,u,target)
        loss=F.binary_cross_entropy(p,target)
        if it==0: initial_loss=float(loss.detach())
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        last_loss=float(loss.detach())
    with torch.no_grad():
        if arch.startswith('orthogonal'):
            learned_rot=float(model.omega.item())
            norm_deviation=0.0
            equivariance_residual=0.0
            learned_matrix = [[math.cos(learned_rot),-math.sin(learned_rot)],
                              [math.sin(learned_rot),math.cos(learned_rot)]]
        else:
            learned_matrix=model.matrix.detach().numpy().tolist()
            m=model.matrix
            norm_deviation=float(torch.linalg.matrix_norm(m.T@m-torch.eye(2)).item())
            angle=2*math.pi/n
            rot=torch.tensor([[math.cos(angle),-math.sin(angle)],
                              [math.sin(angle),math.cos(angle)]],dtype=m.dtype)
            equivariance_residual=float(torch.linalg.matrix_norm(m@rot-rot@m).item())
            learned_rot=None
    return model,{
        'train_phase_indices':train_phases,
        'warmstart_omega':None if warmstart_omega is None else round(warmstart_omega,8),
        'warmstart_extra_candidate_evaluations':warmstart_scans,
        'heldout_phase_indices':list(holdout(n)),
        'train_steps':STEPS, 'batch_size':BATCH,'train_rollout_length':length,
        'initial_teacher_bce':round(initial_loss,8),'final_teacher_bce':round(last_loss,8),
        'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'learned_values':{name:[round(float(t),8) for t in value.detach().flatten().tolist()]
                          for name,value in model.named_parameters()},
        'learned_omega':None if learned_rot is None else round(learned_rot,8),
        'true_omega':0.0 if family=='fixed' else round(2*math.pi/n,8),
        'transition_matrix':[[round(float(x),8) for x in row] for row in learned_matrix],
        'orthogonality_defect':round(norm_deviation,8),
        'rotation_commutator_defect':round(equivariance_residual,8),
        'known_priors':['phase encoded at t=0','binary XOR visible update','z=0 event via cos readout'],
    }


def evaluate(n:int,family:str,model:LearnableClock,word:tuple[int,...])->dict:
    z=torch.arange(n,dtype=torch.long);x=torch.zeros(n)
    u=torch.tensor([word]*n,dtype=torch.float32)
    with torch.no_grad():roll=model.rollout(z,x,u)
    emitted=[[0]+[int(k) for k in row] for row in roll.tolist()]
    oracle=[list(trajectory(n,family,z0,word)) for z0 in range(n)]
    return {
        'predicted_state_traces':emitted,
        'exact_state_traces':oracle,
        'whole_trace_exact_by_phase':[int(a==b) for a,b in zip(emitted,oracle)],
        'total_wrong_raw_state_samples':sum(sum(x!=y for x,y in zip(a,b)) for a,b in zip(emitted,oracle)),
    }


def distance(a:list[list[int]],b:list[list[int]])->int:
    return min(sum(x!=y for x,y in zip(left,right)) for left in a for right in b)


def one_pair(n:int,arch:str,seed:int,split:str)->dict:
    models={};training={}
    for family in FAMILIES:
        models[family],training[family]=fit(n,family,arch,seed,split)
    evaluations={}
    for name,word in schedules(n).items():
        family_evals={fam:evaluate(n,fam,models[fam],word) for fam in FAMILIES}
        f,r=family_evals['fixed'],family_evals['rotating']
        evaluations[name]={
            'word':''.join(map(str,word)),
            'families':family_evals,
            'learned_code_distance':distance(f['predicted_state_traces'],r['predicted_state_traces']),
            'oracle_code_distance':distance(f['exact_state_traces'],r['exact_state_traces']),
            'learned_terminal_overlap':bool({t[-1] for t in f['predicted_state_traces']} & {t[-1] for t in r['predicted_state_traces']}),
            'oracle_terminal_overlap':bool({t[-1] for t in f['exact_state_traces']} & {t[-1] for t in r['exact_state_traces']}),
        }
    return {'n':n,'architecture':arch,'seed':seed,'training_split':split,'training':training,'evaluations':evaluations}


def protocol() -> dict:
    return {
        'tag':'OIC-14-LEARNABLE-PHASE-01',
        'status':'synthetic_training_experiment',
        'n':list(NS),'seeds':list(SEEDS),'splits':list(SPLITS),
        'architectures':list(ARCHITECTURES),'families':list(FAMILIES),
        'phase_initialization':'known Fourier phase of z0, theta0=2*pi*z/n',
        'learned_update':'SO2(angle omega) vs unconstrained real 2x2 matrix',
        'starting_transition':'midpoint rotation pi/n for local/dense; grid variant scans 4n candidate phase steps using first training minibatch',
        'decoder':'sigmoid(gain*cos_phase+bias), known XOR visible gate',
        'supervision':'teacher-forced public visible x_{t+1} only',
        'evaluation':'hard threshold autoregressive visible states',
        'train_steps':STEPS,'batch_size':BATCH,'lr':LR,'train_action_probability':PROBE_P,
        'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),
        'interpretation':'latent transition is learned from public outputs; structure still supplies exact group representation and XOR rule',
    }


def run(ns=NS,arches=ARCHITECTURES,seeds=SEEDS,splits=SPLITS,out:Path|None=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    beginning=time.perf_counter();rows=[]
    for n in ns:
        for split in splits:
            for arch in arches:
                for seed in seeds:
                    record=one_pair(n,arch,seed,split)
                    rows.append(record)
                    if out is not None:
                        out.parent.mkdir(parents=True,exist_ok=True)
                        out.write_text(json.dumps({'protocol':protocol(),'runs':rows,'status':'partial_until_all_requested_runs_complete'},sort_keys=True)+'\n',encoding='utf8')
                    latch=record['evaluations']['latch']
                    acc=[sum(latch['families'][fam]['whole_trace_exact_by_phase'])/n for fam in FAMILIES]
                    print(f'n={n} split={split} arch={arch} seed={seed} acc={acc} '
                          f'distance={latch["learned_code_distance"]}/{latch["oracle_code_distance"]} '
                          f'omega={record["training"]["rotating"]["learned_omega"]}',flush=True)
    return {'protocol':protocol(),'runs':rows,'wall_seconds':round(time.perf_counter()-beginning,3)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--n',nargs='+',type=int,default=list(NS))
    parser.add_argument('--arches',nargs='+',choices=ARCHITECTURES,default=list(ARCHITECTURES))
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    parser.add_argument('--splits',nargs='+',choices=SPLITS,default=list(SPLITS))
    args=parser.parse_args()
    if any(n<4 for n in args.n):raise ValueError('n must be >=4')
    output=run(tuple(args.n),tuple(args.arches),tuple(args.seeds),tuple(args.splits),out=args.out)
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print('saved',args.out,'seconds',output['wall_seconds'],flush=True)


if __name__=='__main__':main()
