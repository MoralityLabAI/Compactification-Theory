"""SCM-16: SAE-conditioned, visit-multiplexed VPD with a causal readout fiber.

This is a controlled low-rank residual adapter, not fitted Goodfire VPD.
The first readout layer is deliberately rank deficient, creating a known
linear kernel in the *pre-readout* state; no final LayerNorm is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..model import LoopedMathTransformer


@dataclass(frozen=True)
class FiberOutput:
    logits: torch.Tensor
    states: tuple[torch.Tensor, ...]
    codes: tuple[torch.Tensor, ...]
    correction_energy: torch.Tensor
    gates: tuple[torch.Tensor, ...]
    visible_energies: tuple[torch.Tensor, ...]
    fiber_energies: tuple[torch.Tensor, ...]


class FiberATVPD(nn.Module):
    """One factor pair, sparse feature router, visit gates, and two causal sectors.

    Raw delta: ((h @ V) * tanh(R(code)) * tanh(g[visit])) @ U.
    Visible sector is the row space of the *frozen* low-rank readout.
    Fiber sector is its orthogonal complement. The trainable sector gains
    control how the same raw delta is reused across recurrent visits.
    """
    def __init__(self, hidden:int, features:int, rank:int, visits:int, readout_dim:int):
        super().__init__()
        if not 0 < readout_dim < hidden:
            raise ValueError('require a proper readout subspace')
        if rank < 1 or visits < 1:
            raise ValueError('rank and visits must be positive')
        self.read_factor = nn.Parameter(torch.randn(hidden,rank)*.08)
        self.write_factor = nn.Parameter(torch.randn(rank,hidden)*.08)
        self.router = nn.Linear(features,rank,bias=False)
        nn.init.normal_(self.router.weight,std=.13)
        self.visit_gate=nn.Parameter(torch.full((visits,rank),.7))
        self.sector_gate=nn.Parameter(torch.ones(visits,2))
        self.register_buffer('readout_basis',torch.zeros(hidden,readout_dim))
        self.register_buffer('frame_ready',torch.tensor(False))
        self.scale=.75/math.sqrt(rank)
        self.visits=visits

    @torch.no_grad()
    def lock_readout(self, readout_weight:torch.Tensor):
        if bool(self.frame_ready):
            raise ValueError('readout basis already frozen')
        if readout_weight.shape!=(self.readout_basis.shape[1],self.readout_basis.shape[0]):
            raise ValueError('invalid readout matrix shape')
        singular=torch.linalg.svdvals(readout_weight.detach().double())
        if float(singular[-1]) < 1e-7:
            raise ValueError('rank deficient readout basis')
        q,_=torch.linalg.qr(readout_weight.detach().float().T,mode='reduced')
        self.readout_basis.copy_(q)
        self.frame_ready.fill_(True)

    def decompose(self, raw:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        if not bool(self.frame_ready):
            raise ValueError('readout basis must be frozen before using VPD')
        q=self.readout_basis
        visible=(raw@q)@q.T
        return visible,raw-visible

    def forward(self,h:torch.Tensor,code:torch.Tensor,visit:int,
                *,visit_order:tuple[int,...]|None=None,sector:str='joint'):
        if sector not in ('joint','visible_only','fiber_only'):
            raise ValueError('invalid causal fiber sector')
        if not 0 <= visit < self.visits:raise ValueError('invalid visit')
        site=visit if visit_order is None else visit_order[visit]
        gates=torch.tanh(self.router(code))*torch.tanh(self.visit_gate[site])
        raw=((h@self.read_factor)*gates)@self.write_factor*self.scale
        parallel,perp=self.decompose(raw)
        gains=torch.tanh(self.sector_gate[site])
        vis=gains[0]*parallel
        fib=gains[1]*perp
        if sector=='visible_only':delta=vis
        elif sector=='fiber_only':delta=fib
        else:delta=vis+fib
        return delta,gates,vis,fib


class FiberLoopedMathTransformer(LoopedMathTransformer):
    """Shared core with a guaranteed readout-null fiber at each terminal step.

    The explicit readout bottleneck has rank r<h. This intentionally differs
    from SCM-15's full-rank readout, and is matched across all SCM-16 arms.
    """
    def __init__(self,*,hidden:int=32,heads:int=4,visits:int=4,
                 sae_width:int=64,sae_k:int=6,vpd_rank:int=4,readout_dim:int=8):
        super().__init__(hidden=hidden,heads=heads,visits=visits,
                         sae_width=sae_width,sae_k=sae_k,vpd_rank=vpd_rank)
        self.head_norm=nn.Identity()
        self.head=nn.Sequential(nn.Linear(hidden,readout_dim,bias=False),
                                nn.Linear(readout_dim,64))
        self.vpd=FiberATVPD(hidden,sae_width,vpd_rank,visits,readout_dim)
        self.readout_dim=readout_dim

    def lock_fiber(self):
        self.vpd.lock_readout(self.head[0].weight)
        for p in self.head.parameters():p.requires_grad_(False)

    def forward(self,data:torch.Tensor,*,use_vpd:bool=False,
                visit_order:tuple[int,...]|None=None,patch:torch.Tensor|None=None,
                patch_token:int=0,code_override:str|None=None,
                sector:str='joint',
                state_patch:tuple[int,torch.Tensor,int]|None=None) -> FiberOutput:
        if data.ndim!=3 or data.shape[1:]!=(4,30):
            raise ValueError('input must be [B,4,30]')
        if visit_order is not None and sorted(visit_order)!=list(range(self.visits)):
            raise ValueError('visit_order must be a permutation')
        if patch_token not in range(4):raise ValueError('invalid patch token')
        if state_patch is not None and not 0<=state_patch[0]<self.visits:
            raise ValueError('invalid intermediate patch visit')
        h=self.embed(data)+self.positions
        if patch is not None:
            if patch.shape!=(data.shape[0],self.hidden):
                raise ValueError('wrong intervention shape')
            h=h.clone();h[:,patch_token,:]+=patch
        states=[];codes=[];gates=[];ve=[];fe=[]
        energy=h.new_zeros(())
        for t in range(self.visits):
            h=self.core(h)
            code=self.sae.encode(F.layer_norm(h,(self.hidden,)))
            if code_override=='zero':code=torch.zeros_like(code)
            elif code_override is not None:raise ValueError('unknown SAE code control')
            codes.append(code)
            if use_vpd:
                delta,gate,visible,fiber=self.vpd(h,code,t,visit_order=visit_order,sector=sector)
                h=h+delta
                gates.append(gate)
                ve.append(visible.square().mean())
                fe.append(fiber.square().mean())
                energy=energy+delta.square().mean()
            if state_patch is not None and t==state_patch[0]:
                patch_vec=state_patch[1]
                at=state_patch[2]
                if at not in range(4) or patch_vec.shape!=(data.shape[0],self.hidden):
                    raise ValueError('bad intermediate state patch')
                h=h.clone();h[:,at,:]=h[:,at,:]+patch_vec
            states.append(h)
        logits=self.head(self.head_norm(h[:,3,:]))
        return FiberOutput(logits,tuple(states),tuple(codes),energy,tuple(gates),tuple(ve),tuple(fe))
