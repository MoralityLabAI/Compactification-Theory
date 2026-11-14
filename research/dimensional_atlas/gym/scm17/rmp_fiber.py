"""SCM-17: SAE-conditioned, visit-multiplexed causal fiber transport for RMP.

The decision fiber is the complement of a source-model *local logit-margin
Jacobian* at clean calibration states. It is not the exact kernel of the
RMP nonlinear head. Run trained checks only with the original RMP loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence
import json
import math
import torch
from torch import nn
from torch.nn import functional as F


def tensor_digest(x: torch.Tensor) -> str:
    """Stable content digest with dtype/shape in the domain separation."""
    import struct
    value=x.detach().cpu().to(torch.float64).contiguous()
    raw=struct.pack('<'+str(value.numel())+'d',*value.flatten().tolist())
    header=json.dumps({'shape':list(value.shape),'encoding':'float64-little-endian'},sort_keys=True).encode()
    return sha256(header+b'\0'+raw).hexdigest()


def _logits_at(model, one_state: torch.Tensor) -> torch.Tensor:
    """The authentic RMP lens includes final LayerNorm and tied unembedding."""
    if one_state.ndim!=1:
        raise ValueError('expected a single hidden state')
    logits=model.model.lens(one_state.reshape(1,1,-1))[0,0]
    if logits.ndim!=1 or not torch.isfinite(logits).all():
        raise ValueError('invalid RMP lens')
    return logits


@dataclass(frozen=True)
class DecisionFrame:
    q: torch.Tensor  # [hidden, rank]; frozen orthonormal basis
    singular_values: tuple[float,...]
    captured_gradient_energy: float
    total_gradients: int
    source_logits_sha256: str
    selection: str


@torch.enable_grad()
def calibrate_decision_frame(analysis, tokens: torch.Tensor, *,
                             contrasts: int=2, max_rank: int=8,
                             calibration_cap: int=16) -> DecisionFrame:
    """Fit a label-blind margin-gradient row space on discovery tokens only.

    Contrasts use clean top-logit vs next clean logits, with *no gold labels*.
    Report retained energy because rank truncation leaves unmodelled slopes.
    """
    if tokens.ndim!=2 or not 1 <= tokens.shape[0] <= calibration_cap:
        raise ValueError('invalid calibration token table')
    if contrasts<1 or max_rank<1:
        raise ValueError('contrasts/max_rank must be positive')
    device=next(analysis.model.parameters()).device
    gradients=[];logit_hashes=[]
    # No gradients through source weights or prior visits; only the readout lens.
    for row in tokens.to(device):
        with torch.no_grad():
            out=analysis.run(row[None])
            h=out.visit_outputs[-1][0,-1].detach()
        state=h.clone().requires_grad_(True)
        logits=_logits_at(analysis,state)
        leaders=torch.argsort(logits.detach(),descending=True)
        lead=int(leaders[0]); other=leaders[1:1+contrasts]
        if len(other)<1:
            raise ValueError('readout requires two logits')
        logit_hashes.append(tensor_digest(logits.detach()))
        for index in other:
            contrast=logits[lead]-logits[int(index)]
            gradient=torch.autograd.grad(contrast,state,retain_graph=True)[0]
            if not torch.isfinite(gradient).all():
                raise ValueError('nonfinite decision gradient')
            gradients.append(gradient.detach().to(dtype=torch.float64))
    matrix=torch.stack(gradients)
    U,S,Vh=torch.linalg.svd(matrix,full_matrices=False)
    nz=int((S>S.max().clamp_min(1)*1e-10).sum().item())
    if nz<1:
        raise ValueError('all calibration gradients are zero')
    rank=min(nz,max_rank)
    q=Vh[:rank].T.contiguous().to(dtype=h.dtype,device=device)
    gram=q.T@q
    if not torch.allclose(gram,torch.eye(rank,device=device,dtype=q.dtype),atol=2e-4,rtol=0):
        raise ValueError('invalid orthonormal frame')
    frac=float((S[:rank].square().sum()/S.square().sum()).item())
    return DecisionFrame(q.detach(),tuple(float(v) for v in S.tolist()),
                         frac,len(gradients),sha256(''.join(logit_hashes).encode()).hexdigest(),
                         'clean_top_logit_minus_clean_alternatives_calibration_only')


class SAEVisitFiber(nn.Module):
    """Shared rank factors with sparse dictionary routing and visit-local gates.

    The readout frame is fixed before adapter training. The fiber means local
    orthogonality to a *retained margin gradient span*, not an exact nonlinear
    readout kernel. The SAE is a frozen pre-existing RMP D-shared dictionary.
    """
    def __init__(self, dictionary: nn.Module, frame: DecisionFrame,
                 *, rank: int, visits: int, amplitude:float=.15):
        super().__init__()
        if visits<1 or rank<1 or not math.isfinite(amplitude) or amplitude<=0:
            raise ValueError('invalid adapter dimensions or amplitude')
        dim, feature_count=frame.q.shape[0],int(dictionary.decoder.shape[0])
        if dictionary.decoder.ndim!=2 or dictionary.decoder.shape[1]!=dim:
            raise ValueError('dictionary hidden dimension mismatch')
        self.dictionary=dictionary
        for param in self.dictionary.parameters():param.requires_grad_(False)
        self.dictionary.eval()
        self.register_buffer('q',frame.q.clone().detach())
        self.register_buffer('frame_energy',torch.tensor(frame.captured_gradient_energy))
        self.read_factor=nn.Parameter(torch.randn(dim,rank)*.06)
        self.write_factor=nn.Parameter(torch.randn(rank,dim)*.06)
        self.router=nn.Linear(feature_count,rank,bias=False)
        nn.init.normal_(self.router.weight,std=.05)
        self.visit_gate=nn.Parameter(torch.full((visits,rank),.6))
        self.sector_gate=nn.Parameter(torch.ones(visits,2))
        self.visits=visits
        self.scale=amplitude/math.sqrt(rank)

    def forward(self, state:torch.Tensor,visit:int,*,sector:str='joint',
                visit_order:Sequence[int]|None=None,zero_codes:bool=False):
        if sector not in ('joint','visible_only','fiber_only'):
            raise ValueError('invalid sector')
        if visit not in range(self.visits):
            raise ValueError('invalid visit')
        site=visit if visit_order is None else int(visit_order[visit])
        if site not in range(self.visits):raise ValueError('invalid visit permutation')
        scale=state.pow(2).mean(dim=-1,keepdim=True).clamp_min(1e-12).sqrt()
        # Fixed SAE and sparse support. Router learns; SAE geometry stays frozen.
        with torch.no_grad():
            codes=self.dictionary.encode(state.detach()/scale.detach())
            if zero_codes:codes=torch.zeros_like(codes)
        gate=torch.tanh(self.router(codes))*torch.tanh(self.visit_gate[site])
        raw=(((state@self.read_factor)*gate)@self.write_factor)*self.scale
        q=self.q
        visible=(raw@q)@q.T
        fiber=raw-visible
        gains=torch.tanh(self.sector_gate[site])
        delta={'joint':gains[0]*visible+gains[1]*fiber,
               'visible_only':gains[0]*visible,
               'fiber_only':gains[1]*fiber}[sector]
        return delta,{'visible_sq':visible.square().mean(),
                      'fiber_sq':fiber.square().mean(),
                      'delta_sq':delta.square().mean()}


def run_with_adapter(analysis,tokens:torch.Tensor,adapter:SAEVisitFiber|None,*,
                     sector:str='joint',zero_codes:bool=False,
                     visit_order:Sequence[int]|None=None,patch=None):
    """Uses AnalysisModel.run's visit_hook; preserves FF-U injection order."""
    if adapter is None and patch is None:return analysis.run(tokens),[]
    if visit_order is not None and sorted(visit_order)!=list(range(adapter.visits)):
        raise ValueError('visit_order must be a permutation')
    energy=[]
    def hook(context):
        state=context.after
        if adapter is not None:
            delta,info=adapter(state,context.visit,sector=sector,
                     zero_codes=zero_codes,visit_order=visit_order)
            state=state+delta
            energy.append(info)
        if patch is not None and patch['visit']==context.visit:
            state=state.clone()
            state[:,patch['position'],:]+=patch['delta']
        return state
    return analysis.run(tokens,hook),energy


def _prediction(analysis,tokens,adapter=None,**kwargs):
    out,_=run_with_adapter(analysis,tokens,adapter,**kwargs)
    return out.logits[:, -1, :].float()


def causal_probe(analysis, tokens:torch.Tensor, frame:DecisionFrame,
                 *,visit_count:int, epsilon:float=1e-2,position:int=-1)->dict:
    """Compare terminal local margin silence with earlier transported effects.

    This is finite-float counterfactual evidence. Source-model gradients yield
    a first-order kernel at the calibration anchor; second-order effects remain.
    """
    if tokens.shape[0]!=1 or epsilon<=0:raise ValueError('one task, positive epsilon')
    with torch.no_grad():
        clean_logits=_prediction(analysis,tokens)
        top=torch.argsort(clean_logits[0],descending=True)[:2]
        decision=(int(top[0]),int(top[1]))
    q=frame.q
    v=torch.linspace(-1,1,steps=q.shape[0],device=q.device,dtype=q.dtype)
    fiber=v-(v@q)@q.T
    if fiber.norm()<1e-6:raise ValueError('no testable complement')
    fiber=epsilon*fiber/fiber.norm()
    plus=[];minus=[]
    with torch.no_grad():
        for visit in range(visit_count):
            res_p=_prediction(analysis,tokens,patch={'visit':visit,'position':position,'delta':fiber[None]})
            res_m=_prediction(analysis,tokens,patch={'visit':visit,'position':position,'delta':-fiber[None]})
            p=float((res_p[0,decision[0]]-res_p[0,decision[1]])-
                    (clean_logits[0,decision[0]]-clean_logits[0,decision[1]]))
            m=float((res_m[0,decision[0]]-res_m[0,decision[1]])-
                    (clean_logits[0,decision[0]]-clean_logits[0,decision[1]]))
            plus.append(p);minus.append(m)
    return {'status':'finite_float_causal_probe_only',
            'selected_clean_margin_logits':list(decision),
            'epsilon':epsilon,'plus_margin_delta':plus,'minus_margin_delta':minus,
            'central_slope_by_visit':[(a-b)/(2*epsilon) for a,b in zip(plus,minus)],
            'frame_retained_energy':frame.captured_gradient_energy}
