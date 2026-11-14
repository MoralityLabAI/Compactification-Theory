"""Small tied recurrent math transformer, shared SAE, and Atlas-Transport VPD.

AT-VPD is a new visit-conditioned low-rank residual adapter motivated by VPD;
it does not reproduce a fitted Goodfire parameter decomposition. The strictly
tied comparator disables the adapter and uses identical physical weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


class SharedTopKSAE(nn.Module):
    def __init__(self, width: int, features: int, k: int = 4):
        super().__init__()
        if not 1 <= k <= features: raise ValueError("invalid SAE sparsity")
        self.k = k
        self.encoder = nn.Linear(width, features)
        self.decoder = nn.Linear(features, width)
        nn.init.normal_(self.decoder.weight, std=0.03)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.encoder(x)
        values, indices = raw.topk(self.k, dim=-1)
        return torch.zeros_like(raw).scatter(-1, indices, F.relu(values))

    def forward(self, x: torch.Tensor):
        codes=self.encode(x)
        return self.decoder(codes), codes


class AtlasTransportVPD(nn.Module):
    """Shared low-rank factor U,V with visit and SAE-conditioned gate.

    delta_v(h) = ((h @ V) * tanh(R a(h)) * tanh(g[v])) @ U.T.
    Factors are shared across all visits; gates mark virtual depth.
    """
    def __init__(self, hidden: int, features: int, rank: int, visits: int):
        super().__init__()
        if rank <= 0 or visits <= 0: raise ValueError("rank/visits must be positive")
        self.read_factor = nn.Parameter(torch.randn(hidden, rank)*0.08)
        self.write_factor = nn.Parameter(torch.randn(rank, hidden)*0.08)
        self.router = nn.Linear(features, rank, bias=False)
        nn.init.normal_(self.router.weight, std=0.13)
        self.visit_gate = nn.Parameter(torch.ones(visits, rank)*0.7)
        self.scale = 0.75 / math.sqrt(rank)

    def forward(self, h: torch.Tensor, code: torch.Tensor, visit: int,
                *, visit_order: tuple[int,...]|None=None) -> tuple[torch.Tensor,torch.Tensor]:
        site = visit if visit_order is None else visit_order[visit]
        gate = torch.tanh(self.router(code))*torch.tanh(self.visit_gate[site])
        correction = ((h @ self.read_factor)*gate) @ self.write_factor
        return correction * self.scale, gate


class TiedBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(),
                                 nn.Linear(hidden*2,hidden))
    def forward(self, x):
        pre=self.ln1(x)
        x=x+self.attn(pre,pre,pre,need_weights=False)[0]
        return x+self.mlp(self.ln2(x))


@dataclass(frozen=True)
class LoopOutput:
    logits: torch.Tensor
    states: tuple[torch.Tensor,...]
    codes: tuple[torch.Tensor,...]
    correction_energy: torch.Tensor
    gates: tuple[torch.Tensor,...]


class LoopedMathTransformer(nn.Module):
    def __init__(self, *, hidden: int=32, heads: int=4, visits: int=3,
                 sae_width: int=64, sae_k: int=6, vpd_rank: int=4):
        super().__init__()
        self.hidden, self.visits = hidden, visits
        self.embed = nn.Linear(30, hidden)
        self.positions = nn.Parameter(torch.randn(4,hidden)*0.02)
        self.core = TiedBlock(hidden,heads)
        self.head_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden,64)
        self.sae = SharedTopKSAE(hidden,sae_width,sae_k)
        self.vpd = AtlasTransportVPD(hidden,sae_width,vpd_rank,visits)

    def forward(self, data:torch.Tensor, *, use_vpd:bool=False,
                visit_order:tuple[int,...]|None=None, patch:torch.Tensor|None=None,
                patch_token:int=0, code_override:str|None=None) -> LoopOutput:
        if data.ndim!=3 or data.shape[1:]!=(4,30):
            raise ValueError("features must be [batch,4,30]")
        if visit_order is not None and sorted(visit_order)!=list(range(self.visits)):
            raise ValueError("visit control must be a permutation")
        h=self.embed(data)+self.positions
        if patch is not None:
            if patch.ndim!=2 or patch.shape!=(data.shape[0],self.hidden):
                raise ValueError("patch must be [batch,hidden]")
            h=h.clone()
            h[:,patch_token,:]=h[:,patch_token,:]+patch
        states=[]; codes=[]; gates=[]
        energy=h.new_zeros(())
        for t in range(self.visits):
            h=self.core(h)
            # normalize input identically when training SAE and fitting router.
            normalized=F.layer_norm(h,(self.hidden,))
            code=self.sae.encode(normalized)
            if code_override == "zero": code=torch.zeros_like(code)
            elif code_override is not None: raise ValueError("unknown code control")
            codes.append(code)
            if use_vpd:
                correction,gate=self.vpd(h,code,t,visit_order=visit_order)
                h=h+correction
                energy=energy+correction.square().mean()
                gates.append(gate)
            states.append(h)
        logits=self.head(self.head_norm(h[:,3,:]))
        return LoopOutput(logits,tuple(states),tuple(codes),energy,tuple(gates))


def set_base_trainable(model:LoopedMathTransformer, enabled:bool):
    for name in ("embed","core","head","head_norm"):
        for param in getattr(model,name).parameters(): param.requires_grad_(enabled)
    model.positions.requires_grad_(enabled)


def set_adapter_trainable(model:LoopedMathTransformer, enabled:bool):
    for p in model.vpd.parameters(): p.requires_grad_(enabled)


def set_sae_trainable(model:LoopedMathTransformer, enabled:bool):
    for p in model.sae.parameters(): p.requires_grad_(enabled)


def atlas_probe(model:LoopedMathTransformer, task, *, eps:float=1e-2,
                patch_token:int=0, observed:tuple[int,...]=(0,1,3)) -> dict:
    """Finite-difference dimensions at fixed task; exploratory, never certified.

    The 2D intervention directions are frozen first-two SAE decoder vectors.
    Jacobians at visits use the same raw residual flattening coordinates.
    """
    from .tasks import encode
    if eps<=0:raise ValueError("eps > 0 required")
    device=next(model.parameters()).device
    x=encode(task).unsqueeze(0).to(device)
    rows=model.sae.decoder.weight[:, :2].detach()
    rows=F.normalize(rows,dim=0)
    with torch.no_grad():
        base=model(x,use_vpd=True)
        jacobians=[]; singular=[]; curvature=[]
        for axis in range(2):
            kick=(rows[:,axis]*eps).unsqueeze(0)
            plus=model(x,use_vpd=True,patch=kick,patch_token=patch_token)
            minus=model(x,use_vpd=True,patch=-kick,patch_token=patch_token)
            if axis==0:
                second_plus=plus;second_minus=minus
            jacobians.append([(p.states[v][0,list(observed)]-m.states[v][0,list(observed)]).reshape(-1)/(2*eps)
                              for v in range(model.visits) for p,m in [(plus,minus)]])
        for v in range(model.visits):
            J=torch.stack([jacobians[0][v],jacobians[1][v]],dim=-1)
            sv=torch.linalg.svdvals(J)
            singular.append([float(value) for value in sv.cpu()])
            cur=(second_plus.states[v][0,list(observed)]-2*base.states[v][0,list(observed)]
                 +second_minus.states[v][0,list(observed)]).norm()/(eps*eps)
            curvature.append(float(cur.cpu()))
        align=[]
        for v in range(1,model.visits):
            a=torch.stack([jacobians[0][v-1],jacobians[1][v-1]],dim=-1)
            b=torch.stack([jacobians[0][v],jacobians[1][v]],dim=-1)
            qa=torch.linalg.qr(a).Q; qb=torch.linalg.qr(b).Q
            angles=torch.linalg.svdvals(qa.T@qb).clamp(0,1)
            align.append([float(t) for t in angles.cpu()])
    return {"visit_singular_values":singular,"empirical_second_difference_norm":curvature,
            "subspace_alignment_singular_values":align,"status":"finite_difference_diagnostic_only",
            "eps":eps,"feature_axes": [0,1]}
