"""LSV-07: predictive bounded-perturbation region propagation for a frozen TRM.

The mathematical interval operators are sound in ideal real arithmetic.
The executable implementation uses FP64 with large protective margins and
an explicit *assumed* GELU/CDF approximation error bound, not a verified libm.
It is a conditional numerical certificate until arithmetic is formally verified.

Inputs are the measured FP32 hidden state after the FIRST recurrence, treated
as exact (binary rational) real-valued inputs to the remaining 3 ideal blocks.
No damaged trajectory is evaluated by the certificate construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import torch
import torch.nn.functional as F

# Numerical model used for conditional certification. We add conservative
# margins for observed FP64 math, but do not assert a portable verified bound
# on torch.erf, torch.sqrt, or BLAS internals without an independently audited
# backend. Therefore output status is 'conditional_float64_interval'.
# A caller must never treat this as a sound FP32/CUDA implementation guarantee.
MACHINE = 2.0**-53
CDF_ABS_ERROR_ASSUMPTION = 2.0e-12
NONLINEAR_GUARD = 2.0e-11
AFFINE_SAFETY_MULT = 16.0


def expand(lo, hi, extra=0.0):
    """Round endpoints outward one ULP following the explicit error padding."""
    if isinstance(extra, float):
        extra = torch.as_tensor(extra, dtype=lo.dtype, device=lo.device)
    lo = lo - extra
    hi = hi + extra
    return (torch.nextafter(lo, torch.full_like(lo, -float('inf'))),
            torch.nextafter(hi, torch.full_like(hi, float('inf'))))


@dataclass
class Interval:
    lo: torch.Tensor
    hi: torch.Tensor

    def __post_init__(self):
        if self.lo.shape != self.hi.shape or self.lo.dtype != torch.float64 or self.hi.dtype != torch.float64:
            raise ValueError('matching float64 tensors required')
        if not bool(torch.isfinite(self.lo).all() and torch.isfinite(self.hi).all() and (self.lo <= self.hi).all()):
            raise ValueError('invalid finite interval endpoints')

    @classmethod
    def around(cls, center:torch.Tensor, radius:float, mask:torch.Tensor | None=None):
        if radius < 0 or not float(radius) < float('inf'):
            raise ValueError('finite nonnegative radius required')
        c = center.detach().to(torch.float64)
        if mask is None:
            r = radius
        else:
            if mask.shape != c.shape or mask.dtype != torch.bool:
                raise ValueError('mask must be a Boolean tensor with the center shape')
            r = radius*mask.to(torch.float64)
        return cls(*expand(c - r,c + r,NONLINEAR_GUARD*MACHINE*(1+c.abs())))

    def add(self, other:'Interval') -> 'Interval':
        if self.lo.shape != other.lo.shape:
            raise ValueError('incompatible interval sums')
        # binary64 addition one roundoff, expanded by more than ulp.
        lo,hi=expand(self.lo+other.lo,self.hi+other.hi,
                     8*MACHINE*(self.lo.abs()+self.hi.abs()+other.lo.abs()+other.hi.abs()+1))
        return Interval(lo,hi)

    def transpose(self,dim1:int,dim2:int)->'Interval':
        return Interval(self.lo.transpose(dim1,dim2), self.hi.transpose(dim1,dim2))

    def width_max(self)->float:
        return float((self.hi-self.lo).amax())


def affine_box(x:Interval, layer:torch.nn.Linear) -> Interval:
    """Dense affine map, with explicit FP64 summation guard, last dim features."""
    w = layer.weight.detach().to(torch.float64)
    b = layer.bias.detach().to(torch.float64)
    a = w.clamp(min=0)
    neg = w.clamp(max=0)
    lo = F.linear(x.lo,a) + F.linear(x.hi,neg,b)
    hi = F.linear(x.hi,a) + F.linear(x.lo,neg,b)
    # Formula above evaluates positive and negative terms separately; bound all
    # floating products/additions by an assumed standard relative-error model.
    n = w.shape[1]
    gamma=(n+6)*MACHINE/(1-(n+6)*MACHINE)
    summed=F.linear(torch.maximum(x.lo.abs(),x.hi.abs()),w.abs(),b.abs())
    guard=AFFINE_SAFETY_MULT*gamma*(summed+1)
    return Interval(*expand(lo,hi,guard))


def gelu_box(x:Interval)->Interval:
    """Intersect valid GELU enclosures: product of Gaussian CDF and x, Lipschitz.

    Ideal real arithmetic is proven for exact Phi endpoints. Executable Phi
    values use torch.erf with an unverified explicitly charged absolute error.
    """
    l,h=x.lo,x.hi
    p_l=0.5*(1+torch.erf(l/sqrt(2.0)))
    p_h=0.5*(1+torch.erf(h/sqrt(2.0)))
    p_l=(p_l-CDF_ABS_ERROR_ASSUMPTION).clamp(0,1)
    p_h=(p_h+CDF_ABS_ERROR_ASSUMPTION).clamp(0,1)
    candidates=torch.stack((l*p_l,l*p_h,h*p_l,h*p_h))
    pl,pu=candidates.amin(dim=0),candidates.amax(dim=0)
    midpoint=(l+h)/2
    center=F.gelu(midpoint)
    # GELU' = Phi + x*phi; global |GELU'| < 5/4.
    r=torch.maximum(midpoint-l,h-midpoint)
    c_err=CDF_ABS_ERROR_ASSUMPTION*midpoint.abs()+NONLINEAR_GUARD*(1+midpoint.abs())
    ll=torch.maximum(pl,center-1.25*r-c_err)
    uu=torch.minimum(pu,center+1.25*r+c_err)
    # Intersections should be nonempty if the assumed backend error holds.
    if not bool((ll <= uu).all()):
        raise ValueError('numerical GELU enclosures disagree; refuse certificate')
    guard=NONLINEAR_GUARD*(1+l.abs()+h.abs())
    return Interval(*expand(ll,uu,guard))


def layernorm_box(x:Interval, layer:torch.nn.LayerNorm)->Interval:
    """Sound mathematical interval over mean, centered square, variance, sqrt.

    Coordinate correlation is dropped but every elementwise enclosure is valid
    in real arithmetic. Final output intersects the global LayerNorm bound.
    """
    l,h=x.lo,x.hi
    d=x.lo.shape[-1]
    mean_l=l.mean(dim=-1,keepdim=True)
    mean_h=h.mean(dim=-1,keepdim=True)
    numeric_guard=NONLINEAR_GUARD*(1+torch.maximum(l.abs(),h.abs()).amax(dim=-1,keepdim=True))
    vl=l-mean_h-numeric_guard
    vh=h-mean_l+numeric_guard
    min_sq=torch.where((vl<=0)&(vh>=0),torch.zeros_like(vl),torch.minimum(vl.square(),vh.square()))
    max_sq=torch.maximum(vl.square(),vh.square())
    var_lo=(min_sq.mean(dim=-1,keepdim=True)-numeric_guard).clamp(min=0)
    var_hi=max_sq.mean(dim=-1,keepdim=True)+numeric_guard
    eps=float(layer.eps)
    denom_lo=(var_lo+eps).sqrt()
    denom_hi=(var_hi+eps).sqrt()
    # Include all four quotient corners; denominators strictly positive.
    quot=torch.stack((vl/denom_lo,vl/denom_hi,vh/denom_lo,vh/denom_hi))
    norm_l,norm_h=quot.amin(dim=0),quot.amax(dim=0)
    gamma=layer.weight.detach().to(torch.float64)
    beta=layer.bias.detach().to(torch.float64)
    a=torch.where(gamma>=0,norm_l,norm_h)*gamma+beta
    b=torch.where(gamma>=0,norm_h,norm_l)*gamma+beta
    # Since sum ((v-mean)/sqrt(var+eps))^2 <= d, normalized
    # coordinate magnitude <= sqrt(d). Round sqrt(d) upward to integer.
    absolute=float(d**0.5).__ceil__()*gamma.abs()+beta.abs()
    a=torch.maximum(a,-absolute)
    b=torch.minimum(b,absolute)
    guard=NONLINEAR_GUARD*(1+absolute+torch.maximum(a.abs(),b.abs()))
    return Interval(*expand(a,b,guard))


def block_box(x:Interval, block:torch.nn.Module)->Interval:
    tok=x.transpose(1,2)
    tok=affine_box(tok,block.tok_mix[0])
    tok=gelu_box(tok)
    tok=affine_box(tok,block.tok_mix[2]).transpose(1,2)
    z=layernorm_box(x.add(tok),block.ln1)
    chan=affine_box(z,block.chan_mix[0])
    chan=gelu_box(chan)
    chan=affine_box(chan,block.chan_mix[2])
    return layernorm_box(z.add(chan),block.ln2)


def logit_box(x:Interval,head:torch.nn.Module)->Interval:
    return affine_box(x,head)


def certified_output_classes(logits:Interval)->torch.Tensor:
    """Return single provable class if its minimum score exceeds every rival max.

    -1 denotes inconclusive; outputs are computed without a damaged trajectory.
    """
    top=logits.lo.argmax(dim=-1)
    target_lo=logits.lo.gather(-1,top[...,None]).squeeze(-1)
    competitors=logits.hi.clone()
    competitors.scatter_(-1,top[...,None],-float('inf'))
    runner_up=competitors.amax(dim=-1)
    return torch.where(target_lo > runner_up,top,torch.full_like(top,-1))


def predictive_goal_region(model, z_after_first:torch.Tensor, epsilon:float, remaining_steps:int=3, mask:torch.Tensor | None=None):
    """Construct finite-horizon goal certificate from frozen model and clean stage 1.

    All uncertainty enters before the remaining recurrence; the certificate
    uses no damaged forward trace, no task labels and no sampled perturbations.
    """
    if z_after_first.ndim!=3 or z_after_first.shape[1:]!=(81,48):
        raise ValueError('expected [B,81,48] hidden-state tensor')
    if remaining_steps not in (1,2,3):
        raise ValueError('model has a four-step registered evaluation')
    tube=Interval.around(z_after_first,epsilon,mask=mask)
    widths=[]
    for _ in range(remaining_steps):
        tube=block_box(tube,model.block)
        widths.append(round(tube.width_max(),6))
    output=logit_box(tube,model.head)
    classes=certified_output_classes(output)
    return classes,{'final_interval_width':round(tube.width_max(),6),
                    'per_step_width_upper':[float(v) for v in widths],
                    'epsilon':epsilon,
                    'certified_count':int((classes>=0).sum()),
                    'n_cells':classes.numel(),
                    'status':'conditional_float64_interval; assumes bounded Torch erf/sqrt/BLAS errors; not certified CUDA/FP32 arithmetic'}


def invariant_box_check(box:Interval,block,head,expected_labels):
    """Check the sufficient condition F(B) subseteq B subseteq goal polyhedron.

    This tests any box (often fails in learned models), and never claims
    invariance from a finite-horizon readout certificate.
    """
    after=block_box(box,block)
    invariant=bool((after.lo>=box.lo).all() and (after.hi<=box.hi).all())
    labels=certified_output_classes(logit_box(box,head))
    goal_proved=bool((labels==expected_labels).all())
    return {'self_map':invariant, 'goal_region':goal_proved,
            'stable_all_future_steps_conditional':invariant and goal_proved}
