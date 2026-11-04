"""LSV-08: centered error propagation through the pretrained recurrent block.

The mathematical sensitivity formulas hold for ideal real arithmetic about
an *exact* clean trajectory. CPU execution obtains nominal centers using Torch
FP64 and assumes those values are exact; it does not certify transcendentals or
roundoff of computed centers. The results are CONDITIONAL numeric bounds.

This is an exploratory algorithm introduced after LSV-07's empty useful boxes;
no claim of a preregistered comparison or novel general theorem is made.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

GELU_LIP = 1.25


def gain_affine(radius:torch.Tensor,layer:torch.nn.Linear)->torch.Tensor:
    """Radius of real affine image, exact algebraic form |W|r (FP64 eval)."""
    w=layer.weight.detach().double().abs()
    return F.linear(radius,w)


def centered_gelu(center, radius):
    """For an ideal exact center, GELU changes by at most (5/4)r."""
    return F.gelu(center),GELU_LIP*radius


def centered_affine(center,radius,layer):
    return F.linear(center,layer.weight.detach().double(),layer.bias.detach().double()),gain_affine(radius,layer)


def layernorm_difference_radius(center:torch.Tensor,radius:torch.Tensor,ln:torch.nn.LayerNorm):
    """Real-arithmetic Jacobian bound on every segment inside center ± radius.

    Let std=sqrt(mean((x-mean x)^2)+eps), v=x-mean(x). Along any segment
    x=c+t*delta with |delta_j|<=r_j, std>=max(sqrt(eps),std(c)-max_j r_j).
    |v_i(x)| <= |v_i(c)|+r_i+mean(r).
    Then |dLN_i/dx_j| <= |gamma_i| ( (delta_ij+1/d)/stdmin
                        + V_i V_j/(d stdmin^3)). Integrate this bound
    along the segment to get the expression below.
    """
    if center.shape!=radius.shape or center.shape[-1]!=ln.normalized_shape[0]:
        raise ValueError('mismatched normalized dimensions')
    if bool((radius<0).any()):raise ValueError('nonnegative radius required')
    d=center.shape[-1]
    mu=center.mean(-1,keepdim=True)
    v=center-mu
    eps=float(ln.eps)
    std=(v.square().mean(-1,keepdim=True)+eps).sqrt()
    max_r=radius.amax(dim=-1,keepdim=True)
    lower=torch.maximum(torch.full_like(std,eps**.5),std-max_r)
    avg_r=radius.mean(-1,keepdim=True)
    V=v.abs()+radius+avg_r
    sum_Vr=(V*radius).sum(-1,keepdim=True)
    bound=ln.weight.detach().double().abs()*(
       (radius+avg_r)/lower + V*sum_Vr/(d*lower.pow(3)))
    # Independently, both any LayerNorm output and the exact clean output
    # lie in [-sqrt(d)|gamma|+beta, sqrt(d)|gamma|+beta]. Therefore output
    # distance is <= 2*sqrt(d)*|gamma|; use the tighter of the two bounds.
    bound=torch.minimum(bound, 2*(d**0.5)*ln.weight.detach().double().abs())
    return bound


def centered_ln(center,radius,ln):
    out=F.layer_norm(center,ln.normalized_shape,
                     ln.weight.detach().double(),ln.bias.detach().double(),ln.eps)
    return out,layernorm_difference_radius(center,radius,ln)


def centered_block(center,radius,block):
    """One frozen ideal-recurrence pass, with worst-case coordinate errors."""
    if center.dtype!=torch.float64 or radius.dtype!=torch.float64:
        raise ValueError('require float64 values for center and radius')
    c,r=center.transpose(1,2),radius.transpose(1,2)
    c,r=centered_affine(c,r,block.tok_mix[0])
    c,r=centered_gelu(c,r)
    c,r=centered_affine(c,r,block.tok_mix[2])
    c,r=c.transpose(1,2),r.transpose(1,2)
    center,radius=centered_ln(center+c,radius+r,block.ln1)
    c,r=centered_affine(center,radius,block.chan_mix[0])
    c,r=centered_gelu(c,r)
    c,r=centered_affine(c,r,block.chan_mix[2])
    return centered_ln(center+c,radius+r,block.ln2)


def robust_classes(center, radius, head):
    """Pairwise exact-real affine logit gaps vs weighted hidden delta bounds."""
    weights=head.weight.detach().double()
    bias=head.bias.detach().double()
    logits=F.linear(center,weights,bias)
    top=logits.argmax(-1)
    top_w=weights[top]
    top_logits=logits.gather(-1,top.unsqueeze(-1))
    diff=(top_w.unsqueeze(-2)-weights).abs()
    error=(diff*radius.unsqueeze(-2)).sum(-1)
    gap=top_logits-logits
    gap.scatter_(-1,top[...,None],float('inf'))
    certified=(gap>error).all(-1)
    return torch.where(certified,top,torch.full_like(top,-1)),{
        'certified_cells':int(certified.sum()),
        'max_final_radius':round(float(radius.amax()),8),
        'mean_final_radius':round(float(radius.mean()),8)}


def certify_centered(model,z1:torch.Tensor,epsilon:float,mask=None,steps:int=3):
    if z1.ndim!=3 or z1.shape[-2:]!=(81,48) or steps!=3:
        raise ValueError('expected a post-first-pass [N,81,48] state, three more steps')
    if epsilon<0 or not float(epsilon)<float('inf'):
        raise ValueError('radius must be nonnegative and finite')
    if mask is not None and (mask.shape!=z1.shape or mask.dtype!=torch.bool):
        raise ValueError('mask dimension or dtype invalid')
    center=z1.detach().to(torch.float64)
    radius=torch.full_like(center,epsilon)
    if mask is not None:radius=radius*mask
    maxrad=[]
    for _ in range(steps):
        center,radius=centered_block(center,radius,model.block)
        if not bool(torch.isfinite(center).all() and torch.isfinite(radius).all() and (radius>=0).all()):
            raise ValueError('overflow or invalid radius')
        maxrad.append(float(radius.amax()))
    classes,meta=robust_classes(center,radius,model.head)
    meta.update({'per_block_radius_upper':[round(r,8) for r in maxrad],
         'status':'conditional_exact_center_FP64_numeric; theorem assumes ideal real centers'})
    return classes,meta,center,radius
