"""SCM-09: rational outward interval execution of the pinned RMP real decoder.

All stored binary floating weights are interpreted as EXACT real constants.
Each operation rounds to a dyadic enclosure; exp, sqrt, normal CDF/GELU
have explicit rational analytic remainder bounds. This is a deliberately
bounded CPU proof backend, not a PyTorch floating-point error certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial, isfinite
from typing import Sequence

import torch
from torch import nn

from exact_input import InputEnclosure, pi_interval
from rmp_lipschitz import (rational, sqrt_lower, sqrt_upper, block_bound)

# Constrain expression growth without trusting machine rounding.
GRID = 1 << 64
EXP_TERMS = 32


def _floor(x: Fraction) -> Fraction:
    return Fraction((x * GRID).numerator // (x * GRID).denominator, GRID)


def _ceil(x: Fraction) -> Fraction:
    y=x*GRID
    return Fraction(-((-y.numerator)//y.denominator),GRID)


@dataclass(frozen=True)
class I:
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        if self.lo>self.hi:
            raise ValueError('reversed enclosure')

    @staticmethod
    def of(a, b=None) -> 'I':
        a=rational(a)
        b=a if b is None else rational(b)
        if a>b:
            raise ValueError('reversed bound')
        return I(_floor(a),_ceil(b))

    @property
    def mid(self) -> Fraction:
        return (self.lo+self.hi)/2

    @property
    def radius(self) -> Fraction:
        return (self.hi-self.lo)/2

    def __add__(self,other):
        o=as_i(other)
        return I(_floor(self.lo+o.lo),_ceil(self.hi+o.hi))

    __radd__=__add__

    def __neg__(self):
        return I(-self.hi,-self.lo)

    def __sub__(self,other):
        return self+(-as_i(other))

    def __rsub__(self,other):
        return as_i(other)+(-self)

    def __mul__(self,other):
        o=as_i(other)
        vals=[self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi]
        return I(_floor(min(vals)),_ceil(max(vals)))

    __rmul__=__mul__

    def reciprocal(self):
        if self.lo<=0<=self.hi:
            raise ValueError('division interval crosses zero')
        return I(_floor(1/self.hi),_ceil(1/self.lo))

    def __truediv__(self,other):
        return self*as_i(other).reciprocal()

    def __rtruediv__(self,other):
        return as_i(other)*self.reciprocal()

    def square(self):
        if self.lo<=0<=self.hi:
            lo=Fraction(0)
        else:
            lo=min(self.lo*self.lo,self.hi*self.hi)
        hi=max(self.lo*self.lo,self.hi*self.hi)
        return I(_floor(lo),_ceil(hi))

    def sqrt(self):
        if self.lo<0:
            raise ValueError('negative sqrt interval')
        return I(_floor(sqrt_lower(self.lo,digits=20)),
                 _ceil(sqrt_upper(self.hi,digits=20)))


def as_i(v):
    return v if isinstance(v,I) else I.of(v)


def _exp_nonnegative(v: Fraction)->I:
    if v<0:
        raise ValueError('expected nonnegative argument')
    if v>30:
        raise ValueError('exp argument outside verified pilot domain')
    halves=0
    x=v
    while x>Fraction(1,4):
        x/=2
        halves+=1
    s=Fraction(0)
    term=Fraction(1)
    for n in range(EXP_TERMS+1):
        s+=term
        term=term*x/(n+1)
    # term is x^(N+1)/(N+1)!, all remaining positive;
    # next term ratio <= x/(N+2).
    tail=term/(1-x/Fraction(EXP_TERMS+2))
    result=I.of(s,s+tail)
    for _ in range(halves):
        result=result.square()
    return result


@lru_cache(maxsize=20000)
def _exp_point(v:Fraction)->I:
    if v>=0:
        return _exp_nonnegative(v)
    return _exp_nonnegative(-v).reciprocal()


def exp_interval(v:I)->I:
    left=_exp_point(v.lo)
    right=_exp_point(v.hi)
    return I.of(left.lo,right.hi)


@lru_cache(maxsize=20000)
def phi_point(x:Fraction)->I:
    """Standard Gaussian CDF interval by alternating exact Maclaurin series.

    The first omitted term bounds the alternating tail once the absolute
    terms decrease. For |x|>6 we explicitly use the loose probability box.
    """
    x=rational(x)
    if x<0:
        pos=phi_point(-x)
        return I.of(1-pos.hi,1-pos.lo)
    if x>6:
        return I.of(Fraction(1,2),1)
    terms=max(44, int((x*x)/2)+30)
    s=Fraction(0)
    term=x
    for n in range(terms):
        s += term if n%2==0 else -term
        # t(n+1)/t(n) for x^(2n+1)/(2^n n! (2n+1))
        term=term*x*x*(2*n+1)/(2*(n+1)*(2*n+3))
    # Alternating remainder bounded by term (sequence decreasing for n>=terms)
    plo,phi=pi_interval()
    inv=I.of(2*plo,2*phi).sqrt().reciprocal()
    result=I.of(Fraction(1,2))+ inv*I.of(s-term,s+term)
    return I.of(max(Fraction(0),result.lo),min(Fraction(1),result.hi))


def gelu_interval(x:I)->I:
    """Exact GELU at the interval midpoint, padded by global |GELU'|<5/4."""
    c=x.mid
    fc=I.of(c)*phi_point(c)
    pad=Fraction(5,4)*x.radius
    return I.of(fc.lo-pad,fc.hi+pad)


def _tensor_rows(t:torch.Tensor)->list[list[I]]:
    if not t.is_floating_point() or not torch.isfinite(t).all():
        raise ValueError('finite real-valued weight tensor required')
    return [[I.of(rational(float(x))) for x in row] for row in t.detach().cpu().tolist()]


def _vector(t:torch.Tensor)->list[I]:
    if t.ndim!=1:
        raise ValueError('vector required')
    return _tensor_rows(t.reshape(1,-1))[0]


def _linear(x:Sequence[I],weights:Sequence[Sequence[I]])->list[I]:
    return [sum((a*b for a,b in zip(row,x)),I.of(0)) for row in weights]


def _layernorm(x:Sequence[I],gamma:Sequence[I],bias:Sequence[I],eps)->list[I]:
    d=len(x)
    mean=sum(x,I.of(0))/d
    centered=[v-mean for v in x]
    var=sum((v.square() for v in centered),I.of(0))/d
    inv=(var+I.of(rational(float(eps)))).sqrt().reciprocal()
    return [g*v*inv+b for v,g,b in zip(centered,gamma,bias)]


@dataclass(frozen=True)
class ValidatedTubes:
    pre_centers:tuple[torch.Tensor,...]
    mid_centers:tuple[torch.Tensor,...]
    post_centers:tuple[torch.Tensor,...]
    pre_center_errors:tuple[tuple[Fraction,...],...]
    mid_center_errors:tuple[tuple[Fraction,...],...]
    post_center_errors:tuple[tuple[Fraction,...],...]
    input_scope:str
    frequency_sha256:str
    max_sequence:int
    max_hidden:int
    method:str='rational_outward_interval_exact_real'


def _export(state:Sequence[Sequence[I]]):
    rows=[]; radii=[]
    for row in state:
        vals=[];sq=Fraction(0)
        for q in row:
            f=float(q.mid)
            if not isfinite(f):
                raise ValueError('center overflow')
            f_exact=rational(f)
            worst=max(abs(f_exact-q.lo),abs(f_exact-q.hi))
            sq+=worst*worst
            vals.append(f)
        rows.append(vals)
        radii.append(sqrt_upper(sq,digits=20))
    return torch.tensor(rows,dtype=torch.float64),tuple(radii)


def _validate_model(model:nn.Module,source:InputEnclosure,*,max_seq:int,max_dim:int,max_visits:int):
    if model.training:
        raise ValueError('eval model required')
    if source.scope!='ideal_real_with_exact_dyadic_frequency_table':
        raise ValueError('expected certified ideal-real input tube')
    if not hasattr(model,'blocks') or not hasattr(model,'module_indices'):
        raise ValueError('expected RMP decoder block schedule')
    n=len(source.radii);d=model.hidden_size
    if source.center.shape!=(n,d) or len(source.exact_centers)!=n:
        raise ValueError('input dimension mismatch')
    if n>max_seq or d>max_dim or len(model.module_indices)>max_visits:
        raise ValueError('pilot verifier size cap exceeded: use future optimized backend')
    if n<1 or d<2 or len(model.module_indices)<1:
        raise ValueError('invalid model dimensions')
    if getattr(model,'carry_policy','reset')!='reset':
        raise ValueError('carry policy outside RMP contract')
    for idx in model.module_indices:
        if not 0<=idx<len(model.blocks):
            raise ValueError('bad module index')
        block=model.blocks[idx]
        block_bound(block,sequence=n) # validates source architecture
        if block.mlp[0].bias is not None or block.mlp[2].bias is not None:
            raise ValueError('biased MLP is outside RMP contract')
        if d%block.num_heads:
            raise ValueError('invalid attention heads')


def _preprocess_block(block:nn.Module):
    d=block.hidden_size
    qkv=_tensor_rows(block.qkv.weight)
    return {
        'gamma1':_vector(block.norm1.weight), 'bias1':_vector(block.norm1.bias),
        'gamma2':_vector(block.norm2.weight), 'bias2':_vector(block.norm2.bias),
        'eps1':block.norm1.eps, 'eps2':block.norm2.eps,
        'qkv':qkv, 'out':_tensor_rows(block.attention_out.weight),
        'up':_tensor_rows(block.mlp[0].weight),'down':_tensor_rows(block.mlp[2].weight),
        'heads':block.num_heads, 'head_dim':d//block.num_heads,
        'alpha':I.of(rational(float(block.alpha))),
        'beta':I.of(rational(float(block.beta))),
    }


def _attend(normed:Sequence[Sequence[I]], spec)->list[list[I]]:
    n=len(normed);d=len(normed[0]);head_count=spec['heads'];dh=spec['head_dim']
    qkvs=[_linear(row,spec['qkv']) for row in normed]
    inv_scale=I.of(dh).sqrt().reciprocal()
    attended=[]
    for i in range(n):
        values=[]
        for h in range(head_count):
            offset=h*dh
            q=qkvs[i][offset:offset+dh]
            score=[]
            for j in range(i+1):
                k=qkvs[j][d+offset:d+offset+dh]
                score.append(sum((x*y for x,y in zip(q,k)),I.of(0))*inv_scale)
            # Subtracting a fixed rational shift leaves softmax unchanged.
            shift=max(v.hi for v in score)
            exponent=[exp_interval(s-I.of(shift)) for s in score]
            for j in range(i+1):
                other_lo=sum((e.lo for jj,e in enumerate(exponent) if jj!=j),Fraction(0))
                other_hi=sum((e.hi for jj,e in enumerate(exponent) if jj!=j),Fraction(0))
                e=exponent[j]
                # Exact interval p_j=e_j/(e_j+sum(other positive e))
                p=I.of(e.lo/(e.lo+other_hi),e.hi/(e.hi+other_lo))
                if j==0:
                    acc=[I.of(0)]*dh
                v=qkvs[j][2*d+offset:2*d+offset+dh]
                acc=[z+p*w for z,w in zip(acc,v)]
            values+=acc
        attended.append(_linear(values,spec['out']))
    return attended


def _one_block(state,spec):
    n1=[_layernorm(row,spec['gamma1'],spec['bias1'],spec['eps1']) for row in state]
    att=_attend(n1,spec)
    mids=[[spec['alpha']*x+spec['beta']*a for x,a in zip(xrow,arow)]
          for xrow,arow in zip(state,att)]
    n2=[_layernorm(row,spec['gamma2'],spec['bias2'],spec['eps2']) for row in mids]
    m=[[gelu_interval(v) for v in _linear(row,spec['up'])] for row in n2]
    update=[_linear(row,spec['down']) for row in m]
    output=[[spec['alpha']*mid+spec['beta']*delta for mid,delta in zip(row,u)]
            for row,u in zip(mids,update)]
    return mids,output


def verified_centered_tubes(model:nn.Module,source:InputEnclosure,*,
                            fixed_input_reinjection:bool=False,
                            max_seq:int=16,max_dim:int=32,max_visits:int=8)->ValidatedTubes:
    """Enclose each clean pre/mid/post state from source input+checkpoint.

    Interval arithmetic proves inclusion for the exact real-valued graph;
    recorded float centers are exported with their exact rounding error.
    No PyTorch execution is trusted as a numerical bound.
    """
    _validate_model(model,source,max_seq=max_seq,max_dim=max_dim,max_visits=max_visits)
    rows=source.center.detach().cpu().tolist()
    inp=[[I.of(rational(float(c))-e,rational(float(c))+e) for c in row]
         for row,e in zip(rows,source.radii)]
    states=inp
    pre=[];mid=[];post=[];perr=[];merr=[];oerr=[]
    cache={}
    for idx in model.module_indices:
        if idx not in cache:
            cache[idx]=_preprocess_block(model.blocks[idx])
        c,r=_export(states);pre.append(c);perr.append(r)
        middle,new=_one_block(states,cache[idx])
        c,r=_export(middle);mid.append(c);merr.append(r)
        if fixed_input_reinjection:
            new=[[x+y for x,y in zip(row,orig)] for row,orig in zip(new,inp)]
        c,r=_export(new);post.append(c);oerr.append(r)
        states=new
    return ValidatedTubes(tuple(pre),tuple(mid),tuple(post),
                          tuple(perr),tuple(merr),tuple(oerr),
                          source.scope,source.frequency_sha256,max_seq,max_dim)


def verified_injection_amplitude(tubes:ValidatedTubes,*,injected_token:int,
                                 after_visit:int,radius_fraction)->Fraction:
    """Upper bound SCM-02's RMS-scaled post-visit injection using centered tube."""
    if not 0<=after_visit<len(tubes.post_centers):
        raise ValueError('invalid visit')
    centers=tubes.post_centers[after_visit]
    if not 0<=injected_token<centers.shape[0]:
        raise ValueError('invalid token')
    scale=rational(radius_fraction)
    if scale<0:
        raise ValueError('negative radius')
    v=[rational(float(x)) for x in centers[injected_token].tolist()]
    norm=sqrt_upper(sum((x*x for x in v),Fraction(0)),digits=20)
    err=tubes.post_center_errors[after_visit][injected_token]
    dim=len(v)
    return scale*max((norm+err)*sqrt_upper(Fraction(1,dim),digits=20),
                     rational(float(1e-6)))


def centered_pair_lower(tubes:ValidatedTubes,chart:torch.Tensor,*,
                        observed_tokens:Sequence[int],visit:int):
    """Exact lower separation of projected clean pair at a specific visit."""
    if chart.ndim!=2 or chart.shape[1]!=2 or not torch.isfinite(chart).all():
        raise ValueError('finite chart [hidden,2] required')
    from rmp_lipschitz import spectral_upper
    if not 0<=visit<len(tubes.post_centers):
        raise ValueError('invalid visit')
    centers=tubes.post_centers[visit]
    if chart.shape[0]!=centers.shape[1]:
        raise ValueError('chart hidden dimension mismatch')
    observed=tuple(observed_tokens)
    if len(set(observed))!=len(observed) or len(observed)<2 or any(i<0 or i>=centers.shape[0] for i in observed):
        raise ValueError('bad observed tokens')
    C=[[rational(float(c)) for c in row] for row in chart.detach().cpu().tolist()]
    Cnorm=spectral_upper(chart)
    results={}
    for a in range(len(observed)):
        for b in range(a+1,len(observed)):
            ii,jj=observed[a],observed[b]
            ai=[rational(float(x)) for x in centers[ii].tolist()]
            aj=[rational(float(x)) for x in centers[jj].tolist()]
            projected=[sum(((ai[k]-aj[k])*C[k][axis] for k in range(len(ai))),Fraction(0)) for axis in (0,1)]
            lower=sqrt_lower(sum((p*p for p in projected),Fraction(0)),digits=20)
            lower-=Cnorm*(tubes.post_center_errors[visit][ii]+tubes.post_center_errors[visit][jj])
            results[f'{a},{b}']=max(Fraction(0),lower)
    return results


def centered_null_disk(tubes:ValidatedTubes,chart:torch.Tensor, observed_tokens:Sequence[int],
                       pair_upper:Sequence[dict], *, margin='0'):
    """Source-certified null-braid if all clean projected clearances exceed disk shifts."""
    if len(pair_upper)!=len(tubes.post_centers):
        raise ValueError('visit count mismatch')
    margin=rational(margin)
    if margin<0:
        raise ValueError('negative margin')
    reports=[]
    for site,bounds in enumerate(pair_upper):
        lower=centered_pair_lower(tubes,chart,observed_tokens=observed_tokens,visit=site)
        pairs={key:{'clean_lower':str(value),'disk_upper':str(rational(bounds[key])),
                    'certified':value>rational(bounds[key])+margin} for key,value in lower.items()}
        reports.append({'visit':site+1,'null_braid_ideal_real':all(x['certified'] for x in pairs.values()),
                        'pair_checks':pairs,'floating_execution_certified':False})
    return reports


def verified_centered_pair_bounds(model:nn.Module,tubes:ValidatedTubes,*,
                                  chart:torch.Tensor,axes:torch.Tensor,
                                  amplitude,observed_tokens:Sequence[int],
                                  injected_token:int,inject_after_visit:int=0)->dict:
    """SCM-07 disk derivative bound using SCM-09 proven pre/mid enclosures."""
    from local_jacobian import visit_token_bounds
    bounds=visit_token_bounds(model,sequence=tubes.pre_centers[0].shape[0],
          chart=chart,axes=axes,amplitude=amplitude,observed_tokens=observed_tokens,
          injected_token=injected_token,injected_after_visit=inject_after_visit,
          pre_centers=tubes.pre_centers,mid_centers=tubes.mid_centers,
          pre_center_errors=tubes.pre_center_errors,
          mid_center_errors=tubes.mid_center_errors)
    bounds['schema']='scm09_rational_interval_local_disk_v1'
    bounds['bound_scope']='derived_from_exact_real_interval_execution'
    bounds['center_enclosures_verified_by_this_program']=True
    bounds['floating_execution_certified']=False
    bounds['input_scope']=tubes.input_scope
    bounds['frequency_sha256']=tubes.frequency_sha256
    bounds['maximum_clean_post_error']=[str(max(r)) for r in tubes.post_center_errors]
    return bounds
