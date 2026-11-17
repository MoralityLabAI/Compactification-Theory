"""SCM-10 experimental large-shape source enclosures, IEEE-binary64 CPU.

Ideal-real semantics: source checkpoint binary floats and the SCM-08 input
frequency table are exact constants. This backend uses outward one-ULP
padding for scalar/elementwise IEEE ops and deliberately oversized Higham
``gamma`` bounds for matrix products and reductions. It requires a CPU/NumPy
IEEE round-to-nearest execution model (including correctly-rounded sqrt and
no unsafe fast-math/FTZ). It DOES NOT certify PyTorch/CUDA execution itself.

The backend is a performance-oriented second proof implementation of the
same RMP explicit, causal, pre-LN graph as validated_forward.py. Validation
on SCM-09 exact rational intervals and independent high-precision witnesses
is mandatory before calling any shape a verified research result.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial, isfinite
from typing import Sequence

import numpy as np
import torch
from torch import nn

from exact_input import InputEnclosure, pi_interval
from rmp_lipschitz import rational, sqrt_upper, sqrt_lower
from validated_forward import _validate_model

U = 2.0 ** -53
INF = float('inf')
EXP_DEGREE = 32
CDF_DEGREE = 60
CDF_SAFE = 4.0


def _down(x):
    return np.nextafter(np.asarray(x, dtype=np.float64), -INF)


def _up(x):
    return np.nextafter(np.asarray(x, dtype=np.float64), INF)


def _check(*arrays):
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('nonfinite interval operation')


def _from_fraction(x:Fraction, *, upward:bool):
    """Smallest/largest machine bound using exact float-rational comparison."""
    x = Fraction(x)
    f = float(x)
    if not isfinite(f):
        raise ValueError('source coefficient exceeds finite binary64')
    while (Fraction.from_float(f) < x if upward else Fraction.from_float(f) > x):
        f = float(np.nextafter(f, INF if upward else -INF))
    return f


def _gamma(n:int)->float:
    """Inflated roundoff budget for any conventional binary64 dot tree.

    The per-term product/sum error has gamma_(2n) <= 2nu/(1-2nu).
    We use gamma_(32(n+1)) as a conservative budget for the reduction, the
    abs-sum estimates, and our generated certificate arithmetic; see note.
    """
    if n < 1 or 32*(n+1)*U >= 1/4:
        raise ValueError('unsupported reduction length')
    w = 32*(n+1)*U
    return float(_up(w/(1-w)))


@dataclass(frozen=True)
class ArrayI:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        if self.lo.shape != self.hi.shape:
            raise ValueError('interval shape mismatch')
        _check(self.lo, self.hi)
        if np.any(self.lo > self.hi):
            raise ValueError('reversed interval')

    @staticmethod
    def point(x):
        x = np.asarray(x, dtype=np.float64)
        return ArrayI(x.copy(), x.copy())

    @staticmethod
    def box(lo,hi):
        return ArrayI(np.asarray(lo, dtype=np.float64), np.asarray(hi,dtype=np.float64))

    def center_radius(self):
        c = .5*self.lo + .5*self.hi
        r = np.maximum(np.abs(_up(self.hi-c)), np.abs(_up(c-self.lo)))
        return c, _up(r)

    def __add__(self, other):
        b = iv(other)
        return ArrayI(_down(self.lo+b.lo),_up(self.hi+b.hi))
    __radd__ = __add__

    def __neg__(self):
        return ArrayI(-self.hi,-self.lo)
    def __sub__(self,other):
        return self+(-iv(other))
    def __rsub__(self,other):
        return iv(other)+(-self)

    def __mul__(self, other):
        b=iv(other)
        v=np.stack((self.lo*b.lo,self.lo*b.hi,self.hi*b.lo,self.hi*b.hi),axis=0)
        return ArrayI(_down(np.min(v,axis=0)), _up(np.max(v,axis=0)))
    __rmul__ = __mul__

    def reciprocal(self):
        if np.any((self.lo<=0)&(self.hi>=0)):
            raise ValueError('reciprocal crosses zero')
        return ArrayI(_down(1./self.hi), _up(1./self.lo))

    def __truediv__(self,other):
        return self * iv(other).reciprocal()

    def square(self):
        lo=np.minimum(self.lo*self.lo,self.hi*self.hi)
        lo=np.where((self.lo<=0)&(self.hi>=0),0.,lo)
        hi=np.maximum(self.lo*self.lo,self.hi*self.hi)
        return ArrayI(np.maximum(0.,_down(lo)),_up(hi))

    def sqrt(self):
        if np.any(self.lo<0):
            raise ValueError('sqrt negative lower bound')
        return ArrayI(_down(np.sqrt(self.lo)),_up(np.sqrt(self.hi)))

    def sum(self,axis:int=-1):
        n=self.lo.shape[axis];g=_gamma(n)
        lo=np.sum(self.lo,axis=axis)
        hi=np.sum(self.hi,axis=axis)
        alo=np.sum(np.abs(self.lo),axis=axis)
        ahi=np.sum(np.abs(self.hi),axis=axis)
        # gamma grows with n enough to dominate all reduction/abs errors.
        return ArrayI(_down(lo - 2*g*alo),_up(hi + 2*g*ahi))


def iv(x)->ArrayI:
    return x if isinstance(x,ArrayI) else ArrayI.point(x)


def dot_fixed(a:ArrayI,w:np.ndarray)->ArrayI:
    """Dense row-vector @ exact-point weight matrix, with a certified ball.

    a (..., K), w (K, M). The matrix product follows an ordinary binary64
    dot-product graph with n products and n reductions. Flat/nonfinite
    operations are rejected; GPU/TF32 tensor cores are out of scope.
    """
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=2 or w.shape[0]!=a.lo.shape[-1]:
        raise ValueError('matrix dimensions')
    c,r=a.center_radius()
    y=c@w
    ab=np.abs(c)@np.abs(w)
    radius=r@np.abs(w)
    g=_gamma(w.shape[0])
    _check(y,ab,radius)
    # Sum of per-coordinate input-radius effects plus forward dot error.
    # Inflate to dominate error in the nonnegative bound dot calculations.
    b=_up((radius + g*ab) * (1+4*g))
    return ArrayI(_down(y-b),_up(y+b))


def dot_pair(a:ArrayI,b:ArrayI,*,scale:ArrayI|None=None)->ArrayI:
    """All i,j inner products between two [T,K] interval matrices."""
    if a.lo.ndim!=2 or b.lo.ndim!=2 or a.lo.shape[-1]!=b.lo.shape[-1]:
        raise ValueError('pair dot shapes')
    ac,ar=a.center_radius();bc,br=b.center_radius()
    y=ac@bc.T
    ab=np.abs(ac)@np.abs(bc).T
    radius=ar@np.abs(bc).T + np.abs(ac)@br.T + ar@br.T
    g=_gamma(ac.shape[-1]);_check(y,ab,radius)
    bound=_up((radius+g*ab)*(1+5*g))
    value=ArrayI(_down(y-bound),_up(y+bound))
    return value if scale is None else value*scale


def constant_vector(t:torch.Tensor)->np.ndarray:
    if not t.is_floating_point() or not bool(torch.isfinite(t).all()):
        raise ValueError('nonfinite weights')
    return t.detach().cpu().to(torch.float64).numpy().copy()


def _norm_point(state:ArrayI,weight:np.ndarray,bias:np.ndarray,eps:float)->ArrayI:
    d=state.lo.shape[-1]
    mean=state.sum(axis=-1)
    centered=state-ArrayI(mean.lo[...,None],mean.hi[...,None])*(1./d)
    var=centered.square().sum(axis=-1)
    var=ArrayI(var.lo[...,None],var.hi[...,None])*(1./d)
    den=(var+eps).sqrt().reciprocal()
    return centered * den * weight + bias


def _norm(state:ArrayI,weight:np.ndarray,bias:np.ndarray,eps:float)->ArrayI:
    """Center-radius LayerNorm using the exact variance-away-from-zero bound.

    LN has Jacobian 2-norm at most ||gamma||_inf/sqrt(eps+||Pc||^2/d).
    Every x in a token box satisfies ||x-c||<=E; along any segment from
    c, ||P(c+t(x-c))||>=max(0,||Pc||-E). Thus a token-dependent Lipschitz
    factor encloses the vector output around the verified point LN(c).
    This avoids independent-interval dependency blow-up in the shared mean.
    """
    c,_ = state.center_radius()
    point=_norm_point(ArrayI.point(c),weight,bias,eps)
    d=c.shape[-1]
    maxgamma=Fraction.from_float(float(np.max(np.abs(weight))))
    epsi=Fraction.from_float(float(eps))
    if epsi<=0:
        raise ValueError('expected positive LayerNorm epsilon')
    pads=[]
    for k in range(c.shape[0]):
        row=c[k]
        r=[max(abs(Fraction.from_float(float(row[j]))-Fraction.from_float(float(state.lo[k,j]))),
               abs(Fraction.from_float(float(state.hi[k,j]))-Fraction.from_float(float(row[j]))))
           for j in range(d)]
        E=sqrt_upper(sum((a*a for a in r),Fraction(0)),digits=20)
        mu=sum((Fraction.from_float(float(x)) for x in row),Fraction(0))/d
        Pc=sqrt_lower(sum(((Fraction.from_float(float(x))-mu)**2 for x in row),Fraction(0)),digits=20)
        floor=max(Fraction(0),Pc-E)
        denominator=sqrt_lower(epsi+floor*floor/d,digits=20)
        pad=maxgamma*E/denominator
        pads.append(_from_fraction(pad,upward=True))
    P=np.asarray(pads,dtype=np.float64)[:,None]
    return ArrayI(_down(point.lo-P),_up(point.hi+P))

def _exp_point_nonpositive(x:np.ndarray)->ArrayI:
    if np.any(x>0) or np.any(x < -64):
        raise ValueError('exp Taylor domain: require -64 <= score <= 0')
    # A single scaling factor preserves all vector shapes and exact powers of 2.
    k=0
    v=float(np.min(x))
    while v/2**k < -.25:
        k+=1
    t=x/(2.**k)
    # Horner evaluated by outward interval operations. All t in [-1/4,0].
    coeff=Fraction(1,factorial(EXP_DEGREE))
    p=ArrayI.box(_from_fraction(coeff,upward=False),_from_fraction(coeff,upward=True))
    x0=iv(t)
    for degree in range(EXP_DEGREE-1,-1,-1):
        coef=Fraction(1,factorial(degree))
        p=p*x0+ArrayI.box(_from_fraction(coef,upward=False),_from_fraction(coef,upward=True))
    # Taylor remainder <= (1/4)^(N+1), since exp(xi)<=1 for xi<=0.
    tail=2.**(-2*(EXP_DEGREE+1))
    result=ArrayI(_down(p.lo-tail),_up(p.hi+tail))
    for _ in range(k):
        result=result.square()
    return result


def exp_interval(a:ArrayI)->ArrayI:
    if np.any(a.lo>0) or np.any(a.hi>0):
        raise ValueError('shifted softmax score must be nonpositive')
    lo=_exp_point_nonpositive(a.lo)
    hi=_exp_point_nonpositive(a.hi)
    return ArrayI(lo.lo,hi.hi)


def _cdf_coefficient(n):
    c=Fraction((-1)**n,2**n*factorial(n)*(2*n+1))
    return ArrayI.box(_from_fraction(c,upward=False),_from_fraction(c,upward=True))


def _cdf_prefactor():
    l,h=pi_interval()
    p=ArrayI.box(_from_fraction(2*l,upward=False),_from_fraction(2*h,upward=True))
    return p.sqrt().reciprocal()

_CDF_COEFFS=[_cdf_coefficient(k) for k in range(CDF_DEGREE)]
_CDF_SCALE=_cdf_prefactor()
_CDF_TAIL=_from_fraction(Fraction(4**(2*CDF_DEGREE+1),2**CDF_DEGREE*factorial(CDF_DEGREE)*(2*CDF_DEGREE+1)),upward=True)


def _cdf_point(c:np.ndarray)->ArrayI:
    if np.any(np.abs(c)>CDF_SAFE):
        raise ValueError('CDF polynomial range')
    x=ArrayI.point(c)
    sq=x.square()
    p=_CDF_COEFFS[-1]
    for coef in reversed(_CDF_COEFFS[:-1]):
        p=p*sq+coef
    phi=iv(.5)+_CDF_SCALE*p*x
    return ArrayI(_down(phi.lo-_CDF_TAIL),_up(phi.hi+_CDF_TAIL))


def gelu_interval(x:ArrayI)->ArrayI:
    c,r=x.center_radius()
    near=np.abs(c)<=CDF_SAFE
    # For large |c|, use 0<=Phi<=1 and a global 5/4 Lipschitz bound.
    c_eval=np.where(near,c,0.)
    p=_cdf_point(c_eval)
    result=iv(c_eval)*p
    base_lo=np.where(near,result.lo,np.minimum(0.,c))
    base_hi=np.where(near,result.hi,np.maximum(0.,c))
    pad=_up(1.25*r)
    return ArrayI(_down(base_lo-pad),_up(base_hi+pad))


def _simplex_weighted(scores:ArrayI,values:ArrayI)->ArrayI:
    """Certified causal attention contraction via the probability simplex.

    For score perturbations ||s-s0||_infty<=delta, the softmax map
    satisfies ||softmax(s)-softmax(s0)||_1 <= delta (global Jacobian
    operator norm). A convex value average responds by at most
    delta*max_j|v0_j| in each coordinate; value uncertainty contributes
    at most max_j radius(v_j), independent of causal-prefix length.
    Exact, point-score softmax weights use our rational-Taylor exp layer.
    """
    n=scores.lo.shape[0]
    if values.lo.shape[0]!=n:
        raise ValueError('value sequence mismatch')
    score_c,score_r=scores.center_radius()
    v_c,v_r=values.center_radius()
    out_lo=np.empty_like(v_c);out_hi=np.empty_like(v_c)
    for i in range(n):
        c=score_c[i,:i+1]
        delta=float(_up(np.max(score_r[i,:i+1])))
        shift=float(np.max(c))
        # c-shift is an exact real constant modeled by outward IEEE intervals.
        shifted=ArrayI(_down(c-shift),_up(c-shift))
        # Ensure the tiny upper-rounding of the max term stays <=0 by
        # conservatively promoting its exp() enclosure to include 1.
        shifted=ArrayI(shifted.lo,np.minimum(shifted.hi,0.))
        pexp=exp_interval(shifted)
        total=pexp.sum(axis=-1)
        prob=pexp/total
        prob=ArrayI(np.maximum(0.,prob.lo),np.minimum(1.,prob.hi))
        weighted=_weight_values(ArrayI(prob.lo.reshape(1,-1),prob.hi.reshape(1,-1)),
                                ArrayI.point(v_c[:i+1]))
        valerr=np.max(v_r[:i+1],axis=0)
        response=_up(delta*np.max(np.abs(v_c[:i+1]),axis=0))
        total_err=_up(valerr+response)
        out_lo[i]=_down(weighted.lo[0]-total_err)
        out_hi[i]=_up(weighted.hi[0]+total_err)
    return ArrayI(out_lo,out_hi)


def _weight_values(probs:ArrayI,value:ArrayI)->ArrayI:
    # Probabilities in [m,n], values [n,head]. With interval uncertainty.
    ac,ar=probs.center_radius();bc,br=value.center_radius()
    mid=ac@bc
    err=ar@np.abs(bc)+np.abs(ac)@br+ar@br
    g=_gamma(ac.shape[1]);sumabs=np.abs(ac)@np.abs(bc)
    bound=_up((err+g*sumabs)*(1+5*g))
    return ArrayI(_down(mid-bound),_up(mid+bound))


@dataclass(frozen=True)
class FastTubes:
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
    method:str='scm10_ieee_roundoff_gamma_rational_nonlinear_exact_real'


def _extract(a:ArrayI):
    center=.5*a.lo+.5*a.hi
    radius=np.maximum(np.abs(_up(center-a.lo)),np.abs(_up(a.hi-center)))
    norms=np.linalg.norm(radius,axis=-1)
    # Round norm upwards using a high precision exact-rational sum on the
    # very small error intervals, avoiding opaque BLAS norm rounding.
    vals=[]
    for row in radius:
        sq=sum((Fraction.from_float(float(v))**2 for v in row),Fraction(0))
        vals.append(sqrt_upper(sq,digits=20))
    return torch.tensor(center,dtype=torch.float64),tuple(vals)


def _weights(block):
    d=block.hidden_size
    return {
        'wqkv': constant_vector(block.qkv.weight).T,
        'wo':constant_vector(block.attention_out.weight).T,
        'wu':constant_vector(block.mlp[0].weight).T,
        'wd':constant_vector(block.mlp[2].weight).T,
        'g1':constant_vector(block.norm1.weight),
        'b1':constant_vector(block.norm1.bias),
        'g2':constant_vector(block.norm2.weight),
        'b2':constant_vector(block.norm2.bias),
        'eps1':float(block.norm1.eps),'eps2':float(block.norm2.eps),
        'h':block.num_heads,'d':d//block.num_heads,
        'alpha':float(block.alpha),'beta':float(block.beta),
    }


def _block(x:ArrayI,w:dict)->tuple[ArrayI,ArrayI]:
    n=x.lo.shape[0];d=x.lo.shape[1];h=w['h'];dh=w['d']
    normed=_norm(x,w['g1'],w['b1'],w['eps1'])
    qkv=dot_fixed(normed,w['wqkv'])
    v_heads=[]
    invscale=iv(1.)/iv(dh).sqrt()
    for head in range(h):
        a=head*dh
        query=ArrayI(qkv.lo[:,a:a+dh],qkv.hi[:,a:a+dh])
        key=ArrayI(qkv.lo[:,d+a:d+a+dh],qkv.hi[:,d+a:d+a+dh])
        val=ArrayI(qkv.lo[:,2*d+a:2*d+a+dh],qkv.hi[:,2*d+a:2*d+a+dh])
        scores=dot_pair(query,key,scale=invscale)
        v_heads.append(_simplex_weighted(scores,val))
    attended=dot_fixed(ArrayI(np.concatenate([x.lo for x in v_heads],axis=1),
                              np.concatenate([x.hi for x in v_heads],axis=1)),w['wo'])
    middle=x*w['alpha']+attended*w['beta']
    n2=_norm(middle,w['g2'],w['b2'],w['eps2'])
    hidden=gelu_interval(dot_fixed(n2,w['wu']))
    output=middle*w['alpha']+dot_fixed(hidden,w['wd'])*w['beta']
    return middle,output


def fast_centered_tubes(model:nn.Module,source:InputEnclosure,*,
                         fixed_input_reinjection:bool=False,
                         max_seq:int=64,max_dim:int=128,max_visits:int=8)->FastTubes:
    """IEEE binary64 source-only interval program; excludes GPU inference.

    Exports a sufficient enclosure for every ideal-real RMP pre/mid/post
    state under the declared CPU numeric contract. No result at all is
    emitted on unsupported shapes, architectures or failed operations.
    """
    _validate_model(model,source,max_seq=max_seq,max_dim=max_dim,max_visits=max_visits)
    source_c=source.center.detach().cpu().numpy().astype(np.float64)
    if len(source.radii)!=source_c.shape[0]:
        raise ValueError('input radius lengths')
    rr=np.array([_from_fraction(Fraction(r),upward=True) for r in source.radii])[:,None]
    inp=ArrayI(_down(source_c-rr),_up(source_c+rr))
    state=inp
    pre,mid,post=[],[],[]
    perr,merr,oerr=[],[],[]
    specs={}
    for idx in model.module_indices:
        if idx not in specs:
            specs[idx]=_weights(model.blocks[idx])
        c,e=_extract(state);pre.append(c);perr.append(e)
        m,after=_block(state,specs[idx])
        c,e=_extract(m);mid.append(c);merr.append(e)
        state=after+inp if fixed_input_reinjection else after
        c,e=_extract(state);post.append(c);oerr.append(e)
    return FastTubes(tuple(pre),tuple(mid),tuple(post),tuple(perr),tuple(merr),tuple(oerr),
                     source.scope,source.frequency_sha256,max_seq,max_dim)


def ieee_preflight()->dict:
    """Verify basic binary64/gradual-underflow invariants for the CPU host.

    The BLAS kernel's internal arithmetic semantics are an explicit *assumption*
    and cannot be proven from NumPy's public runtime API; user results retain
    conditional IEEE status until a separately audited kernel is supplied.
    """
    import sys
    if np.dtype('float64').itemsize!=8 or sys.float_info.radix!=2:
        raise RuntimeError('unsupported non-binary64 CPU environment')
    if not (np.nextafter(np.float64(0.),np.float64(INF))>0.):
        raise RuntimeError('gradual underflow or nextafter unavailable')
    if float(np.sqrt(np.float64(4.)))!=2.:
        raise RuntimeError('sqrt preflight failed')
    if float(np.float64(1.)+np.float64(2**-52))<=1.:
        raise RuntimeError('round-to-nearest preflight failed')
    return {'ieee_binary64_preflight_passed':True,
            'blas_dot_gamma_assumed':True,
            'physical_torch_cuda_execution_certified':False,
            'numpy_version':np.__version__}
