"""LSV-16: stage-one-to-LN2 sound centered perturbation transport (Arb).

An ambient l-infinity box on the 3888 stage-one coordinates is propagated
through all three remaining recurrent blocks with componentwise error envelopes.
The clean trajectory is evaluated in exact-real Arb arithmetic, while all
weights and the stage-one anchor are exact binary floats embedded in R.

Only a mathematically verified upper bound may be called a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from flint import arb, arb_mat

from validated_minor import (
    State, CertifiedBlock, exact_matrix, _jet_layernorm, _jet_gelu,
)

D = 48
TOKENS = 81


def absupper(x):
    """A point-ball above |x| (including every value in x)."""
    return abs(x).upper()


def nonnegative_upper(x):
    """Bound a nonnegative real expression by an exact Arb upper endpoint."""
    return max(arb(0), x.upper())


def positive_matrix_upper(mat):
    return arb_mat([[nonnegative_upper(v) for v in row] for row in mat.tolist()])


def abs_matrix_upper(mat):
    return arb_mat([[absupper(v) for v in row] for row in mat.tolist()])


def as_dyadic(x):
    """Convert exact Arb binary endpoint to Python Fraction without rounding."""
    a,e = x.man_exp()
    e = int(e)
    m = int(a)
    return Fraction(m*(2**e),1) if e>=0 else Fraction(m,2**(-e))


def upper_decimal(x, digits=20):
    """Round upwards to a compact decimal rational. Sound for all finite x."""
    if not 1 <= digits <= 60:
        raise ValueError('decimal digit cap invalid')
    v = as_dyadic(x.upper())
    scale=10**digits
    ceilv= -(-v.numerator*scale // v.denominator)
    return Fraction(ceilv,scale)


def centered_abs_bounds(row):
    """Bounds on |P*error| from absolute errors in individual coordinates.

    Exact inequality: |e_j - mean(e)| <= (1-1/d)b_j+sum(k!=j)b_k/d.
    The L2 perturbation bound uses ||Pe||_2 <= ||e||_2 and is usually tighter
    than taking the RMS of the coordinatewise centered errors.
    """
    d=len(row)
    if d<2:
        raise ValueError('LayerNorm feature dimension >=2 required')
    b=[nonnegative_upper(x) for x in row]
    total=sum(b,arb(0))
    cj=[nonnegative_upper((d-2)*b[j]/d+total/d) for j in range(d)]
    rms=nonnegative_upper((sum((v*v for v in b),arb(0))/d).sqrt())
    return cj,rms


def layernorm_difference_bounds(centers, radii, norm):
    """Sound componentwise bound on LN(x+e)-LN(x) with |e_j|<=r_j.

    Finite bound requires s(clean) > ||e||_2/sqrt(d). The bound is local
    in clean variance but universal over the input box; shared translations
    are removed algebraically by the projection P.
    """
    center_rows=centers.tolist(); bound_rows=radii.tolist()
    if len(center_rows)!=len(bound_rows) or not center_rows:
        raise ValueError('input shape mismatch')
    gamma=[absupper(arb(float(g))) for g in norm.weight.detach().cpu().tolist()]
    eps=arb(float(norm.eps))
    if len(gamma)!=len(center_rows[0]) or any(len(row)!=len(gamma) for row in center_rows):
        raise ValueError('LayerNorm channel shape mismatch')
    d=len(gamma)
    result=[]
    min_ratio=None
    for row,rad in zip(center_rows,bound_rows):
        mean=sum(row,arb(0))/d
        u=[x-mean for x in row]
        s=(eps+sum((x*x for x in u),arb(0))/d).sqrt().lower()
        centered,rms=centered_abs_bounds(rad)
        if s<=rms:
            raise ValueError('local LayerNorm denominator floor not positive: shrink input box')
        gap=(s-rms).lower()
        if gap <= 0:
            raise ValueError('roundoff lost positive LayerNorm denominator floor')
        # Bound |gamma_j| [ c_j/(s-R) + |u_j| R/(s*(s-R)) ].
        out=[nonnegative_upper(gamma[j]*(centered[j]/gap + absupper(u[j])*rms/(s*gap)))
             for j in range(d)]
        result.append(out)
    return arb_mat(result)


@dataclass
class Enclosure:
    center: arb_mat
    rad: arb_mat


class CertifiedTransport:
    def __init__(self, model):
        self.net=CertifiedBlock(model.block)
        self.at0=abs_matrix_upper(self.net.t0)
        self.at1=abs_matrix_upper(self.net.t1)
        self.ac0=abs_matrix_upper(self.net.c0)
        self.ac1=abs_matrix_upper(self.net.c1)

    def linear(self, state, w, ab, bias):
        center=state.center*w
        if bias is not None:
            center+=bias
        return Enclosure(center,positive_matrix_upper(state.rad*ab))

    def transpose(self, st):
        return Enclosure(st.center.transpose(),st.rad.transpose())

    def gelu(self, st):
        # 1+max_x |x phi(x)| <= 1 + 1/sqrt(2*pi*e) < 5/4.
        center=_jet_gelu(State(st.center,())).value
        bound=positive_matrix_upper(st.rad*arb(5)/4)
        return Enclosure(center,bound)

    def add(self, a,b):
        return Enclosure(a.center+b.center,positive_matrix_upper(a.rad+b.rad))

    def ln(self, st, norm):
        center=_jet_layernorm(State(st.center,()),norm).value
        bound=layernorm_difference_bounds(st.center,st.rad,norm)
        return Enclosure(center,bound)

    def _first_half(self, st):
        t=self.linear(self.transpose(st),self.net.t0,self.at0,self.net.t0b)
        t=self.linear(self.gelu(t),self.net.t1,self.at1,self.net.t1b)
        return self.ln(self.add(st,self.transpose(t)),self.net.ln1)

    def pre_second_ln(self,st):
        first=self._first_half(st)
        c=self.linear(first,self.net.c0,self.ac0,self.net.c0b)
        c=self.linear(self.gelu(c),self.net.c1,self.ac1,self.net.c1b)
        return self.add(first,c)

    def block(self,st):
        return self.ln(self.pre_second_ln(st),self.net.ln2)

    def from_stage_one(self,stage1,radius):
        """Propagate an ambient 81x48 l∞ box until final pre-LN2 bus."""
        r=Fraction(str(radius))
        if not 0<r<=Fraction(1):
            raise ValueError('positive ambient radius <=1 required')
        rb=(arb(r.numerator)/r.denominator).upper()
        center=exact_matrix(stage1)
        if center.nrows()!=TOKENS or center.ncols()!=D:
            raise ValueError('stage-one state must be 81 x 48')
        box=Enclosure(center,arb_mat([[rb for _ in range(D)] for _ in range(TOKENS)]))
        for _ in range(2):
            box=self.block(box)
        return self.pre_second_ln(box)


def goal_cell_radius(out,cell):
    """Return certified max coordinate error after centering at final LN2 bus."""
    if not (0<=cell<TOKENS):
        raise ValueError('invalid goal cell')
    row=out.rad.tolist()[cell]
    cent,_=centered_abs_bounds(row)
    val=max(as_dyadic(x.upper()) for x in cent)
    return val
