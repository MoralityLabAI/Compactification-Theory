"""LSV-15: exact rational correlated-gauge quotient and transverse margins.

The input intervals must enclose the exact-real FINAL pre-LayerNorm input.
No floating-point inference, point-gradient proxy, or zero-width assumption
is involved in the rational replay. A shared arbitrary row translation is
eliminated by LayerNorm algebraically, before forming any interval hull.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import isqrt
from verify_serialized_minors import parse_ball


def as_q(text):
    if isinstance(text, bool):
        raise TypeError("boolean is not a rational")
    if isinstance(text, (Q, int)):
        return Q(text)
    if isinstance(text, str):
        return Q(text)
    raise TypeError("exact fractions, integers and decimal strings only")


@dataclass(frozen=True)
class Bound:
    lo: Q
    hi: Q

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("invalid interval")

    @classmethod
    def exact(cls, q):
        v = as_q(q)
        return cls(v, v)

    @classmethod
    def from_arb(cls, s):
        if isinstance(s, str) and "/" in s:
            # Exact-rational test fixtures; production Arb uses decimal balls.
            return cls.exact(Q(s))
        a, b = parse_ball(s)
        return cls(a, b)

    def __add__(self, b):
        b = b if isinstance(b, Bound) else Bound.exact(b)
        return Bound(self.lo+b.lo, self.hi+b.hi)

    __radd__ = __add__

    def __neg__(self):
        return Bound(-self.hi, -self.lo)

    def __sub__(self, b):
        return self + (-b if isinstance(b, Bound) else Bound.exact(-as_q(b)))

    def __mul__(self, b):
        b = b if isinstance(b, Bound) else Bound.exact(b)
        xs = (self.lo*b.lo, self.lo*b.hi, self.hi*b.lo, self.hi*b.hi)
        return Bound(min(xs), max(xs))

    __rmul__ = __mul__

    def inverse(self):
        if self.lo <= 0 <= self.hi:
            raise ValueError("interval denominator contains zero")
        return Bound(Q(1)/self.hi, Q(1)/self.lo)

    def __truediv__(self, b):
        b = b if isinstance(b, Bound) else Bound.exact(b)
        return self * b.inverse()

    def square(self):
        if self.lo <= 0 <= self.hi:
            return Bound(Q(0), max(self.lo*self.lo, self.hi*self.hi))
        return Bound(min(self.lo*self.lo, self.hi*self.hi),
                     max(self.lo*self.lo, self.hi*self.hi))

    def max_abs(self):
        return max(abs(self.lo), abs(self.hi))

    def min_abs(self):
        return Q(0) if self.lo <= 0 <= self.hi else min(abs(self.lo), abs(self.hi))


def center_rationals(xs):
    """The exact projection P=I-11^T/d; shared translations vanish."""
    xs = tuple(as_q(x) for x in xs)
    if len(xs) < 2:
        raise ValueError("feature dimension must exceed one")
    m = sum(xs, Q(0)) / len(xs)
    return tuple(x-m for x in xs)


def safe_sqrt_rational(x, decimal_digits=30):
    """Certified rational enclosure for sqrt(x) via integer square root."""
    x = x if isinstance(x, Bound) else Bound.exact(x)
    if x.lo <= 0 or decimal_digits < 4 or decimal_digits > 80:
        raise ValueError("strictly positive squared variance required")
    s = 10**decimal_digits
    qlo = x.lo.numerator*s*s // x.lo.denominator
    qhi = x.hi.numerator*s*s // x.hi.denominator
    lo = Q(isqrt(qlo), s)
    hi_int = isqrt(qhi)
    hi = Q(hi_int if hi_int*hi_int == qhi and x.hi.denominator == 1 else hi_int+1, s)
    # Even when qhi was rounded down to an integer before sqrt, +1 is safe.
    if hi*hi < x.hi:
        hi = Q(hi_int+1, s)
    assert lo*lo <= x.lo and hi*hi >= x.hi
    return Bound(lo, hi)


def layernorm_stat_bounds(u_intervals, eps):
    """Bounds for sigma=sqrt(eps+mean(u_j^2)), from captured centered inputs."""
    u = [Bound.from_arb(x) if isinstance(x, str) else x for x in u_intervals]
    if len(u) < 2:
        raise ValueError("need nontrivial feature dimension")
    eps=as_q(eps)
    if eps <= 0:
        raise ValueError("LayerNorm epsilon must be positive")
    v=Bound.exact(eps)
    for x in u:
        v += x.square()/len(u)
    return safe_sqrt_rational(v)


def normal_output_bounds(u_intervals, gamma_hex, beta_hex, eps_hex):
    """Independent exact-rational bound on clean LayerNorm output."""
    u=[Bound.from_arb(s) for s in u_intervals]
    if len(u)!=len(gamma_hex) or len(u)!=len(beta_hex):
        raise ValueError("shape mismatch")
    gamma=[Q.from_float(float.fromhex(h)) for h in gamma_hex]
    beta=[Q.from_float(float.fromhex(h)) for h in beta_hex]
    eps=Q.from_float(float.fromhex(eps_hex))
    centered_sum=Bound.exact(0)
    for item in u:
        centered_sum += item
    if not (centered_sum.lo<=0<=centered_sum.hi):
        raise ValueError("purported centered input has inconsistent feature sum")
    sigma=layernorm_stat_bounds(u,eps)
    out=[Bound.exact(g)*(v/sigma)+b for g,v,b in zip(gamma,u,beta)]
    return out,sigma


def head_intervals(state, weight_hex, bias_hex):
    if not weight_hex or len(bias_hex)!=len(weight_hex) or any(len(row)!=len(state) for row in weight_hex):
        raise ValueError("bad head dimensions")
    rows=[]
    for row,b in zip(weight_hex,bias_hex):
        out=Bound.exact(Q.from_float(float.fromhex(b)))
        for w,x in zip(row,state):
            out += x*Q.from_float(float.fromhex(w))
        rows.append(out)
    return rows


def certificate_for_case(case, weights, radii):
    """Replay fully rational, checkpoint-source-locked class invariance bounds.

    A record supplies rigorous centered-preactivation intervals produced by
    Arb; this independent verifier rederives all standard-library inequalities.
    """
    if len(case['u_intervals'])!=48:
        raise ValueError("expected 48 centered features")
    gamma=weights['gamma_hex']; beta=weights['beta_hex']
    w=weights['head_weight_hex']; b=weights['head_bias_hex']
    clean,sigma=normal_output_bounds(case['u_intervals'],gamma,beta,weights['epsilon_hex'])
    logits=head_intervals(clean,w,b)
    winner=case['winner_digit']; rival=case['rival_digit']
    if type(winner)!=int or type(rival)!=int or not 0<=winner<10 or not 0<=rival<10 or rival==winner:
        raise ValueError("bad goal pair")
    old_wins=all(logits[winner].lo > logits[k].hi for k in range(10) if k!=winner)
    old_second=all(logits[rival].lo > logits[k].hi for k in range(10) if k not in (winner,rival))
    result={'clean_top1_validated':old_wins,'clean_top2_validated':old_wins and old_second,
            'sigma_lower_bound':str(sigma.lo),'sigma_upper_bound':str(sigma.hi),
            'transverse_radii':{}}
    if not old_wins:
        raise ValueError("source-locked winner cannot be rigorously established")
    if not old_second:
        raise ValueError("source-locked runner up cannot be rigorously established")
    for radius in radii:
        r=as_q(radius)
        if r <= 0:
            raise ValueError("transverse radius must be positive")
        if r>=sigma.lo:
            result['transverse_radii'][str(radius)]={
                'top1_certified':False,'top2_certified':False,
                'min_top1_slack':None,'min_top2_slack':None,
                'reason':'radius_not_below_certified_clean_sigma',
            }
            continue
        # Reverse-triangle bound on augmented norm (u/sqrt(d),sqrt(eps)).
        # Arbitrary shared gauge shifts do not enter this bound.
        deltas=[abs(Q.from_float(float.fromhex(g)))*r/(sigma.lo-r)*
                (1+Bound.from_arb(s).max_abs()/sigma.lo)
                for g,s in zip(gamma,case['u_intervals'])]
        head=[[Q.from_float(float.fromhex(x)) for x in row] for row in w]
        slacks={}
        for k in range(10):
            if k==winner:continue
            # Exact rational interval of clean winner-versus-k logit margin.
            clean_gap_lo=(logits[winner]-logits[k]).lo
            distortion=sum(abs(head[winner][j]-head[k][j])*deltas[j] for j in range(48))
            slacks[str(k)]=str(clean_gap_lo-distortion)
        top1=all(Q(v)>0 for v in slacks.values())
        second_slacks={}
        for k in range(10):
            if k in (winner,rival):continue
            clean_gap_lo=(logits[rival]-logits[k]).lo
            distortion=sum(abs(head[rival][j]-head[k][j])*deltas[j] for j in range(48))
            second_slacks[str(k)]=str(clean_gap_lo-distortion)
        top2=top1 and all(Q(v)>0 for v in second_slacks.values())
        result['transverse_radii'][str(radius)]={
            'top1_certified':top1,'top2_certified':top2,
            'min_top1_slack':str(min(map(Q,slacks.values()))),
            'min_top2_slack':str(min(map(Q,second_slacks.values()))),
        }
    return result


def replay_receipt(receipt, source_lock, reference, require_complete=True):
    """Independent standard-library audit; rejects fabricated result counters."""
    if receipt.get('schema')!='LSV-15/gauge-quotient-transverse-v1':
        raise ValueError('bad receipt schema')
    if receipt['checkpoint_sha256']!=source_lock['checkpoint_sha256'] or receipt['source_revision']!=source_lock['revision']:
        raise ValueError('checkpoint source lock mismatch')
    if receipt['roster_sha256']!=reference['roster_sha256']:
        raise ValueError('roster mismatch')
    if len(reference['cases'])!=64 or reference['rigorously_verified_rank_four_cases']!=64:
        raise ValueError('expected 64 verified source-locked cases')
    if require_complete and len(receipt['cases'])!=64:
        raise ValueError('need complete frozen roster')
    radii=receipt['radii']
    if not radii or len(set(radii))!=len(radii):
        raise ValueError('invalid radius roster')
    counts={r:{'top1':0,'top2':0} for r in radii}
    for idx,case in enumerate(receipt['cases']):
        baseline=reference['cases'][idx]
        if case['case_index']!=idx or case['selected_goal_cell']!=baseline['selected_goal_cell'] or case['winner_digit']!=baseline['winner_digit'] or case['rival_digit']!=baseline['rival_digit']:
            raise ValueError('source-locked goal selection altered')
        result=certificate_for_case(case,receipt['weights'],radii)
        if result!=case['certificate']:
            raise ValueError(f'case {idx}: claimed certificate differs from exact replay')
        for radius in radii:
            counts[radius]['top1']+=int(result['transverse_radii'][radius]['top1_certified'])
            counts[radius]['top2']+=int(result['transverse_radii'][radius]['top2_certified'])
    if counts!=receipt['counts']:
        raise ValueError('summary counts differ from exact replay')
    return counts
