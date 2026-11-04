"""LSV-09: exact rational task-relative robust margins for affine recurrence.

For general nonlinear models, a uniform *verified* derivative envelope is needed;
measured gradients at the clean point are diagnostics only, not certificates.
"""
from fractions import Fraction as Q


def exact(x):
    if isinstance(x,bool) or not isinstance(x,(int,str,Q)):
        raise TypeError('only integer, fraction and fraction strings allowed')
    return Q(x)


def square_matrix(values):
    a=tuple(tuple(exact(v) for v in row) for row in values)
    if not a or any(len(row)!=len(a) for row in a):
        raise ValueError('square nonempty matrix required')
    return a


def affine_map(a,x,b):
    n=len(a)
    if len(x)!=n or len(b)!=n:raise ValueError('affine dimension mismatch')
    return tuple(sum((a[i][j]*x[j] for j in range(n)),Q(0))+b[i] for i in range(n))


def row_mat(row,a):
    n=len(a)
    if len(row)!=n:raise ValueError('row mismatch')
    return tuple(sum((row[i]*a[i][j] for i in range(n)),Q(0)) for j in range(n))


def scored_affine_recurrence(A,b,x,readout,bias,steps,radii):
    """Exact certificate for a digit score after k repeated affine steps.

    Nominal scores are computed from the exact iterate; each pairwise score
    derivative is the exact row (W_winner-W_other) @ A^k.
    """
    a=square_matrix(A)
    n=len(a)
    b=tuple(map(exact,b));x=tuple(map(exact,x))
    r=tuple(map(exact,radii))
    w=tuple(tuple(map(exact,row)) for row in readout)
    bias=tuple(map(exact,bias))
    if len(x)!=n or len(b)!=n or len(r)!=n or any(v<0 for v in r):
        raise ValueError('invalid input or radius')
    if len(w)<2 or len(w)!=len(bias) or any(len(row)!=n for row in w):
        raise ValueError('bad output head')
    if type(steps) is not int or steps<0:raise ValueError('steps must be nonnegative integer')
    center=x
    for _ in range(steps):center=affine_map(a,center,b)
    scores=tuple(sum((t*v for t,v in zip(row,center)),bi) for row,bi in zip(w,bias))
    label=max(range(len(scores)),key=lambda i:scores[i])
    comparisons=[]
    stable=True
    for j in range(len(scores)):
        if j==label:continue
        margin=scores[label]-scores[j]
        derivative=tuple(p-q for p,q in zip(w[label],w[j]))
        for _ in range(steps):derivative=row_mat(derivative,a)
        decrease=sum((abs(q)*ri for q,ri in zip(derivative,r)),Q(0))
        good=margin>decrease
        stable=stable and good
        comparisons.append({'competitor':j,'margin':margin,
                            'max_margin_decrease':decrease,
                            'gradient':derivative,'strict_proved':good})
    return {'winner':label,'certified':stable,'scores':scores,
            'comparisons':comparisons}


def independent_corner_oracle(A,b,x,readout,bias,steps,radii):
    """Exhaust all vertices for small affine systems; tests proof calculation."""
    from itertools import product
    a=square_matrix(A)
    x=tuple(map(exact,x));r=tuple(map(exact,radii))
    b=tuple(map(exact,b));w=tuple(tuple(map(exact,row)) for row in readout)
    bias=tuple(map(exact,bias))
    outcomes=[]
    for signs in product((-1,1),repeat=len(x)):
        p=tuple(a0+ri*sgn for a0,ri,sgn in zip(x,r,signs))
        for _ in range(steps):p=affine_map(a,p,b)
        vals=[sum((ri*xi for ri,xi in zip(row,p)),bi) for row,bi in zip(w,bias)]
        winner=max(range(len(vals)),key=lambda i:vals[i])
        outcomes.append((winner,vals))
    return outcomes
