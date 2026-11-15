"""OLM-11: exact rational sharp top-k and top-1 SAE selection cells.

All inputs are finite rational weights/scales/anchors. Floats must be
converted via Fraction.from_float to preserve their exact binary values.
No torch, NumPy, interval heuristics or neural-model dependencies.
"""
from fractions import Fraction as Q
from math import isqrt


def rational(x):
    return Q.from_float(x) if isinstance(x,float) else Q(x)


def affine_normalized_encoder(weights,bias,mean,scale):
    """Return physical-state affine row normals and offsets.

    a_i(h) = dot(v_i,h) + c_i, with
      v_ij = weight_ij/scale_j
      c_i = bias_i-dot(v_i,mean)
    """
    w=tuple(tuple(rational(x) for x in row) for row in weights)
    if not w or not w[0]:
        raise ValueError('nonempty encoder required')
    n=len(w[0]); m=len(w)
    if any(len(row)!=n for row in w) or len(bias)!=m or len(mean)!=n or len(scale)!=n:
        raise ValueError('inconsistent encoder/normalization shapes')
    mu=tuple(map(rational,mean))
    sig=tuple(map(rational,scale))
    if any(z<=0 for z in sig):
        raise ValueError('normalization scales must be positive')
    out=tuple(tuple(w[i][j]/sig[j] for j in range(n)) for i in range(m))
    offset=tuple(rational(bias[i])-
                 sum((out[i][j]*mu[j] for j in range(n)),Q(0))
                 for i in range(m))
    return out,offset


def scores_at(rows,intercepts,anchor):
    if len(rows)!=len(intercepts) or not rows or len(anchor)!=len(rows[0]):
        raise ValueError('wrong source anchor or offsets')
    h=tuple(map(rational,anchor))
    return tuple(Q(intercepts[i])+
                 sum((a*b for a,b in zip(row,h)),Q(0))
                 for i,row in enumerate(rows))


def squared_norm(vector):
    return sum((v*v for v in vector),Q(0))


def distance_squared(margin,normal):
    """Exact squared distance to a strict affine half-space's boundary.

    None represents a constant positive inequality without a boundary.
    """
    margin=rational(margin)
    if margin<=0:
        raise ValueError('strictly positive score margin required')
    norm=squared_norm(normal)
    if norm<0:
        raise ArithmeticError('negative squared norm')
    return None if norm==0 else margin*margin/norm


def dyadic_sqrt_bracket(squared_radius,bits=40):
    """Integer-certified dyadic [floor(sqrt(q)),ceil(sqrt(q))] bracket."""
    q=rational(squared_radius)
    if q<0 or not isinstance(bits,int) or not 1<=bits<=256:
        raise ValueError('nonnegative squared radius and 1..256 bits required')
    numerator=q.numerator<<(2*bits)
    s=isqrt(numerator//q.denominator)
    lower=Q(s,1<<bits)
    upper=lower if lower*lower==q else Q(s+1,1<<bits)
    assert lower*lower<=q<=upper*upper
    return lower,upper


def exact_cell(scores,rows,selected,cache=None):
    """Sharp positive selected-vs-unselected cell distance, squared.

    Only a strict *set* of selected identities is fixed. Ties within the
    selected set are permitted. The top-1 policy needs a singleton call.
    """
    m=len(scores)
    if m<2 or len(rows)!=m or not selected:
        raise ValueError('invalid finite selection problem')
    selected=tuple(sorted(set(selected)))
    if any(type(i)!=int or i<0 or i>=m for i in selected):
        raise ValueError('selected index out of range')
    rest=tuple(j for j in range(m) if j not in selected)
    if not rest:
        raise ValueError('proper nonempty subset required')
    if len({len(row) for row in rows})!=1:
        raise ValueError('feature dimension mismatch')
    sc=tuple(map(rational,scores))
    min_q=None;arg=None
    def update(gap,normal,type_,i,j):
        nonlocal min_q,arg
        if gap<=0:
            raise ValueError('non-strict score margin; certification abstains')
        d2=distance_squared(gap,normal)
        if d2 is not None and (min_q is None or d2<min_q):
            min_q=d2
            arg=(type_,i,j)
    for i in selected:
        update(sc[i],rows[i],'zero',i,None)
    norms={} if cache is None else cache
    for i in selected:
        for j in rest:
            gap=sc[i]-sc[j]
            key=(i,j) if i<j else (j,i)
            if key not in norms:
                vec=tuple(a-b for a,b in zip(rows[i],rows[j]))
                norms[key]=squared_norm(vec)
            d2=None if norms[key]==0 else gap*gap/norms[key]
            if gap<=0:
                raise ValueError('selected neuron fails strict score ordering')
            if d2 is not None and (min_q is None or d2<min_q):
                min_q=d2
                arg=('pair',i,j)
    if min_q is None:
        return {'radius_squared':None,'active_boundary':None,
                'dyadic_sqrt_lower':None,'dyadic_sqrt_upper':None}
    low,high=dyadic_sqrt_bracket(min_q)
    return {'radius_squared':min_q,'active_boundary':arg,
            'dyadic_sqrt_lower':low,'dyadic_sqrt_upper':high}


def topk_selection_certificate(rows,intercepts,anchor,k=4,cache=None):
    scores=scores_at(rows,intercepts,anchor)
    if not isinstance(k,int) or not 1<=k<len(scores):
        raise ValueError('k must lie in 1..m-1')
    ordered=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    positive=(scores[ordered[k-1]]>0)
    separation=(scores[ordered[k-1]]>scores[ordered[k]])
    if not (positive and separation) or scores[ordered[0]]<=scores[ordered[1]]:
        raise ValueError('ReLU top-k or maximum lacks strict positive separation')
    support=tuple(sorted(ordered[:k]))
    top=int(ordered[0])
    support_cert=exact_cell(scores,rows,support,cache)
    top_cert=exact_cell(scores,rows,(top,),cache)
    return {'top_index':top,'topk_indices':support,
            'top1':top_cert,'topk':support_cert,
            'top1_gap':scores[ordered[0]]-scores[ordered[1]],
            'topk_gap':scores[ordered[k-1]]-scores[ordered[k]],
            'min_selected_score':scores[ordered[k-1]],
            'exact_preactivations':scores}


def strict_guard(squared_radius,delta):
    """True iff closed L2 delta-ball lies inside strict selection cell."""
    delta=rational(delta)
    if delta<0:
        raise ValueError('nonnegative perturbation radius required')
    if squared_radius is None:
        return True
    return delta*delta<rational(squared_radius)


def sharp_boundary_displacement(rows,scores,certificate):
    """Exact vector that reaches the minimizing selection boundary."""
    active=certificate['active_boundary']
    if active is None or certificate['radius_squared'] is None:
        raise ValueError('no finite hyperplane boundary')
    mode,i,j=active
    normal=tuple(rows[i] if mode=='zero' else
                 (rows[i][v]-rows[j][v] for v in range(len(rows[i]))))
    margin=scores[i] if mode=='zero' else scores[i]-scores[j]
    norm=squared_norm(normal)
    delta=tuple(-margin*v/norm for v in normal)
    assert squared_norm(delta)==certificate['radius_squared']
    return delta


if __name__=='__main__':
    import json
    W=((2,0),(0,2),(1,1),(-1,0),(0,-1))
    rows,offs=affine_normalized_encoder(W,(1,1,0,0,0),(0,0),(1,1))
    c=topk_selection_certificate(rows,offs,(1,0),k=2)
    print(json.dumps({k:str(v) for k,v in c.items() if k!='exact_preactivations'},indent=2))
