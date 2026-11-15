"""OLM-12: exact rational angular and decoder-noise cone certificates.

A fixed affine top-k SAE has physical score-row normals v_i. Its selected
top neuron t persists for all positive ray lengths along nonzero direction d
exactly when v_t.d >=0 and (v_t-v_j).d >=0 for every j != t.
The resulting homogeneous cone has a *sharp* spherical/angular and raw
relative-direction robustness threshold.

This is source-independent exact rational mathematics. Tensor conversion
and checkpoint hashing belong to the applied SMC-12 wrapper.
"""
from fractions import Fraction as Q
from math import isqrt


def rat(x):
    return Q.from_float(x) if isinstance(x,float) else Q(x)


def dot(a,b):
    if len(a)!=len(b):
        raise ValueError("inner product dimensions differ")
    return sum((rat(x)*rat(y) for x,y in zip(a,b)),Q(0))


def norm_sq(v):
    return dot(v,v)


def physical_encoder_normals(weights,train_scale):
    """Convert exact binary stored W,scale to normal rows in physical metric."""
    if not weights or not train_scale:
        raise ValueError("nonempty weight matrix and scale required")
    d=len(train_scale)
    scales=tuple(map(rat,train_scale))
    if any(s<=0 for s in scales):
        raise ValueError("training scales must be strictly positive")
    if any(len(row)!=d for row in weights):
        raise ValueError("encoder weight dimensions differ")
    return tuple(tuple(rat(value)/scales[j] for j,value in enumerate(row))
                 for row in weights)


def constraints(normals,selected):
    """Affine gap normals for strict positive top-k support at source anchor."""
    if not normals or len(normals)<2:
        raise ValueError("at least two SAE score rows needed")
    n=len(normals[0])
    if not n or any(len(row)!=n for row in normals):
        raise ValueError("malformed SAE row normals")
    chosen=tuple(sorted(set(selected)))
    if not chosen or len(chosen)==len(normals) or any(
            not isinstance(i,int) or i<0 or i>=len(normals) for i in chosen):
        raise ValueError("selected indices must be a proper nonempty set")
    others=[j for j in range(len(normals)) if j not in chosen]
    output=[]
    for i in chosen:
        output.append((('positive',i,None),tuple(map(rat,normals[i]))))
        for j in others:
            output.append((('competitor',i,j),tuple(
                rat(a)-rat(b) for a,b in zip(normals[i],normals[j]))))
    return tuple(output)


def cone_certificate(normals,selected,raw_direction):
    """Return the sharp squared relative raw-direction robustness threshold.

    If any gap slope is negative, the ray exits the selected-support cell
    at some finite positive amplitude. If all are nonnegative, it stays
    in the same strict selection region for EVERY finite positive amplitude,
    provided the source anchor has strict positive selected gaps.

    The minimum squared normalized slope, q², is the squared Euclidean
    distance to the complement hyperplane, divided by ||raw_direction||².
    Its arc angular margin is asin(sqrt(q²)); its unit-chord margin has
    squared value 2-2*sqrt(1-q²). All numerical certificates below use q²
    exactly, without transcendental rounding.
    """
    d=tuple(map(rat,raw_direction))
    rows=constraints(normals,selected)
    if len(d)!=len(rows[0][1]) or norm_sq(d)<=0:
        raise ValueError("nonzero correctly shaped physical direction required")
    norm_d=norm_sq(d)
    values=[];negative=[];zero=[]
    for name,n in rows:
        n2=norm_sq(n)
        if n2==0:
            continue                 # globally constant strict inequality
        m=dot(n,d)
        if m<0:
            negative.append(name)
        elif m==0:
            zero.append(name)
            values.append((Q(0),name))
        else:
            values.append((m*m/(n2*norm_d),name))
    if negative:
        status='outside'
        min_q2=None
        weakest=None
    elif not values:
        status='unconstrained'
        min_q2=None                # whole space is the cone
        weakest=None
    else:
        min_q2,weakest=min(values,key=lambda p:(p[0],p[1]))
        if not 0<=min_q2<=1:
            raise ArithmeticError("Cauchy-Schwarz inequality failed")
        status='boundary' if zero else 'interior'
    return {
        'status':status,
        'all_positive_ray_amplitudes_preserve_selected_set':not negative,
        'strict_angular_interior':status in ('interior','unconstrained'),
        'minimal_normalized_slope_squared':min_q2,
        'limiting_constraint':weakest,
        'negative_gap_slopes':tuple(negative),
        'zero_gap_slopes':tuple(zero),
        'nontrivial_constraints':len(values)+len(negative),
        'raw_direction_norm_squared':norm_d,
    }


def relative_raw_guard(cert,relative_radius):
    """Strict open radius around d, normalized by ||d||, within its cone."""
    r=rat(relative_radius)
    if r<0:
        raise ValueError("nonnegative radius required")
    if cert['status']=='outside':
        return False
    if r==0:
        return True
    q2=cert['minimal_normalized_slope_squared']
    return q2 is None or r*r<q2


def unit_chord_guard(cert,chord_radius):
    """Sharp inner spherical (unit-vector L2 chord) cone guard.

    For 0<δ²<2:
       δ² < 2-2sqrt(1-q²)  <=>  q² > δ²-δ⁴/4.
    The strict condition protects an *open* angular neighborhood.
    """
    d=rat(chord_radius)
    if d<0:
        raise ValueError("negative chord radius")
    if cert['status']=='outside':
        return False
    if d==0:
        return True
    q2=cert['minimal_normalized_slope_squared']
    if q2 is None:
        return True
    x=d*d
    if x>=2:
        return False
    return q2>x-x*x/4


def joint_normal_and_decoder_guard(cert,relative_decoder_error,relative_normal_error):
    """Sufficient source-normal + raw decoder error guard with exact rationals.

    Each nonzero old constraint normal n can move by ||Δn||<=η||n||,
    while decoder raw d moves by ||e||<=ρ||d||. If
           sqrt(q²) > ρ+η(1+ρ)
    then every new affine-gap slope (n+Δn).(d+e) remains strictly positive.
    This theorem alone does NOT prove starting-source score gaps after
    changing the encoder; those must be separately verified.
    """
    rho,eta=map(rat,(relative_decoder_error,relative_normal_error))
    if min(rho,eta)<0:
        raise ValueError("negative perturbation budget")
    if cert['status']=='outside':
        return False
    q2=cert['minimal_normalized_slope_squared']
    if q2 is None:
        return True
    margin=rho+eta*(1+rho)
    return q2>margin*margin


def dyadic_sqrt_bracket(q_squared,bits=40):
    q=rat(q_squared)
    if q<0 or not isinstance(bits,int) or not 1<=bits<=256:
        raise ValueError("invalid nonnegative squared quantity or bits")
    scale=1<<(2*bits)
    s=isqrt((q.numerator*scale)//q.denominator)
    lower=Q(s,1<<bits)
    upper=lower if lower*lower==q else Q(s+1,1<<bits)
    assert lower*lower<=q<=upper*upper
    return lower,upper


def sharp_boundary_adversary(normals,selected,raw_direction,cert,overshoot=Q(1,100)):
    """Construct exact raw decoder perturbation reaching limiting hyperplane.

    e=-(n.d)/(n.n) n; n.(d+e)==0, and n.(d+(1+ε)e)<0.
    The relative squared decoder perturbation is EXACTLY minimum q².
    """
    if cert['status']!='interior' or cert['limiting_constraint'] is None:
        raise ValueError("positive angular interior and finite limiter required")
    eps=rat(overshoot)
    if eps<=0:
        raise ValueError("positive overshoot required")
    d=tuple(map(rat,raw_direction))
    name=cert['limiting_constraint']
    relevant=dict(constraints(normals,selected))
    n=relevant[name]
    slope=dot(n,d)
    n2=norm_sq(n)
    if slope<=0 or n2<=0:
        raise ArithmeticError("invalid sharp limiting constraint")
    e=tuple(-slope*x/n2 for x in n)
    boundary=tuple(x+y for x,y in zip(d,e))
    outside=tuple(x+(1+eps)*y for x,y in zip(d,e))
    rel2=norm_sq(e)/norm_sq(d)
    if rel2!=cert['minimal_normalized_slope_squared']:
        raise AssertionError("raw relative squared distance not sharp")
    if dot(n,boundary)!=0 or dot(n,outside)>=0:
        raise AssertionError("adversarial cone crossing witness invalid")
    return {
        'limiting_constraint':name,
        'relative_raw_boundary_distance_squared':rel2,
        'exact_boundary_ray':boundary,
        'exact_overshoot_ray':outside,
        'boundary_gap_slope':dot(n,boundary),
        'outside_gap_slope':dot(n,outside),
    }




def feedback_matrix_self_selection(encoder_weights,decoder_weights):
    """Exact rational diagonal-dominance predicate for frozen SAE WD.

    W shape mxd, D shape dxm. K=W*D is mxm; atom t reinforces itself
    at all positive amplitudes under the EXACT raw physical ray
    scale⊙D[:,t] and physical score-row normals W/scale iff
       K[t,t] >= 0 and K[t,t] >= K[j,t] for every competitor j.
    Strict inequalities imply positive angular-cone interior.

    This is an algebraic condition on the encoder-decoder cross-talk
    matrix, independent of positive training scales in exact arithmetic.
    """
    W=tuple(tuple(map(rat,row)) for row in encoder_weights)
    D=tuple(tuple(map(rat,row)) for row in decoder_weights)
    if not W or not D or any(len(row)!=len(D) for row in W) or any(
        len(row)!=len(W) for row in D):
        raise ValueError("expected W(m,d) and D(d,m) matrices")
    m=len(W)
    result=[]
    for t in range(m):
        slopes=[
            sum((W[j][i]*D[i][t] for i in range(len(D))),Q(0))
            for j in range(m)]
        diag=slopes[t]
        others=[slopes[j] for j in range(m) if j!=t]
        support=(diag>=0 and all(diag>=v for v in others))
        interior=(diag>0 and all(diag>v for v in others))
        gaps=[diag]+[diag-v for v in others]
        result.append({
            'atom':t,'positive_ray_self_selection':support,
            'strict_interior':interior,'diagonal_score_exact':diag,
            'minimum_column_score_margin_exact':min(gaps),
        })
    return tuple(result)

if __name__=='__main__':
    import json
    a=((Q(1),Q(0)),(Q(0),Q(1)),(Q(-1),Q(0)))
    c=cone_certificate(a,(0,),(Q(2),Q(1)))
    print(json.dumps({
        'status':c['status'],
        'min_q2':str(c['minimal_normalized_slope_squared']),
        'limiting':c['limiting_constraint'],
        'raw_10pct':relative_raw_guard(c,Q(1,10)),
        'chord_10pct':unit_chord_guard(c,Q(1,10)),
    },indent=2))
