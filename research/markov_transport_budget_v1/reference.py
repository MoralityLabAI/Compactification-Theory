"""MTB-01: exact affine Brenier drift and Markov-type hypercube obstruction.

Pure standard-library Fraction oracles. All finite probabilities and moments
are rational. Source OpenAI Math #327's general superreflexivity theorem
and #374's sharp 1/3 Brenier theorem are conditional external inputs.
The affine/refresh example does not reproduce the sharp 1/3 counterexample.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import math
from itertools import product

SCHEMA="mtb01.transport_markov_budget.v1"
SOURCE_PAPERS=(327,374)
BOX_CELLS=(
    (F(1),F(1),F(101,100),F(1),F(1,100)),
    (F(1),F(1),F(2),F(1),F(1,4)),
    (F(3,2),F(2),F(2),F(5,2),F(1,4)),
)
HORIZONS=(0,1,2,4,8,16)
CUBES=(4,8,16,32,64,128,256)
THRESHOLD=F(1,2)
MARGIN=F(1,5)


def rational(v):
    if isinstance(v,bool):
        raise ValueError("boolean is not a numeric parameter")
    return F(v)


def box_params(a,b,c,d,s):
    values=tuple(rational(z) for z in (a,b,c,d,s))
    if any(z<=0 for z in values[:4]) or not 0<=values[4]<=1:
        raise ValueError("positive rectangle scales and 0<=refresh<=1 required")
    return values


def refresh_retention(s,t):
    s=rational(s)
    if not 0<=s<=1 or type(t) is not int or t<0:
        raise ValueError("0<=s<=1 and nonnegative integer time required")
    return (1-s)**t


def rectangular_w2_squared(a,b,c,d):
    a,b,c,d,_=box_params(a,b,c,d,0)
    # Coordinatewise monotone coupling is optimal for product rectangles.
    return ((a-c)**2+(b-d)**2)/3


def source_same_transport_squared(a,b,s,t):
    a,b,_,_,s=box_params(a,b,a,b,s)
    return (a*a+b*b)*(1-refresh_retention(s,t))/6


def one_step_energy(a,b,s):
    a,b,_,_,s=box_params(a,b,a,b,s)
    return (a*a+b*b)*s/6


def cross_transport_squared(a,b,c,d,s,t):
    a,b,c,d,s=box_params(a,b,c,d,s)
    q=refresh_retention(s,t)
    return rectangular_w2_squared(a,b,c,d)+(a*c+b*d)*(1-q)/6


def independently_integrated_cross_squared(a,b,c,d,s,t):
    """Independent integration route using E[U²]=1/3 and E[UV]=1/4.

    With probability q, U_t=U_0; with probability 1-q, U_t is an
    independent fresh uniform variable. Each source coordinate
    follows the same independent refresh law.
    """
    a,b,c,d,s=box_params(a,b,c,d,s)
    q=refresh_retention(s,t)
    E_joint=q*F(1,3)+(1-q)*F(1,4)
    x=(c*c+a*a)/3-2*a*c*E_joint
    y=(d*d+b*b)/3-2*b*d*E_joint
    return x+y


def box_certificate(a,b,c,d,s,t):
    a,b,c,d,s=box_params(a,b,c,d,s)
    if type(t) is not int or t<0:
        raise ValueError("nonnegative time required")
    q=refresh_retention(s,t)
    w2=rectangular_w2_squared(a,b,c,d)
    cross=cross_transport_squared(a,b,c,d,s,t)
    independent=independently_integrated_cross_squared(a,b,c,d,s,t)
    one=one_step_energy(a,b,s)
    same=source_same_transport_squared(a,b,s,t)
    if cross!=independent or same>t*one:
        raise AssertionError("exact affine drift or Hilbert Markov type failed")
    # The transport triangle and the Hilbert Markov-type-2 budget
    # imply cross <= (sqrt(W2²)+sqrt(t E1))². We check this without
    # floating-point square roots using elementary Cauchy.
    if cross>2*w2+2*t*one:
        raise AssertionError("squared triangle-derived budget failed")
    return {
        "original_rectangle":[str(a),str(b)],
        "target_rectangle":[str(c),str(d)],
        "refresh_probability":str(s),"steps":t,
        "persistence_probability":str(q),
        "target_W2_squared":str(w2),
        "actual_transport_map_L2_squared":str(w2),
        "cross_drift_squared":str(cross),
        "independent_cross_integration":str(independent),
        "same_map_drift_squared":str(same),
        "one_step_same_map_energy":str(one),
        "Hilbert_type2_upper_squared":str(t*one),
        "generic_squared_triangle_upper":str(2*w2+2*t*one),
        "exact_affine_map_stability_power_one":True,
        "source_374_sharp_worst_case_exponent_reproduced":False,
    }


def clamp01(x):
    return min(F(1),max(F(),rational(x)))


def probability_above_threshold(scale,threshold=THRESHOLD):
    scale=rational(scale)
    threshold=rational(threshold)
    if scale<=0:
        raise ValueError("positive rectangle scale required")
    return clamp01(1-threshold/scale)


def threshold_flip_probability(a,c,s,t,threshold=THRESHOLD):
    a,c,s=rational(a),rational(c),rational(s)
    q=refresh_retention(s,t)
    p0=probability_above_threshold(a,threshold)
    p1=probability_above_threshold(c,threshold)
    same_uniform_flip=abs(p1-p0)
    independently_refreshed_flip=p0+p1-2*p0*p1
    return q*same_uniform_flip+(1-q)*independently_refreshed_flip


def threshold_margin_probability(a,threshold=THRESHOLD,margin=MARGIN):
    a=rational(a);threshold=rational(threshold);margin=rational(margin)
    if a<=0 or margin<=0:
        raise ValueError("positive rectangle scale and margin required")
    low=max(F(),threshold-margin)
    high=min(a,threshold+margin)
    return max(F(),high-low)/a


def margin_flip_certificate(a,b,c,d,s,t,threshold=THRESHOLD,margin=MARGIN):
    a,b,c,d,s=box_params(a,b,c,d,s)
    margin=rational(margin)
    if margin<=0:
        raise ValueError("positive decision margin required")
    flip=threshold_flip_probability(a,c,s,t,threshold)
    small=threshold_margin_probability(a,threshold,margin)
    drift=cross_transport_squared(a,b,c,d,s,t)
    bound=min(F(1),small+drift/(margin*margin))
    if flip>bound:
        raise AssertionError("margin/Markov classifier guarantee violated")
    return {
        "threshold":str(rational(threshold)), "margin":str(margin),
        "margin_event_probability":str(small),
        "exact_decision_flip_probability":str(flip),
        "Markov_margin_flip_upper":str(bound),
        "probability_source":"Lebesgue-uniform [0,1]² with independent refresh",
        "finite_neural_monitor_result":False,
    }


def cube_odd_probability(n,t):
    if type(n) is not int or n<4 or type(t) is not int or t<0:
        raise ValueError("hypercube dimension>=4 and time>=0 required")
    return (1-(1-F(2,n))**t)/2


def cube_moments(n,t):
    """Closed exact first/second moment of Hamming displacement."""
    p=cube_odd_probability(n,t)
    a=(1-F(2,n))**t
    b=(1-F(4,n))**t
    both=(1-2*a+b)/4
    mean=n*p
    second=n*p+n*(n-1)*both
    if mean<0 or second<0:
        raise AssertionError("hypercube moment is negative")
    if t==0 and (mean or second):
        raise AssertionError("zero-time cube displacement")
    if t==1 and (mean!=1 or second!=1):
        raise AssertionError("every simple-cube walk moves one bit at step one")
    return {"n":n,"t":t,
            "odd_coordinate_probability":str(p),
            "two_coordinates_both_odd":str(both),
            "mean_hamming":str(mean),
            "second_hamming_moment":str(second),
            "L1_Markov_type2_constant_squared_lower":str(second/t) if t else None,
            "Hilbert_squared_displacement":str(mean),
            "Hilbert_type2_upper":t,
            "type_3_over_2_Jensen_lower_float":
                float(mean)/(t**(F(2,3))) if t else 0.0}


def hamming_birth_death(n,t):
    """Independent n+1-state Markov chain for the Hamming distance."""
    cube_odd_probability(n,t)
    state=[F(1)]+[F()]*n
    for _ in range(t):
        nxt=[F()]*(n+1)
        for j,p in enumerate(state):
            if j<n: nxt[j+1]+=p*F(n-j,n)
            if j>0: nxt[j-1]+=p*F(j,n)
        state=nxt
    if sum(state)!=1 or any(x<0 for x in state):
        raise AssertionError("birth-death chain not a probability distribution")
    return {
        "mean":sum((j*p for j,p in enumerate(state)),F()),
        "second":sum((j*j*p for j,p in enumerate(state)),F()),
        "pmf":tuple(state)}


def hamming_full_cube(n,t):
    """Independent 2^n-state coordinate-flip transition oracle; n<=8."""
    cube_odd_probability(n,t)
    if n>8 or t>10:
        raise ValueError("full-state oracle capped at n<=8,t<=10")
    distribution={0:F(1)}
    for _ in range(t):
        nxt={}
        for mask,mass in distribution.items():
            for j in range(n):
                after=mask^(1<<j)
                nxt[after]=nxt.get(after,F())+mass/n
        distribution=nxt
    mean=sum((mask.bit_count()*p for mask,p in distribution.items()),F())
    second=sum((mask.bit_count()**2*p for mask,p in distribution.items()),F())
    return {"mean":mean,"second":second,"states":len(distribution)}


def hypercube_certificate(n,t):
    x=cube_moments(n,t)
    if n<=16 and t<=8:
        y=hamming_birth_death(n,t)
        if y["mean"]!=F(x["mean_hamming"]) or y["second"]!=F(x["second_hamming_moment"]):
            raise AssertionError("independent Hamming DP / closed moments disagree")
    if n<=8 and t<=8:
        z=hamming_full_cube(n,t)
        if z["mean"]!=F(x["mean_hamming"]) or z["second"]!=F(x["second_hamming_moment"]):
            raise AssertionError("full 2^n state chain / moment formula disagree")
    if F(x["Hilbert_squared_displacement"])>t:
        raise AssertionError("Hilbert Markov type2 constant-one bound failed")
    return x


def cube_n_step_bound(n):
    """Jensen lower bound on Markov-type-p for l1^n, with t=n."""
    x=hypercube_certificate(n,n)
    mean=F(x["mean_hamming"])
    constant=F(1,2)*(1-(1-F(2,n))**n)
    if mean!=n*constant:
        raise AssertionError("n-step first moment mismatch")
    # 1 - 2/n <= exp(-2/n); thus mean >=(1-exp(-2))*n/2.
    analytic_baseline=(1-math.exp(-2))/2
    if float(mean/n)+1e-14<analytic_baseline:
        raise AssertionError("dimension-free Jensen lower constant failed")
    return {
        "n":n,"time":n,
        "E_Hamming":str(mean),
        "E_Hamming_squared":x["second_hamming_moment"],
        "markov_type2_constant_squared_lower":x["L1_Markov_type2_constant_squared_lower"],
        "markov_type2_constant_lower_float":
            math.sqrt(float(F(x["L1_Markov_type2_constant_squared_lower"]))),
        "Jensen_type2_lower_float":float(mean)/math.sqrt(n),
        "Jensen_type3_over2_lower_float":float(mean)/(n**(F(2,3))),
        "dimension_free_mean_fraction_lower_float":analytic_baseline,
        "Hilbert_type2_constant_one_suffices":True,
    }


def source_report():
    boxes=[]
    for i,params in enumerate(BOX_CELLS):
        for t in HORIZONS:
            row=box_certificate(*params,t)
            row.update(cell=i,margin=margin_flip_certificate(*params,t))
            boxes.append(row)
    cube_rows=[]
    for n in CUBES:
        times=sorted(set((1,2,4,8,n)))
        for t in times:
            cube_rows.append(hypercube_certificate(n,t))
    bounds=[cube_n_step_bound(n) for n in CUBES]
    return {
        "schema":SCHEMA,
        "status":"EXACT_AFFINE_TRANSPORT_MARKOV_CONTROL",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_papers":list(SOURCE_PAPERS),
        "rectangle_cells":boxes,
        "cube_moment_cells":cube_rows,
        "hypercube_n_step_lower_bounds":bounds,
        "affine_source":"uniform unit square",
        "target_rectangles":"independent positive diagonal Brenier scalings",
        "stationary_reversible_refresh":True,
        "general_type_transport_theorem_conditional_on_source":True,
        "source_327_superreflexivity_equivalence_reproved":False,
        "source_374_sharp_one_third_theorem_reproved":False,
        "trained_neural_checkpoint_evaluated":False,
        "probabilities_and_moments_exact_rationals":True,
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args(argv)
    report=source_report()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":report["status"],
        "boxes":[{"cell":x["cell"],"t":x["steps"],
                  "W2_squared":x["target_W2_squared"],
                  "cross_drift_squared":x["cross_drift_squared"],
                  "flip":x["margin"]["exact_decision_flip_probability"]}
                 for x in report["rectangle_cells"]
                 if x["steps"] in (0,1,4,16)],
        "hypercube":[{"n":x["n"],"type2_L1_constant_lower":
                       x["markov_type2_constant_lower_float"],
                       "Jensen_p1_5_lower":
                       x["Jensen_type3_over2_lower_float"]}
                     for x in report["hypercube_n_step_lower_bounds"]],
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
