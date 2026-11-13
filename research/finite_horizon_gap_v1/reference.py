"""FHC-01 exact finite controls for horizon-dependent compactification.

Only the Python standard library is required. These are exact-rational
checks of Gaussian 4th moments, covering, distortion-transfer, and finite
packing. They DO NOT construct the compact doubling witness of OpenAI #098.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path


def rational(value: int | str | Fraction) -> Fraction:
    return Fraction(value)


def fourth_gaussian_moment(weights) -> Fraction:
    """Wick expansion of E[(sum_i a_i Z_i)^4] for iid N(0,1) variables."""
    a = tuple(map(rational, weights))
    total = Fraction()
    for i, j, k, l in itertools.product(range(len(a)), repeat=4):
        pairings = int(i == j and k == l) + int(i == k and j == l) + int(i == l and j == k)
        total += a[i] * a[j] * a[k] * a[l] * pairings
    return total


def gaussian_isometry_moment_identity(weights) -> bool:
    a = tuple(map(rational, weights))
    square_norm = sum((v * v for v in a), Fraction())
    return fourth_gaussian_moment(a) == 3 * square_norm * square_norm


def packing_upper_cardinality(d: int, distortion) -> Fraction:
    """Volume packing bound N <= (1+2D)^d for any d-dimensional norm."""
    D = rational(distortion)
    if d < 1 or D < 1:
        raise ValueError("need integer target dimension >=1 and distortion >=1")
    return (1 + 2 * D) ** d


def first_forbidden_equilateral_size(d: int, distortion) -> int:
    """Smallest N strictly above the universal volume packing limit."""
    value = packing_upper_cardinality(d, distortion)
    return value.numerator // value.denominator + 1


def equilateral_packing_obstruction(n: int, d: int, distortion) -> bool:
    """A True result rigorously rules out D-embedding of N equilateral points."""
    if n < 2:
        raise ValueError("need >=2 equilateral points")
    return rational(n) > packing_upper_cardinality(d, distortion)


def exact_line_equilateral_distortion(n: int) -> int:
    """Optimal 1D distortion of N equilateral points: D=N-1."""
    if n < 2:
        raise ValueError("need >=2 distinct points")
    return n - 1


def net_transfer_distortion(delta, epsilon, distortion) -> Fraction:
    """D*(delta+2eps)/(delta-2eps) for approximating a finite witness."""
    delta, epsilon, D = map(rational, (delta, epsilon, distortion))
    if delta <= 0 or epsilon < 0 or 2 * epsilon >= delta or D < 1:
        raise ValueError("require delta>2epsilon>=0 and D>=1")
    return D * (delta + 2 * epsilon) / (delta - 2 * epsilon)


def robust_net_threshold(delta, distortion, forbidden_distortion) -> Fraction:
    """If A forbids Dprime, every eps-net with eps <= returned bound forbids D."""
    delta, D, Dprime = map(rational, (delta, distortion, forbidden_distortion))
    if delta <= 0 or D < 1 or Dprime <= D:
        raise ValueError("require delta>0 and 1<=D<Dprime")
    return delta * (Dprime - D) / (2 * (Dprime + D))


def doubling_cover_bound(lam: int, diameter, epsilon) -> dict:
    """Finite epsilon-net count bound using intrinsic doubling of compact K.

    Start with open ball radius 2*diameter, halve until radius <= eps/2.
    A strict epsilon-cover then follows from its centers in K.
    """
    R, eps = map(rational, (diameter, epsilon))
    if lam < 1 or R <= 0 or eps <= 0:
        raise ValueError("require Lambda>=1 and R,epsilon>0")
    r = 2 * R
    steps = 0
    while r > eps / 2:
        r /= 2
        steps += 1
    return {"steps": steps, "net_count_upper": lam ** steps,
            "final_radius": r}


def squared_l2_distance(a, b) -> Fraction:
    if len(a) != len(b):
        raise ValueError("vectors must have matching dimension")
    return sum(((rational(x)-rational(y))**2 for x,y in zip(a,b)), Fraction())


def fourth_l4_distance(a, b) -> Fraction:
    if len(a) != len(b):
        raise ValueError("vectors must have matching dimension")
    return sum(((rational(x)-rational(y))**4 for x,y in zip(a,b)), Fraction())


def exact_l2_to_l4_certificate(source, encoded, scale, distortion) -> dict:
    """Check every pair after raising positive norm bounds to 4th powers.

    Normalized claim: scale*||x-y||_2 <= ||code(x)-code(y)||_4
                         <= scale*D*||x-y||_2 .
    """
    source, encoded = list(source), list(encoded)
    s, D = map(rational, (scale, distortion))
    if len(source) < 2 or len(source) != len(encoded) or s <= 0 or D < 1:
        raise ValueError("need matched finite samples, positive scale, D>=1")
    if len(set(map(tuple, source))) != len(source):
        raise ValueError("source states must be distinct")
    pairs = []
    for i in range(len(source)):
        for j in range(i+1,len(source)):
            a = squared_l2_distance(source[i],source[j])
            b = fourth_l4_distance(encoded[i],encoded[j])
            lower = s**4 * a**2
            upper = (s*D)**4 * a**2
            pairs.append({"pair":[i,j], "passed":lower<=b<=upper,
                          "source_distance_squared":str(a),
                          "target_distance_fourth":str(b),
                          "lower_fourth":str(lower),
                          "upper_fourth":str(upper)})
    return {"passed":all(p["passed"] for p in pairs),
            "pairs_checked":len(pairs),
            "failed_pairs":[p for p in pairs if not p["passed"]]}


def nearest_net_point(x, net):
    """Choose closest point in a finite rational line net; ties by index."""
    return min(net, key=lambda q:abs(rational(x)-rational(q)))


def coarse_embedding_1d_certificate(states, net, images: dict, epsilon,
                                    distortion) -> dict:
    """Exact FHC-3 additive-bound check on a finite line-metric example."""
    states = tuple(map(rational,states))
    net = tuple(map(rational,net))
    eps,D = map(rational,(epsilon,distortion))
    if not states or not net or eps<0 or D<1:
        raise ValueError("nonempty states/net, eps>=0, D>=1")
    if set(net)-set(images):
        raise ValueError("missing encoded net coordinates")
    if any(min(abs(x-q) for q in net)>eps for x in states):
        raise ValueError("net does not epsilon-cover supplied states")
    for q in net:
        for r in net:
            target=abs(rational(images[q])-rational(images[r]))
            if not abs(q-r)<=target<=D*abs(q-r):
                raise ValueError("net map is not a normalized D-embedding")
    rows=[]
    for x,y in itertools.combinations(states,2):
        qx,qy=nearest_net_point(x,net),nearest_net_point(y,net)
        original=abs(x-y)
        observed=abs(rational(images[qx])-rational(images[qy]))
        lo=max(Fraction(),original-2*eps)
        hi=D*(original+2*eps)
        rows.append((lo<=observed<=hi, x, y))
    return {"passed":all(row[0] for row in rows),
            "pairs_checked":len(rows),
            "violations":[[str(x),str(y)] for ok,x,y in rows if not ok]}



def shift_observation_matrix(width: int, horizon: int) -> list[list[Fraction]]:
    """O_T rows e_(t mod width), t=0,...,T-1, for a cyclic shift."""
    if width < 1 or horizon < 0:
        raise ValueError("width>=1 and horizon>=0 required")
    return [[Fraction(int(j == t % width)) for j in range(width)]
            for t in range(horizon)]


def exact_rational_rank(rows) -> int:
    """Fractional row elimination, proving rank for a finite supplied matrix."""
    a = [[rational(v) for v in row] for row in rows]
    if not a:
        return 0
    n = len(a[0])
    if any(len(row) != n for row in a):
        raise ValueError("ragged matrix")
    pivot_row = 0
    for col in range(n):
        next_pivot = next((i for i in range(pivot_row,len(a))
                          if a[i][col] != 0),None)
        if next_pivot is None:
            continue
        a[pivot_row],a[next_pivot]=a[next_pivot],a[pivot_row]
        pivot=a[pivot_row][col]
        a[pivot_row]=[v/pivot for v in a[pivot_row]]
        for k in range(len(a)):
            if k!=pivot_row and a[k][col]!=0:
                factor=a[k][col]
                a[k]=[u-factor*v for u,v in zip(a[k],a[pivot_row])]
        pivot_row+=1
        if pivot_row==len(a):
            break
    return pivot_row


def finite_shift_rank_frontier(width: int, horizons) -> list[dict]:
    """Linear encoder dimension min(T,width) to preserve all future readouts."""
    out=[]
    for T in horizons:
        rank=exact_rational_rank(shift_observation_matrix(width,T))
        if rank!=min(width,T):
            raise AssertionError("shift-register observability rank mismatch")
        out.append({"horizon":T, "required_linear_dimension":rank})
    return out


def control_report() -> dict:
    """Reference cases fixed before using neural data."""
    weights=[Fraction(1,2),Fraction(-2,3),Fraction(5,4)]
    assert gaussian_isometry_moment_identity(weights)
    assert first_forbidden_equilateral_size(2,Fraction(6,5))==12
    assert equilateral_packing_obstruction(12,2,Fraction(6,5))
    assert not equilateral_packing_obstruction(11,2,Fraction(6,5))
    assert exact_line_equilateral_distortion(4)==3
    assert robust_net_threshold(1,1,2)==Fraction(1,6)
    assert net_transfer_distortion(1,Fraction(1,6),1)==2
    assert doubling_cover_bound(4,1,Fraction(1,8))["net_count_upper"]==1024
    cert=exact_l2_to_l4_certificate([[0,0],[1,0],[0,1]],
                                    [[0,0],[1,0],[0,1]],
                                    Fraction(1,2), 2)
    assert cert["passed"]
    net=coarse_embedding_1d_certificate(
        list(range(5)), [0,2,4],
        {Fraction(0):0,Fraction(2):2,Fraction(4):4},1,1)
    assert net["passed"]
    frontier=finite_shift_rank_frontier(8,[1,2,4,8,16])
    assert [r["required_linear_dimension"] for r in frontier]==[1,2,4,8,8]
    return {
        "status":"CONTROL_ONLY",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gaussian_fourth_moment":str(fourth_gaussian_moment(weights)),
        "gaussian_expected_fourth_moment":
            str(3*sum(w*w for w in weights)**2),
        "equilateral_N_first_forbidden_d2_D6over5":12,
        "equilateral_one_dim_min_D_at_N4":3,
        "robust_epsilon_at_delta1_D1_Dprime2":"1/6",
        "doubling_example_net_cardinality_upper":1024,
        "exact_geometry_pairs_checked":cert["pairs_checked"],
        "finite_net_pairs_checked":net["pairs_checked"],
        "shift_register_linear_frontier_width8":frontier,
        "constructs_compact_doubling_witness":False,
        "runs_trained_models":False,
    }


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args(argv)
    out=control_report()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
