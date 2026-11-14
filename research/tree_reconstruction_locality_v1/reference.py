"""KSR-01: exact noisy-tree reconstruction and leaf-counter monitoring.

Sources: OpenAI Math #229 (three-state tree reconstruction) and #236
(factor-of-IID free Ising threshold). This code independently verifies
finite broadcast-tree statistics, Bayes root decoding, and sparse
leaf-observation moment identities. It does not reproduce the original
infinite-tree or factor-of-IID constructions.

Requires Python 3.10+ standard library only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path


def validate(q: int, b: int, eigenvalue) -> Fraction:
    if type(q) is not int or not 2 <= q <= 5:
        raise ValueError("number of labels must be integer 2..5")
    if type(b) is not int or b < 2 or b > 8:
        raise ValueError("branching must be integer 2..8")
    lam = Fraction(eigenvalue)
    if not -Fraction(1, q - 1) <= lam <= 1:
        raise ValueError("symmetric q-state channel has invalid eigenvalue")
    return lam


def transition(q: int, eigenvalue) -> tuple[tuple[Fraction, ...], ...]:
    lam = validate(q, 2, eigenvalue)
    return tuple(tuple((1 - lam) / q + (lam if i == j else 0)
                       for j in range(q)) for i in range(q))


def moment_factor(b: int, eigenvalue, depth: int) -> Fraction:
    """Second moment of leaf-mean score divided by source-score variance.

    M_L=(b*lambda)^(-L) sum_{|v|=L} [q*1{spin_v=j}-1].
    E[M_L**2]=(q-1)*V_L, E[M_L*g_j(root)]=q-1.
    """
    if type(b) is not int or b < 2 or type(depth) is not int or depth < 0:
        raise ValueError("branching >=2, depth>=0")
    lam = Fraction(eigenvalue)
    if lam == 0 and depth:
        raise ValueError("zero eigenvalue has no normalized leaf-sum estimator")
    if depth == 0:
        return Fraction(1)
    t = b * lam * lam
    value = Fraction(1)
    for _ in range(depth):
        value = value / t + Fraction(b - 1, b)
    return value


def moment_closed(b: int, eigenvalue, depth: int) -> Fraction:
    lam=Fraction(eigenvalue)
    if not depth:
        return Fraction(1)
    if not lam:
        raise ValueError("lambda=0 and depth>0")
    t=b*lam**2
    return t**(-depth) + Fraction(b - 1, b)*sum(
        (t**(-j) for j in range(depth)), Fraction())


def dropout_moment_factor(b: int, eigenvalue, depth: int,
                          retention) -> Fraction:
    r=Fraction(retention)
    if not 0 < r <= 1:
        raise ValueError("retention must be in (0,1]")
    lam=Fraction(eigenvalue)
    if depth and not lam:
        raise ValueError("lambda=0 cannot normalize leaf estimator")
    if depth == 0:
        return Fraction(1, r)
    t=b*lam**2
    return moment_factor(b,lam,depth) + (1-r)/(r*t**depth)


def bayes_floor(q: int, b: int, eigenvalue,
                depth: int, retention=1) -> Fraction:
    lam=validate(q,b,eigenvalue)
    v=dropout_moment_factor(b,lam,depth,retention)
    return Fraction(1,q)+Fraction(q-1,q)/v


def supercritical_limit(q: int, b: int, eigenvalue) -> dict:
    lam=validate(q,b,eigenvalue)
    t=b*lam**2
    if t <= 1:
        raise ValueError("requires b*lambda**2 > 1")
    V=Fraction(b-1,b)*t/(t-1)
    floor=Fraction(1,q)+Fraction(q-1,q)/V
    return {"branching":b,"states":q,"lambda":str(lam),"t":str(t),
            "limiting_variance_factor":str(V),
            "limiting_recovery_lower_bound":str(floor),
            "limiting_advantage_over_guess":str(floor-Fraction(1,q))}




def sparse_schedule(q: int, b: int, eigenvalue, depth: int)->dict:
    """Set r_L=(b*lambda**2)^(-L), retaining lambda^(-2L) leaves in mean.

    When t>1 this is a valid retention probability for all depths.
    V_drop,L=V_L+1-t^(-L), tending to V_infty+1.
    """
    lam=validate(q,b,eigenvalue)
    if type(depth) is not int or depth<0:
        raise ValueError("nonnegative integer depth required")
    t=b*lam**2
    if t<=1:
        raise ValueError("sparse KS schedule applies only when t>1")
    r=t**(-depth)
    V=dropout_moment_factor(b,lam,depth,r)
    expected=r*b**depth
    if V != moment_factor(b,lam,depth) + 1-t**(-depth):
        raise AssertionError("sparse schedule moment algebra differs")
    limit=supercritical_limit(q,b,lam)
    Vinf=Fraction(limit["limiting_variance_factor"])+1
    floorinf=Fraction(1,q)+Fraction(q-1,q)/Vinf
    return {
        "q":q,"b":b,"lambda":str(lam),"t":str(t),"depth":depth,
        "retention_probability":str(r),
        "expected_observed_leaves":str(expected),
        "full_leaf_count":b**depth,
        "moment":str(V),
        "count_recovery_floor":str(bayes_floor(q,b,lam,depth,r)),
        "limiting_sparse_moment":str(Vinf),
        "limiting_sparse_recovery_floor":str(floorinf),
        "counter_memory_bits":bit_budget(q,b,depth,r)["persistent_bit_upper_bound"],
    }

def bit_budget(q: int, b: int, depth: int, retention=1) -> dict:
    if type(depth) is not int or depth < 0:
        raise ValueError("depth>=0 required")
    validate(q,b,1)
    r=Fraction(retention)
    if not 0 < r <= 1:
        raise ValueError("retention in (0,1] required")
    leaves=b**depth
    slots=q-1 if r==1 else q
    per_count_bits=leaves.bit_length()
    return {
        "depth":depth,"leaf_symbols":leaves,
        "counts_stored":slots,
        "bits_per_counter":per_count_bits,
        "persistent_bit_upper_bound":slots*per_count_bits,
        "raw_alphabet_symbols":leaves,
        "uses_root_observation":False,
    }


def _path_digits(index:int,b:int,depth:int)->tuple[int,...]:
    digits=[]
    for _ in range(depth):
        digits.append(index%b)
        index//=b
    return tuple(reversed(digits))


def leaf_pair_covariance_factor(b:int,depth:int,i:int,j:int,eigenvalue)->Fraction:
    """E[g(X_i)g(X_j)]/(q-1)=lambda^(2*(depth-LCA_depth))."""
    if (type(depth) is not int or depth < 0 or
            type(i) is not int or type(j) is not int or
            not 0 <= i < b**depth or not 0 <= j < b**depth):
        raise ValueError("invalid leaf pair")
    lam=Fraction(eigenvalue)
    pa=_path_digits(i,b,depth);pb=_path_digits(j,b,depth)
    lca=0
    for a,c in zip(pa,pb):
        if a!=c:
            break
        lca+=1
    return lam**(2*(depth-lca))


def covariance_oracle(b:int,eigenvalue,depth:int)->Fraction:
    """Independent pairwise LCA sum, exact check of leaf martingale moment."""
    if depth>5 or b**depth>100:
        raise ValueError("pairwise oracle limited to <=100 leaves")
    lam=Fraction(eigenvalue)
    if depth and not lam:
        raise ValueError("lambda=0 cannot normalize")
    leaves=b**depth
    total=sum((leaf_pair_covariance_factor(b,depth,i,j,lam)
               for i in range(leaves) for j in range(leaves)),Fraction())
    return total / ((b*lam)**(2*depth) if depth else 1)


def exact_broadcast(q:int,b:int,eigenvalue,depth:int,retention=1)->dict:
    """Enumerate ALL finite observed leaf words and exact Bayes MAP values.

    Unobserved symbols use None; independently retained leaf symbols have
    probability r, all observations use known length/position erasures.
    The Bayes full decoder may use the entire observed pattern.
    The compressed decoder receives only q label counts (q-1 when r=1).
    Root prior is uniform over q states.
    """
    lam=validate(q,b,eigenvalue)
    r=Fraction(retention)
    if not 0<r<=1:
        raise ValueError("retention in (0,1] required")
    if type(depth) is not int or not 0<=depth<=5:
        raise ValueError("depth in 0..5 required")
    n=b**depth
    alphabet=tuple(range(q)) + ((None,) if r!=1 else ())
    if len(alphabet)**n > 20000:
        raise ValueError("exact enumeration capped at 20000 leaf patterns")
    matrix=transition(q,lam)
    @lru_cache(maxsize=None)
    def message(level:int,obs:tuple):
        if level==0:
            return tuple(Fraction(1 if (obs[0] is None or obs[0]==a) else 0)
                         for a in range(q))
        width=b**(level-1)
        children=tuple(message(level-1,obs[k*width:(k+1)*width])
                       for k in range(b))
        out=[]
        for a in range(q):
            total=Fraction(1)
            for child in children:
                total*=sum((matrix[a][s]*child[s] for s in range(q)),
                           Fraction())
            out.append(total)
        return tuple(out)

    total_prob=Fraction()
    full_map=Fraction()
    count_joint=defaultdict(lambda:[Fraction() for _ in range(q)])
    root_score_cov=Fraction()
    score_second=Fraction()
    scaling=Fraction(1,1) if depth==0 else (b*lam)**depth
    if scaling==0:
        raise ValueError("normalized score undefined at lambda=0")
    for observed in itertools.product(alphabet,repeat=n):
        nseen=sum(v is not None for v in observed)
        mass=r**nseen*(1-r)**(n-nseen)
        if not mass:
            continue
        lk=message(depth,observed)
        joint=[lk[a]*mass/q for a in range(q)]
        total=sum(joint,Fraction())
        total_prob+=total
        full_map+=max(joint)
        counts=tuple(observed.count(i) for i in range(q))
        score=Fraction(q*counts[0]-nseen,1)/(r*scaling)
        score_second+=total*score**2
        root_score_cov+=sum((joint[a]*score*(q*(a==0)-1)
                             for a in range(q)),Fraction())
        target=count_joint[counts]
        for a in range(q):
            target[a]+=joint[a]
    if total_prob!=1:
        raise AssertionError(f"finite broadcast law normalized to {total_prob}")
    count_map=sum((max(joint) for joint in count_joint.values()),Fraction())
    V=dropout_moment_factor(b,lam,depth,r)
    if score_second!=(q-1)*V:
        raise AssertionError("exact posterior score variance vs analytic V mismatch")
    if root_score_cov!=q-1:
        raise AssertionError("exact posterior score/source covariance mismatch")
    bound=bayes_floor(q,b,lam,depth,r)
    if not Fraction(1,q)<=bound<=count_map<=full_map<=1:
        raise AssertionError("Cauchy recovery floor or MAP optimality violated")
    return {
        "q":q,"b":b,"lambda":str(lam),"depth":depth,
        "retention":str(r),"leaves":n,
        "observation_patterns":len(alphabet)**n,
        "nonzero_count_cells":len(count_joint),
        "moment_factor":str(V),
        "moment_factor_float":float(V),
        "score_second_moment":str(score_second),
        "score_root_covariance":str(root_score_cov),
        "count_lower_bound":str(bound),
        "count_lower_bound_float":float(bound),
        "count_MAP":str(count_map),
        "count_MAP_float":float(count_map),
        "full_leaf_MAP":str(full_map),
        "full_leaf_MAP_float":float(full_map),
        "storage":bit_budget(q,b,depth,r),
    }


def reference_report()->dict:
    specs=(
        (2,2,Fraction(4,5),1,1),
        (2,2,Fraction(4,5),2,1),
        (2,2,Fraction(4,5),3,1),
        (3,2,Fraction(4,5),1,1),
        (3,2,Fraction(4,5),2,1),
        (3,2,Fraction(4,5),3,1),
        (3,2,Fraction(1,2),2,1),
        (3,2,Fraction(1,2),3,1),
        (3,4,Fraction(1,2),1,1),
        (2,4,Fraction(1,2),1,1),
        (2,2,Fraction(4,5),3,Fraction(1,2)),
        (3,2,Fraction(4,5),2,Fraction(1,2)),
        (3,2,Fraction(4,5),2,Fraction(1,4)),
    )
    rows=[exact_broadcast(*args) for args in specs]
    for b in (2,3,4):
        for lam in (Fraction(1,2),Fraction(3,4)):
            for depth in range(4):
                if b**depth<=100 and moment_factor(b,lam,depth)!=covariance_oracle(b,lam,depth):
                    raise AssertionError("LCA covariance and recursion differ")
    limits=[supercritical_limit(q,2,Fraction(4,5)) for q in (2,3)]
    assert [x["limiting_recovery_lower_bound"] for x in limits]==["23/32","5/8"]
    sparsity=[
        sparse_schedule(q,2,Fraction(4,5),L)
        for q in (2,3) for L in (0,1,2,3,8,16)
    ]
    assert sparse_schedule(3,2,Fraction(4,5),16)["limiting_sparse_recovery_floor"]=="37/69"
    assert sparse_schedule(2,2,Fraction(4,5),16)["limiting_sparse_recovery_floor"]=="15/23"
    critical={
        "degree":5,"branching":4,"lambda":"1/2",
        "t":"1",
        "ising_FIID_boundary_under_source_236":True,
        "count_monitor_moment_at_depth_8":str(moment_factor(4,Fraction(1,2),8)),
        "count_monitor_bound_q3_depth8":
            str(bayes_floor(3,4,Fraction(1,2),8)),
    }
    return {
        "schema":"ksr01.exact_broadcast.v1",
        "status":"EXACT_FINITE_TREE_CONTROL",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_result_families":[229,236],
        "analytic_supercritical":limits,
        "sparse_schedule":sparsity,
        "critical_branch":critical,
        "cases":rows,
        "reproduces_source_229_full_infinite_tree_proof":False,
        "reproduces_source_236_infinite_factor_construction":False,
        "trains_neural_checkpoint":False,
        "observes_hidden_chain_of_thought":False,
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args(argv)
    receipt=reference_report()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    compact={
        "status":receipt["status"],
        "source_sha256":receipt["source_sha256"],
        "limits":receipt["analytic_supercritical"],
        "cases":[{k:r[k] for k in (
            "q","b","lambda","depth","retention",
            "count_lower_bound_float","count_MAP_float",
            "full_leaf_MAP_float")}
                 for r in receipt["cases"]],
    }
    print(json.dumps(compact,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
