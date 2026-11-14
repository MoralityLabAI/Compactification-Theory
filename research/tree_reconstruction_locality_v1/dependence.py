"""KSR-01C: exact finite coarse-state dependence and robust MAP floors.

No trained checkpoint or external theorem is needed by this module.
Fractions certify finite discrete laws. A native model's float decisions
can supply those laws but do not become real-arithmetic neural proofs.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache


def probability_row(values):
    row = tuple(F(x) for x in values)
    if not row or any(x < 0 for x in row) or sum(row) != 1:
        raise ValueError("expected a normalized nonnegative probability row")
    return row


def total_variation(a, b):
    a, b = probability_row(a), probability_row(b)
    if len(a) != len(b):
        raise ValueError("probability alphabets differ")
    return sum((abs(x-y) for x,y in zip(a,b)), F())/2


def coarse_dependence(labels, rows, q):
    """Independent children given H; parents H are uniform on this cohort.

    Per label c, J_c=E[p_H p_H^T|c], while the coarse IID surrogate is
    E[p_H|c] E[p_H|c]^T. Their difference is a covariance matrix.
    """
    if type(q) is not int or q < 2 or not labels or len(labels) != len(rows):
        raise ValueError("nonempty matched cohort and q>=2 required")
    if any(type(c) is not int or c not in range(q) for c in labels):
        raise ValueError("parent label outside alphabet")
    rows = [probability_row(row) for row in rows]
    if any(len(row) != q for row in rows):
        raise ValueError("child row outside alphabet")
    out = []
    weighted = F()
    for c in range(q):
        indices = [i for i,z in enumerate(labels) if z == c]
        if not indices:
            out.append({"label":c,"parents":0,"transition":None,
                        "diagonal_excess":None,"witness":None})
            continue
        n = len(indices)
        mean = tuple(sum((rows[i][j] for i in indices),F())/n for j in range(q))
        joint = tuple(tuple(sum((rows[i][a]*rows[i][b] for i in indices),F())/n
                            for b in range(q)) for a in range(q))
        excess = sum((joint[j][j]-mean[j]**2 for j in range(q)),F())
        alternate = sum((sum((rows[i][j]-mean[j])**2 for j in range(q))
                         for i in indices),F())/n
        if excess != alternate or excess < 0:
            raise AssertionError("conditional covariance identity failed")
        best = F(); witness = None
        for at,i in enumerate(indices):
            for k in indices[at+1:]:
                tv = total_variation(rows[i],rows[k])
                if tv > best:
                    best = tv
                    witness = {"parent_indices":[i,k],"child_law_TV":str(tv),
                               "left":[str(x) for x in rows[i]],
                               "right":[str(x) for x in rows[k]]}
        if (excess == 0) != (best == 0):
            raise AssertionError("pairwise constant kernel equivalence failed")
        weighted += F(n,len(labels))*excess
        out.append({"label":c,"parents":n,
                    "transition":[str(x) for x in mean],
                    "sibling_joint":[[str(x) for x in row] for row in joint],
                    "diagonal_excess":str(excess),
                    "conditional_same_label_excess_float":float(excess),
                    "max_same_parent_label_child_TV":str(best),"witness":witness})
    return {"states":q,"parents":len(labels),"by_parent_label":out,
            "weighted_sibling_diagonal_excess":str(weighted),
            "coarse_sibling_independence_exact":weighted == 0,
            "missing_parent_labels":[r["label"] for r in out if r["parents"]==0]}


def symmetric_projection(transition):
    """Equal-row least-squares symmetric-channel fit, available for all rows."""
    q = len(transition)
    if q < 2 or any(row is None for row in transition):
        return {"available":False,"reason":"unobserved_parent_label"}
    T = [probability_row(row) for row in transition]
    if any(len(row)!=q for row in T):
        raise ValueError("transition matrix must be square")
    lam = (sum(T[i][i] for i in range(q))-1)/(q-1)
    P = [[(1-lam)/q + (lam if i==j else 0) for j in range(q)]
         for i in range(q)]
    return {"available":True,"lambda":str(lam),
            "matrix":[[str(x) for x in row] for row in P],
            "max_row_TV":str(max(total_variation(a,b) for a,b in zip(T,P)))}


def histogram_convolve(left, right):
    if not left or not right:
        raise ValueError("empty probability law")
    q = len(next(iter(left)))
    out = defaultdict(F)
    for a,pa in left.items():
        for b,pb in right.items():
            if len(a)!=q or len(b)!=q:
                raise ValueError("histogram alphabet mismatch")
            if pa < 0 or pb < 0:
                raise ValueError("negative histogram probability")
            out[tuple(x+y for x,y in zip(a,b))] += pa*pb
    return dict(out)


def iid_histogram_power(law, copies):
    if type(copies) is not int or copies < 1 or not law:
        raise ValueError("positive number of independent copies required")
    q = len(next(iter(law)))
    out = {(0,)*q:F(1)}
    for _ in range(copies):
        out = histogram_convolve(out,law)
    return out


def finite_map_floor(joint, q):
    """Arbitrary prior/dependence: MAP >= sum pi_j^2 + sum Cov_j^2/Var_j.

    joint[(root_label, count_tuple)] is any finite normalized law. A zero
    score variance contributes zero. Also retain the prior-majority bound.
    Exact MAP here refers to this declared finite law, not new test roots.
    """
    if type(q) is not int or q < 2 or not joint:
        raise ValueError("finite joint law and q>=2 required")
    if any(p < 0 for p in joint.values()) or sum(joint.values()) != 1:
        raise ValueError("joint law must normalize")
    prior = [F() for _ in range(q)]
    cells = defaultdict(lambda:[F() for _ in range(q)])
    means = [F() for _ in range(q)]; second = means.copy(); cross = means.copy()
    for (c,count),weight in joint.items():
        if type(c) is not int or c not in range(q) or len(count)!=q:
            raise ValueError("invalid root or summary alphabet")
        if any(type(v) is not int or v<0 for v in count):
            raise ValueError("histogram counts must be nonnegative integers")
        prior[c] += weight; cells[count][c] += weight
        for j in range(q):
            means[j] += weight*count[j]
            second[j] += weight*count[j]**2
        cross[c] += weight*count[c]
    vars_ = [second[j]-means[j]**2 for j in range(q)]
    covs = [cross[j]-prior[j]*means[j] for j in range(q)]
    gains = []
    for v,c in zip(vars_,covs):
        if v < 0 or (v==0 and c!=0):
            raise AssertionError("invalid covariance/variance pair")
        gains.append(c*c/v if v else F())
    collision = sum(p*p for p in prior)
    raw = collision+sum(gains,F())
    best_prior = max(prior); bound = max(best_prior,raw)
    accuracy = sum((max(row) for row in cells.values()),F())
    if not 0 <= best_prior <= bound <= accuracy <= 1:
        raise AssertionError("dependence-robust posterior bound failed")
    return {"root_prior":[str(x) for x in prior],
            "majority_prior":str(best_prior),"collision_baseline":str(collision),
            "score_variances":[str(x) for x in vars_],
            "root_score_covariances":[str(x) for x in covs],
            "explained_root_variance_terms":[str(x) for x in gains],
            "covariance_floor":str(bound),"covariance_floor_float":float(bound),
            "count_MAP":str(accuracy),"count_MAP_float":float(accuracy),
            "histogram_cells":len(cells),
            "decoder_fit_to_same_finite_prior":True}


def shared_innovation_moment(b, lam, depth, rho):
    """Children share one channel draw with probability rho at each vertex."""
    lam,rho = F(lam),F(rho)
    if type(b) is not int or b<2 or type(depth) is not int or depth<0:
        raise ValueError("b>=2 and nonnegative depth required")
    if not 0<lam<=1 or not 0<=rho<=1:
        raise ValueError("positive eigenvalue and rho in [0,1] required")
    v = F(1)
    extra = F(b-1,b)*(1+rho*(1-lam*lam)/(lam*lam))
    for _ in range(depth):
        v = v/(b*lam*lam)+extra
    return v


def shared_innovation_joint(q, b, lam, depth, rho):
    """Exact histogram law for the sibling-correlated positive channel.

    At each node choose: independent child labels (1-rho), or one
    channel label copied to all b children (rho). Later subtree noise
    remains independent given the chosen child states.
    """
    lam,rho = F(lam),F(rho)
    if type(q) is not int or not 2<=q<=4 or b not in (2,3,4):
        raise ValueError("finite oracle requires q=2..4, b=2..4")
    if type(depth) is not int or not 0<=depth<=3 or b**depth>16:
        raise ValueError("finite oracle limited to depth<=3 and <=16 leaves")
    shared_innovation_moment(b,lam,depth,rho)
    matrix = [[(1-lam)/q+(lam if a==s else 0) for s in range(q)] for a in range(q)]
    @lru_cache(maxsize=None)
    def recurse(a, level):
        if level==0:
            return {tuple(int(j==a) for j in range(q)):F(1)}
        children = [recurse(s,level-1) for s in range(q)]
        one_child = defaultdict(F)
        common = defaultdict(F)
        for s,dist in enumerate(children):
            for count,p in dist.items():
                one_child[count] += matrix[a][s]*p
            for count,p in iid_histogram_power(dist,b).items():
                common[count] += matrix[a][s]*p
        independent = iid_histogram_power(dict(one_child),b)
        all_counts = set(independent)|set(common)
        return {c:(1-rho)*independent.get(c,F())+rho*common.get(c,F())
                for c in all_counts}
    joint = {}
    for a in range(q):
        for count,p in recurse(a,depth).items():
            if p:
                joint[a,count] = p/q
    if sum(joint.values())!=1:
        raise AssertionError("shared-innovation law did not normalize")
    return joint


def exact_controls():
    simple = coarse_dependence([0,0],[(1,0),(0,1)],2)
    if simple["weighted_sibling_diagonal_excess"]!="1/2":
        raise AssertionError("coarse-state mixture counterexample changed")
    q,b,lam,depth,rho = 3,4,F(1,2),1,F(1)
    joint = shared_innovation_joint(q,b,lam,depth,rho)
    stats = finite_map_floor(joint,q)
    v = shared_innovation_moment(b,lam,depth,rho)
    naive_v = shared_innovation_moment(b,lam,depth,0)
    naive_bound = F(1,q)+F(q-1,q)/naive_v
    corrected = F(1,q)+F(q-1,q)/v
    if not (naive_bound==F(5,7) and F(stats["count_MAP"])==F(2,3)
            and corrected==F(1,2) and naive_bound>F(stats["count_MAP"])):
        raise AssertionError("dependent-sibling falsification fixture changed")
    return {"status":"EXACT_DEPENDENCE_CONTROL",
            "same_label_latent_mixture":simple,
            "correlated_children":{"q":q,"b":b,"lambda":str(lam),
                "rho":str(rho),"depth":depth,"naive_IID_floor":str(naive_bound),
                "true_count_MAP":stats["count_MAP"],
                "dependence_corrected_floor":str(corrected),
                "corrected_moment":str(v),"naive_moment":str(naive_v)},
            "requires_external_229_or_236":False}


if __name__ == "__main__":
    import json
    print(json.dumps(exact_controls(),indent=2,sort_keys=True))
