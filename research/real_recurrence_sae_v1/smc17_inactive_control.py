"""SMC-17 fixed-budget inactive SAE preactivation matching geometry.

The controls in this module depend only on clean SAE coefficients, frozen
dictionary matrices, SMC-16 selected physical displacements and seeded random
tangent vectors. They never inspect a Sudoku answer or a model output.
"""
import hashlib
import math

import torch

from smc16_selective_kill import (
    encode_one,projected_normal,physical_covectors,
    selective_vectors,TAU,CODE_TOL,ORTH_TOL,
)

FAMILIES={
    "smc17_novel_30":(202617030000,30),
    "smc17_novel_34":(202617040000,34),
}
N_RANDOM=128


def fresh_rosters(old,limit=64):
    """New exact-unique noncanonical puzzles, distinct from SMC13/15/16."""
    if limit not in (4,64):
        raise ValueError("SMC17 pilot/full roster only")
    from smc13_family import (
        new_full_solution,dig_unique,valid_solution,
        count_solutions,rectangle_count,fingerprint,
    )
    from smc15_active_necessity import fresh_rosters as old15
    from smc16_selective_kill import new_rosters as old16
    prior15,_=old15(old,limit=64)
    prior16,_=old16(old,limit=64)
    seen_y={tuple(r["target"]) for rows in
            list(old.values())+list(prior15.values())+list(prior16.values()) for r in rows}
    seen_x={tuple(r["x"]) for rows in
            list(old.values())+list(prior15.values())+list(prior16.values()) for r in rows}
    out={}
    for name,(base,holes) in FAMILIES.items():
        group=[]
        for j in range(limit):
            seed=base+j
            y=new_full_solution(seed)
            if not valid_solution(y) or rectangle_count(y)<=0:
                raise ValueError("SMC17 target is invalid/canonical")
            x=dig_unique(y,seed+1000000,holes)
            if x.count(0)!=holes or count_solutions(x,2)!=1:
                raise ValueError("SMC17 puzzle not unique or has wrong holes")
            if tuple(y) in seen_y or tuple(x) in seen_x:
                raise ValueError("SMC17 Sudoku cohort overlaps prior solutions")
            seen_y.add(tuple(y));seen_x.add(tuple(x))
            group.append({"seed":seed,"x":x,"target":y,
                          "holes":holes,"rectangles":rectangle_count(y)})
        out[name]=group
    return out,{k:fingerprint(rows) for k,rows in out.items()}


def random_tangent(pair,family,puzzle_seed,slot,feature,attempt,active):
    """A single fixed-seed 48d Gaussian projected into ker(active)."""
    msg=f"SMC17|{pair}|{family}|{puzzle_seed}|{slot}|{feature}|{attempt}"
    seed=int.from_bytes(hashlib.sha256(msg.encode()).digest()[:8],"big")&((1<<63)-1)
    vec=torch.randn(48,dtype=torch.float64,
                    generator=torch.Generator().manual_seed(seed))
    tangent=projected_normal(vec,active)
    return tangent,seed


def best_inactive_matched_sham(
        h,code,feature,sae,mean,scale,cov,delta,baseline,
        pair,family,puzzle_seed,slot,random_count=N_RANDOM):
    """Return the best label-blind sham among 131 preregistered candidates.

    The baseline, signed normalized LS vector and 128 new random vectors all
    have identical physical L2 dose and satisfy the same SMC16 reencoding
    constraints. The baseline is always first, and hence never worse.
    """
    if random_count!=N_RANDOM:
        raise ValueError("frozen 128 random controls required")
    if code.shape!=(96,) or cov.shape!=(96,48):
        raise ValueError("unexpected 96x48 SAE encoding geometry")
    active=torch.nonzero(code>0).flatten().tolist()
    if len(active)!=4 or feature not in active:
        raise ValueError("matched clean target must be among four active features")
    d=float(delta.norm())
    if d<=0 or abs(float(baseline.norm())-d)>ORTH_TOL:
        raise ValueError("base sham and kill physical doses differ")
    A=cov[active]
    inactive=[i for i in range(96) if i not in active]
    B=cov[inactive]
    if tuple(B.shape)!=(92,48):
        raise AssertionError("wrong inactive encoder block")
    # Full SVD makes the 44-dimensional tangent basis independent of
    # individual Gaussian directions and avoids an explicit Gram inversion.
    _,sing,vh=torch.linalg.svd(A,full_matrices=True)
    if float(sing.min())<=1e-10:
        raise ValueError("SMC17 active row rank obstruction")
    U=vh[4:].T.contiguous()
    if U.shape!=(48,44) or float((A@U).abs().max())>ORTH_TOL:
        raise ArithmeticError("nullspace basis violates active rows")
    b=B@delta
    C=B@U
    # Rank-deficient synthetic/corner geometries can make the default
    # LAPACK least-squares rank decision intermittent (occasionally returning
    # the zero solution). Use a fixed-tolerance SVD pseudoinverse ONLY in the
    # rank-deficient case. The original full-column-rank SMC17 model cases
    # retain precisely their registered least-squares path and receipts.
    if int(torch.linalg.matrix_rank(C,rtol=1e-10))<C.shape[1]:
        t0=torch.linalg.pinv(C,rtol=1e-10)@b
    else:
        t0=torch.linalg.lstsq(C,b[:,None]).solution[:,0]
    s0=U@t0
    floor=float((C@t0-b).norm())
    if not math.isfinite(floor) or floor<0:
        raise ArithmeticError("nonfinite inactive residual floor")
    original_code=code.detach().double()
    selected=baseline.detach().double()
    base_eps=float((B@(delta-baseline)).norm())
    best_eps=base_eps
    choice="baseline"
    candidate_receipts=[]
    h64=h.detach().double()

    def consider(kind,vector,seed=None):
        nonlocal selected,best_eps,choice
        record={"kind":kind,"seed":seed,"valid":False,"inactive_epsilon":None}
        if vector is None or not bool(torch.isfinite(vector).all()):
            candidate_receipts.append(record)
            return
        n=float(vector.norm())
        if n<=1e-8:
            candidate_receipts.append(record)
            return
        sham=(d/n)*vector
        if float((A@sham).abs().max())>ORTH_TOL or abs(float(sham.norm())-d)>ORTH_TOL:
            candidate_receipts.append(record)
            return
        z=encode_one(sae,mean,scale,h64-sham)
        drift=max(abs(float(z[i]-original_code[i]))/(1+float(original_code[i]))
                  for i in active)
        if any(float(z[i])<=0 for i in active) or drift>CODE_TOL:
            candidate_receipts.append(record)
            return
        eps=float((B@(delta-sham)).norm())
        if not math.isfinite(eps):
            candidate_receipts.append(record)
            return
        record.update({"valid":True,"inactive_epsilon":eps,
                       "max_active_scaled_drift":drift})
        candidate_receipts.append(record)
        # Strict smaller result, stable tie-break by candidate order.
        if eps<best_eps-1e-12:
            best_eps=eps
            selected=sham
            choice=kind

    consider("baseline",baseline)
    if float(s0.norm())>1e-8:
        consider("ls_positive",s0)
        consider("ls_negative",-s0)
    else:
        candidate_receipts.append({"kind":"ls_positive","seed":None,
                                   "valid":False,"inactive_epsilon":None})
        candidate_receipts.append({"kind":"ls_negative","seed":None,
                                   "valid":False,"inactive_epsilon":None})
    for attempt in range(random_count):
        tangent,seed=random_tangent(pair,family,puzzle_seed,slot,feature,attempt,A)
        consider(f"random_{attempt:03d}",tangent,seed)
    if len(candidate_receipts)!=131:
        raise AssertionError("incorrect frozen candidate budget")
    if best_eps>base_eps+1e-9 or floor>best_eps+1e-7:
        raise AssertionError("inactive matching worsened or lower bound violated")
    # Check full feature preactivation-difference vectors, preserving all 96.
    diffkill=(cov@delta).tolist()
    diffbase=(cov@baseline).tolist()
    diffbest=(cov@selected).tolist()
    norms=[float(x) for x in (delta.norm(),baseline.norm(),selected.norm())]
    diagnostics={
      "active_ids":active,
      "matched_id":feature,
      "inactive_ids":inactive,
      "physical_doses":norms,
      "full_inactive_row_rank":int(torch.linalg.matrix_rank(B)),
      "inactive_singular_value_min":float(torch.linalg.svdvals(B)[-1]),
      "tangent_dimension":44,
      "inactive_residual_floor":floor,
      "epsilon_baseline":base_eps,
      "epsilon_best":best_eps,
      "epsilon_ratio":best_eps/base_eps if base_eps>1e-12 else None,
      "chosen_candidate":choice,
      "candidate_receipts":candidate_receipts,
      "all96_kill_preactivation_delta":diffkill,
      "all96_baseline_preactivation_delta":diffbase,
      "all96_best_preactivation_delta":diffbest,
      "target_clean_coefficient":float(code[feature]),
      "overshoot":TAU,
    }
    return selected,diagnostics
