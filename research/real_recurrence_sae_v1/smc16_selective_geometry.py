"""SMC-16 geometry and frozen puzzle-cluster scorer.

Every selection decision uses source SAE codes, receiving SAE codes, frozen
encoder/decoder weights and visible clue/goal positions. Task labels and
intervention logits enter only after a whole eligible mask has been frozen.
"""
import hashlib
import math
import random

import torch

FAMILIES = {
    "smc16_novel_30": (202616030000, 30),
    "smc16_novel_34": (202616040000, 34),
}
TAU = 0.02
DOSE_MULTIPLE = 6.0
PROJECTION_MIN_NORM = 1e-8
ORTHO_ABS_TOL = 1e-8
CODE_REL_TOL = 1e-4
N_GOALS = 8
N_PUZZLES = 64
N_BOOTSTRAPS = 2000


def fresh_rosters(old, limit=N_PUZZLES):
    """Generate new noncanonical source-disjoint unique Sudoku solutions."""
    if limit not in (4, 64):
        raise ValueError("only preregistered pilot or full roster")
    from smc13_family import (
        new_full_solution, dig_unique, valid_solution,
        count_solutions, rectangle_count, fingerprint,
    )
    solutions=set()
    inputs=set()
    for rows in old.values():
        for row in rows:
            target=tuple(row["target"])
            puzzle=tuple(row["x"])
            if target in solutions or puzzle in inputs:
                raise ValueError("overlap in prior source/test cohorts")
            solutions.add(target)
            inputs.add(puzzle)
    rosters={}
    for name,(seed_base,holes) in FAMILIES.items():
        cohort=[]
        for i in range(limit):
            seed=seed_base+i
            target=new_full_solution(seed)
            if not valid_solution(target) or rectangle_count(target)<=0:
                raise ValueError("solution violates noncanonical orbit lock")
            puzzle=dig_unique(target,seed+1000000,holes)
            if puzzle.count(0)!=holes or count_solutions(puzzle,limit=2)!=1:
                raise ValueError("puzzle is not uniquely solvable with fixed holes")
            if tuple(target) in solutions or tuple(puzzle) in inputs:
                raise ValueError("SMC16 shares a solution or input with an earlier cohort")
            solutions.add(tuple(target));inputs.add(tuple(puzzle))
            cohort.append({"seed":seed,"x":puzzle,"target":target,
                           "holes":holes,"rectangles":rectangle_count(target)})
        rosters[name]=cohort
    return rosters,{k:fingerprint(v) for k,v in rosters.items()}


def physical_encoder_rows(sae,scale):
    rows=sae.encoder.weight.detach().double()/scale.detach().double()[None,:]
    if tuple(rows.shape)!=(96,48) or not bool(torch.isfinite(rows).all()):
        raise ValueError("bad physical encoder covectors")
    return rows


def seeded_normal(pair_index,family,grid_seed,slot,attempt):
    material=f"SMC16|{pair_index}|{family}|{grid_seed}|{slot}|{attempt}"
    seed=int.from_bytes(hashlib.sha256(material.encode()).digest()[:8],"big") & ((1<<63)-1)
    return torch.randn(48,dtype=torch.float64,generator=torch.Generator().manual_seed(seed)),seed


def reencode(sae, h64, mean, scale):
    with torch.no_grad():
        h32=h64.float()
        z=sae.encode((h32-mean)/scale)
    if tuple(z.shape)!=(96,) or not bool(torch.isfinite(z).all()):
        raise ValueError("unexpected top4 SAE code")
    return z.detach().double()


def _reject(reason,detail):
    detail["reason"]=reason
    detail["eligible"]=False
    return detail,torch.zeros(48,dtype=torch.float64),torch.zeros(48,dtype=torch.float64)


def selective_candidate(h,sae,mean,scale,code,matched,native_columns,
                        pair_index,family,grid_seed,slot,source_strict=True):
    """Physical minimal-norm target kill and exact active-encoder tangent sham.

    Returns geometry receipt plus two FP64 displacements. Every rejection
    depends on encoder/decoder geometry and reencoding, never on target digits.
    """
    if tuple(h.shape)!=(48,) or tuple(code.shape)!=(96,) or native_columns.shape!=(96,48):
        raise ValueError("incorrect input shape")
    if matched not in range(96):
        raise ValueError("target feature outside SAE")
    d={"eligible":False,"reason":None,"matched_id":int(matched),
       "source_strict":bool(source_strict),"sham_trials":0,
       "tau":TAU,"dose_multiplier":DOSE_MULTIPLE}
    if not source_strict:
        return _reject("source_top_not_strict",d)
    z=code.detach().double()
    active=torch.nonzero(z>0).flatten().tolist()
    d["active_ids"]=[int(i) for i in active]
    d["matched_clean_coefficient"]=float(z[matched])
    if float(z[matched])<=0:
        return _reject("matched_target_inactive",d)
    if len(active)!=4 or matched not in active:
        return _reject("receiving_four_not_strictly_positive",d)
    others=[int(k) for k in active if k!=matched]
    if len(others)!=3:
        raise AssertionError("incorrect protected active rows")
    d["protected_ids"]=others
    d["clean_active_coefficients"]=[float(z[k]) for k in others+[matched]]
    rows=physical_encoder_rows(sae,scale)
    O=rows[others]
    a=rows[matched]
    if int(torch.linalg.matrix_rank(O))!=3:
        return _reject("protected_encoder_rank_deficient",d)
    try:
        v=a-O.T@torch.linalg.solve(O@O.T,O@a)
    except RuntimeError:
        return _reject("protected_gram_singular",d)
    if not bool(torch.isfinite(v).all()):
        return _reject("nonfinite_projection",d)
    norm_v=float(torch.linalg.vector_norm(v))
    if norm_v<=PROJECTION_MIN_NORM:
        return _reject("target_projection_tiny",d)
    ortho=float((O@v).abs().max())
    if ortho>ORTHO_ABS_TOL:
        return _reject("target_projection_not_orthogonal",d)
    q=float(z[matched])+TAU
    delta_kill=(q/float(v@v))*v
    kill_norm=float(delta_kill.norm())
    native_norm=float(z[matched]*native_columns[matched].norm())
    d.update({"target_gram_projection_norm":norm_v,
              "native_physical_norm":native_norm,
              "kill_physical_norm":kill_norm,
              "norm_cap":DOSE_MULTIPLE*native_norm,
              "projection_orthogonality_residual":ortho})
    if not (math.isfinite(kill_norm) and math.isfinite(native_norm)):
        return _reject("nonfinite_physical_dose",d)
    if native_norm<=1e-12 or kill_norm>DOSE_MULTIPLE*native_norm:
        return _reject("kill_exceeds_registered_native_dose_cap",d)
    h64=h.detach().double()
    zk=reencode(sae,h64-delta_kill,mean,scale)
    d["post_kill_matched_coefficient"]=float(zk[matched])
    d["post_kill_protected_coefficients"]=[float(zk[k]) for k in others]
    kill_drift=[abs(float(zk[k]-z[k]))/(1+float(z[k])) for k in others]
    d["post_kill_max_scaled_protected_drift"]=max(kill_drift)
    raw_kill=sae.encoder(((h64-delta_kill).float()-mean)/scale)
    d["post_kill_raw_target_preactivation"]=float(raw_kill[matched])
    if float(zk[matched])!=0 or float(raw_kill[matched])>0:
        return _reject("matched_feature_survives_kill",d)
    if any(float(zk[k])<=0 for k in others) or max(kill_drift)>CODE_REL_TOL:
        return _reject("protected_features_change_under_kill",d)
    A=rows[others+[matched]]
    if int(torch.linalg.matrix_rank(A))!=4:
        return _reject("four_active_encoder_rank_deficient",d)
    selected_sham=None
    selected_seed=None
    zs=None
    sham_residual=None
    for attempt in range(4):
        g,seed=seeded_normal(pair_index,family,grid_seed,slot,attempt)
        d["sham_trials"]=attempt+1
        try:
            tangent=g-A.T@torch.linalg.solve(A@A.T,A@g)
        except RuntimeError:
            continue
        n=float(tangent.norm())
        err=float((A@tangent).abs().max())
        if n<=PROJECTION_MIN_NORM or err>ORTHO_ABS_TOL or not math.isfinite(n):
            continue
        candidate=(kill_norm/n)*tangent
        preserved=reencode(sae,h64-candidate,mean,scale)
        if any(float(preserved[k])<=0 for k in others+[matched]):
            continue
        drift=max(abs(float(preserved[k]-z[k]))/(1+float(z[k]))
                  for k in others+[matched])
        if drift>CODE_REL_TOL:
            continue
        # Four positive selected coefficients imply all four retain top-four.
        if len(torch.nonzero(preserved>0))!=4:
            continue
        if any(float(preserved[k])<=0 for k in others+[matched]):
            continue
        selected_sham=candidate
        selected_seed=seed
        zs=preserved
        sham_residual=float((A@candidate).abs().max())
        d["post_sham_max_scaled_active_drift"]=drift
        break
    if selected_sham is None:
        return _reject("no_valid_tangent_sham_in_four_fixed_draws",d)
    dose_ratio=kill_norm/float(selected_sham.norm())
    if abs(dose_ratio-1)>ORTHO_ABS_TOL:
        raise AssertionError("norm-matched sham violated")
    d.update({"eligible":True,"reason":"eligible",
        "sham_seed":selected_seed,
        "encoder_active_rows":A.tolist(),
        "delta_kill":delta_kill.tolist(),
        "delta_sham":selected_sham.tolist(),
        "target_q":q,
        "sham_physical_norm":float(selected_sham.norm()),
        "dose_ratio":dose_ratio,
        "kill_protected_orthogonality_residual":float((O@delta_kill).abs().max()),
        "kill_target_preactivation_change":float(a@delta_kill),
        "sham_active_orthogonality_residual":sham_residual,
        "post_sham_active_coefficients":[float(zs[k]) for k in others+[matched]],
    })
    return d,delta_kill,selected_sham


def bootstrap_seed(pair_index,family):
    return 202616130+100*pair_index+len(family)


def score_rows(rows, seed, draws=N_BOOTSTRAPS):
    """Preregistered token-weighted paired bootstrap resampling puzzle IDs."""
    if draws!=N_BOOTSTRAPS or len(rows) not in (4,64):
        raise ValueError("invalid fixed bootstrap design")
    per=[]
    eligible=0
    kill_flips=sham_flips=clean_correct=0
    for row in rows:
        go=[g for g in row["goals"] if g["eligible"]]
        ds=[float(g["kill_drop"])-float(g["sham_drop"]) for g in go]
        per.append(ds)
        eligible+=len(go)
        kill_flips+=sum(bool(g["kill_flip"]) for g in go)
        sham_flips+=sum(bool(g["sham_flip"]) for g in go)
        clean_correct+=sum(bool(g["clean_correct"]) for g in go)
    n_puzzles=sum(bool(g) for g in per)
    all_d=[x for p in per for x in p]
    weighted=sum(all_d)/eligible if eligible else None
    unweighted=(sum(sum(x)/len(x) for x in per if x)/n_puzzles) if n_puzzles else None
    rng=random.Random(seed)
    sample=[]
    for _ in range(draws):
        picks=[rng.randrange(len(rows)) for _ in range(len(rows))]
        pooled=[x for p in picks for x in per[p]]
        if pooled:
            sample.append(sum(pooled)/len(pooled))
    sample.sort()
    bounds=[sample[int(.025*draws)],sample[int(.975*draws)-1]] if len(sample)==draws else None
    good=(len(rows)==64 and eligible>=24 and n_puzzles>=20
          and len(sample)==draws and bounds is not None and bounds[0]>0
          and kill_flips>sham_flips)
    return {
      "eligible_goals":eligible,"eligible_puzzles":n_puzzles,
      "selected_goals":len(rows)*N_GOALS,
      "mean_kill_minus_sham_margin_loss":weighted,
      "unweighted_mean_puzzle_difference":unweighted,
      "puzzle_bootstrap95":bounds,
      "valid_bootstrap_resamples":len(sample),
      "requested_bootstrap_resamples":draws,
      "clean_correct_eligible_goals":clean_correct,
      "kill_clean_correct_to_wrong":kill_flips,
      "sham_clean_correct_to_wrong":sham_flips,
      "kill_flip_rate_on_all_eligible":kill_flips/eligible if eligible else None,
      "sham_flip_rate_on_all_eligible":sham_flips/eligible if eligible else None,
      "registered_success":good,
    }
