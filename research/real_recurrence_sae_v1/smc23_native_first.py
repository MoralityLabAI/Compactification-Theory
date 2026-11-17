"""SMC-23: target-native active SAE erasure on frozen recurrent Sudoku teachers.

Registration is SMC23_PROTOCOL.md, committed before this implementation.
Calibration never generates fresh SMC-23 cohorts. Full test reads an immutable
calibration receipt and never tunes its frozen selection or controls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from smc13_family import (new_full_solution, dig_unique, valid_solution,
                          rectangle_count, count_solutions, fingerprint)
from smc14_causal_transfer import (source_group_context, goal_positions,
                                  ORIGINAL_SOURCE_RECEIPT_SHA256,
                                  NEW_TEACHER_RECEIPT_SHA256,
                                  CONDITIONED_JSON_SHA256)
from smc15_tangent_conditioning import (load_and_lock_inputs,
                                       fresh_rosters as smc15_fresh)

SCHEMA = "SMC-23/target-native-encoder-erasure-20261117-v1"
CAL_SCHEMA = "SMC-23/calibration-only-feature-list-20261117-v1"
COMPONENTS = ("pretrainedA_71","newB_501_SAE_71","newB_501_SAE_89",
              "newB_701_SAE_71","newB_701_SAE_89")
B_COMPONENTS = COMPONENTS[1:]
FAMILIES = {"smc23_novel30":(202633030000,30),
            "smc23_novel34":(202633040000,34)}
OLDER_SMC16 = {"smc16_novel30":(202616030000,30),
               "smc16_novel34":(202616040000,34)}
PUZZLES=64
GOALS=8
TAU=0.02
MAX_DOSE_MULTIPLE=6.0
ORTHO_TOL=1e-8
CODE_TOL=1e-4
SHAM_RANDOM=32
BOOT=2000
MIN_GOALS=60
MIN_PUZZLES=30
BOUNDED_SOURCE_OLD_HASHES=(ORIGINAL_SOURCE_RECEIPT_SHA256,
                           NEW_TEACHER_RECEIPT_SHA256,CONDITIONED_JSON_SHA256)


def sha(x:bytes)->str:
    return hashlib.sha256(x).hexdigest()


def protocol_digest():
    return sha((HERE/"SMC23_PROTOCOL.md").read_bytes())


def seed(*parts):
    x="SMC23|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(x.encode()).digest()[:8],"big") & ((1<<63)-1)


def fp32_code(sae,mean,scale,h):
    with torch.no_grad():
        return sae.encode(((h.detach().float()-mean)/scale).unsqueeze(0))[0].detach().double()


def affine_physical(sae,mean,scale):
    with torch.no_grad():
        cov=sae.encoder.weight.detach().double()/scale.detach().double()[None,:]
        dec=(sae.decoder.weight.detach().double()*scale.detach().double()[:,None]).T
        if cov.shape!=(96,48) or dec.shape!=(96,48):
            raise ValueError("unexpected 48 to 96 physical SAE geometry")
    return cov,dec


def null_project(target,protected):
    if protected.ndim!=2 or protected.shape[1]!=48:
        raise ValueError("bad protected physical matrix")
    _,s,vh=torch.linalg.svd(protected,full_matrices=False)
    if float(s.min())<=1e-10:
        return None
    result=target-vh.T@(vh@target)
    if float(result.norm())<=1e-8 or float((protected@result).abs().max())>ORTHO_TOL:
        return None
    return result


def plan_target_native(h,sae,mean,scale,cov,dec,key):
    """Select top active j from target's own SAE, then freeze all input edits.

    All rejection gates use no task label, correct response or native output.
    """
    with torch.no_grad():
        code=fp32_code(sae,mean,scale,h)
        values,indices=torch.topk(code,5,sorted=True)
        active=indices[:4].tolist()
        target=active[0]
        info={"status":"pending", "feature":target, "active_ids":active,
              "top5_activities":[float(v) for v in values],
              "n_active":int((code>0).sum())}
        def fail(reason, **extras):
            return {"status":reason, **info, **extras} | {"status":reason},None,None
        if (info["n_active"]!=4 or float(values[3])<=0 or
            float(values[0])<=float(values[1])):
            return fail("incomplete_or_tied_topfour")
        w=cov[target]; protected=cov[active[1:]]
        projected=null_project(w,protected)
        if projected is None:
            return fail("target_projection_rank")
        z=float(code[target])
        kill=(z+TAU)*projected/float(projected@projected)
        dose=float(kill.norm())
        native=z*float(dec[target].norm())
        info.update({"dose":dose,"native_decoder_contribution":native,
                     "dose_multiple":dose/native if native>0 else None})
        if native<=1e-12 or dose>MAX_DOSE_MULTIPLE*native:
            return fail("dose_cap")
        if abs(float(w@kill)-(z+TAU))>ORTHO_TOL or float((protected@kill).abs().max())>ORTHO_TOL:
            return fail("encoder_affine_geometry")
        erased=fp32_code(sae,mean,scale,h.double()-kill)
        drift=max(abs(float(erased[i]-code[i]))/(1+float(code[i])) for i in active[1:])
        info.update({"erased_code":float(erased[target]),
                     "protected_max_recode_drift":drift,
                     "erased_support":torch.nonzero(erased>0).flatten().tolist()})
        if (float(erased[target])>1e-7 or drift>CODE_TOL or
            any(float(erased[i])<=0 for i in active[1:])):
            return fail("target_erasure_recode")
        # Dimension 44 tangent to the four active physical covectors.
        A=cov[active]
        _,sv,vh=torch.linalg.svd(A,full_matrices=True)
        if float(sv[-1])<=1e-10:
            return fail("four_covar_rank")
        Q=vh[4:].T.contiguous()
        if Q.shape!=(48,44):
            raise ArithmeticError("incorrect tangent dimension")
        if float((A@Q).abs().max())>ORTHO_TOL:
            raise ArithmeticError("active-covector nullspace not orthogonal")
        inactive=[k for k in range(96) if k not in active]
        B=cov[inactive]
        C=B@Q
        target_drift=B@kill
        ls=torch.linalg.lstsq(C,target_drift).solution
        floor=float((C@ls-target_drift).norm())
        possibilities=[Q@ls]
        for index in range(SHAM_RANDOM):
            g=torch.Generator().manual_seed(seed(key,index))
            sample=torch.randn(48,generator=g,dtype=torch.float64)
            possibilities.append(Q@(Q.T@sample))
        best=None
        best_dist=float("inf")
        eligible_candidates=0
        for idx,candidate in enumerate(possibilities):
            magnitude=float(candidate.norm())
            if magnitude<=1e-10 or not math.isfinite(magnitude):
                continue
            sham=candidate*(dose/magnitude)
            if abs(float(sham.norm())-dose)>ORTHO_TOL or float((A@sham).abs().max())>ORTHO_TOL:
                continue
            recovered=fp32_code(sae,mean,scale,h.double()-sham)
            rel=max(abs(float(recovered[i]-code[i]))/(1+float(code[i])) for i in active)
            if rel>CODE_TOL or any(float(recovered[i])<=0 for i in active):
                continue
            eligible_candidates+=1
            dist=float((B@(kill-sham)).norm())
            if dist<best_dist:
                best=sham
                best_dist=dist
                info["winning_sham_index"]=idx
                info["sham_active_max_recode_drift"]=rel
        info["candidate_count"]=len(possibilities)
        info["admitted_candidates"]=eligible_candidates
        info["inactive_preweighted_mismatch_floor"]=floor
        info["inactive_preweighted_mismatch"]=best_dist if best is not None else None
        info["inactive_covar_rank"]=int(torch.linalg.matrix_rank(B))
        if best is None:
            return fail("no_recode_stable_sham")
        if abs(float(best.norm())-dose)>ORTHO_TOL:
            raise ArithmeticError("matched dose changed")
        info["status"]="eligible"
        return info,kill,best


def four_arm_logits(model,stage,token,kill,sham):
    """Five arms with identical source, output position and native recurrence."""
    with torch.no_grad():
        h=stage.unsqueeze(0).double().repeat(5,1,1)
        h[2,token]-=kill
        h[3,token]-=sham
        h[4,token]+=kill
        for _ in range(3):
            h=model.block(h)
        return model.head(h)[:,token,:]


def outcome_rows(model,sae,mean,scale,cohort,component,family,roster):
    """All feature and sham plans are frozen BEFORE any target digit is read."""
    cov,dec=affine_physical(sae,mean,scale)
    records=[]
    for pi in range(len(roster)):
        stage=cohort["stage1"][pi]
        positions=cohort["positions"][pi]
        plans=[]
        for j,pos in enumerate(positions.tolist()):
            plans.append(plan_target_native(stage[pos],sae,mean,scale,cov,dec,
                                            f"{family}|{pi}|{component}|{j}"))
        clean=cohort["clean_logits"][pi]
        truth=cohort["truth_at_goals"][pi]
        items=[]
        for j,(info,k,s) in enumerate(plans):
            token=int(positions[j])
            row={"goal_index":j,"position":token,
                 "feature":info["feature"],"status":info["status"],
                 "input_plan":info}
            if info["status"]!="eligible":
                items.append(row)
                continue
            logits=four_arm_logits(model,stage,token,k,s)
            if not torch.equal(logits[0],logits[1]):
                raise ArithmeticError("zero-dose self-patch altered the full vocabulary")
            if float((logits[0]-clean[token]).abs().max())>1e-8:
                raise ArithmeticError("per-goal source-clean mismatch")
            digit=int(truth[j])
            if not 1<=digit<=9:
                raise ValueError("Sudoku truth digit outside 1...9")
            rival=clean[token].clone()
            rival[digit]=-float("inf")
            rival_index=int(rival.argmax())
            margins=(logits[:,digit]-logits[:,rival_index]).tolist()
            clean_correct=int(logits[0].argmax())==digit
            flips=[clean_correct and int(logits[a].argmax())!=digit for a in (2,3,4)]
            row.update({
                "truth_digit":digit,"rival_digit":rival_index,
                "clean_correct":bool(clean_correct),
                "margins":{"clean":margins[0],"self":margins[1],
                           "kill":margins[2],"sham":margins[3],
                           "mirror":margins[4]},
                "paired_margin_effect":margins[3]-margins[2],
                "kill_margin_damage":margins[0]-margins[2],
                "sham_margin_damage":margins[0]-margins[3],
                "abs_specificity":abs(margins[0]-margins[2])-abs(margins[0]-margins[3]),
                "kill_clean_correct_to_wrong":flips[0],
                "sham_clean_correct_to_wrong":flips[1],
                "mirror_clean_correct_to_wrong":flips[2],
                "native_output_linf_kill":float((logits[2]-logits[0]).abs().max()),
                "native_output_linf_sham":float((logits[3]-logits[0]).abs().max()),
            })
            items.append(row)
        records.append({"puzzle_index":pi,"grid_seed":roster[pi]["seed"],
                        "goal_positions":positions.tolist(),
                        "goals":items})
    return records


def tally(rows):
    out=Counter()
    for p in rows:
        for g in p["goals"]:
            out[g["status"]]+=1
    return dict(out)


def score(rows,component,family,chosen=None,draws=BOOT):
    """Cluster by independent puzzles; frozen all-active effect or secondary."""
    sums=[];counts=[];puzzle_avgs=[];flipk=0;flips=0;correct=0;abs_effects=[]
    doses=[];sham_mismatch=[]
    for puzzle in rows:
        eligible=[r for r in puzzle["goals"] if r["status"]=="eligible" and
                  (chosen is None or r["feature"] in chosen)]
        diffs=[r["paired_margin_effect"] for r in eligible]
        n=len(diffs)
        sums.append(sum(diffs));counts.append(n)
        puzzle_avgs.append(sum(diffs)/n if n else None)
        flipk+=sum(r["kill_clean_correct_to_wrong"] for r in eligible)
        flips+=sum(r["sham_clean_correct_to_wrong"] for r in eligible)
        correct+=sum(r["clean_correct"] for r in eligible)
        abs_effects.extend(r["abs_specificity"] for r in eligible)
        doses.extend(r["input_plan"]["dose"] for r in eligible)
        sham_mismatch.extend(r["input_plan"]["inactive_preweighted_mismatch"] for r in eligible)
    n_goals=sum(counts);n_puzzles=sum(x>0 for x in counts)
    if n_goals==0:
        return {"status":"no_eligible_goals","eligible_goals":0,
                "eligible_puzzles":0,"registered_pass":False}
    rng=random.Random(seed("bootstrap",20262317,family,component,"all" if chosen is None else "selected"))
    boot=[]
    for _ in range(draws):
        selected=[rng.randrange(len(rows)) for _ in range(len(rows))]
        count=sum(counts[i] for i in selected)
        if count:
            boot.append(sum(sums[i] for i in selected)/count)
    boot.sort()
    n_boot=len(boot)
    ci95=([boot[int(.025*n_boot)],boot[int(.975*n_boot)-1]]
          if n_boot>=1990 else None)
    ci_corr=([boot[max(0,int(.003125*n_boot))],boot[min(n_boot-1,int(.996875*n_boot))]]
             if n_boot>=1990 and chosen is None and component in B_COMPONENTS else None)
    mean=sum(sums)/n_goals
    min_goals=MIN_GOALS if chosen is None else 24
    min_puzzles=MIN_PUZZLES if chosen is None else 20
    enough=n_goals>=min_goals and n_puzzles>=min_puzzles and n_boot>=1990
    gate=bool(chosen is None and component in B_COMPONENTS and enough
              and ci_corr is not None and ci_corr[0]>0.02 and flipk>flips)
    return {
        "status":"measured" if enough else "underpowered_coverage",
        "eligible_goals":n_goals,"eligible_puzzles":n_puzzles,
        "n_puzzles_total":len(rows),"n_without_eligible":len(rows)-n_puzzles,
        "token_weighted_margin_difference":mean,
        "unweighted_per_eligible_puzzle_mean":
            sum(x for x in puzzle_avgs if x is not None)/n_puzzles,
        "unconditional_puzzle_mean_zeros_for_no_eligible":
            sum(x or 0.0 for x in puzzle_avgs)/len(rows),
        "cluster_bootstrap_draws":n_boot,
        "puzzle_cluster_ci95":ci95,
        "bonferroni_99_375_ci":ci_corr,
        "registered_pass":gate,
        "kill_correct_to_wrong":flipk,"sham_correct_to_wrong":flips,
        "n_clean_correct":correct,
        "mean_abs_specificity":sum(abs_effects)/len(abs_effects),
        "mean_physical_dose":sum(doses)/len(doses),
        "mean_inactive_preactivation_mismatch":sum(sham_mismatch)/len(sham_mismatch),
        "eligibility":tally(rows)
    }


def pick_calibration_features(rows):
    values=defaultdict(list)
    for p in rows:
        for g in p["goals"]:
            if g["status"]=="eligible":
                values[g["feature"]].append((p["puzzle_index"],g["paired_margin_effect"]))
    stats=[]
    for f,examples in values.items():
        count=len(examples);cluster=len({p for p,e in examples})
        mean=sum(e for p,e in examples)/count
        stats.append({"feature":f,"calibration_goals":count,
                      "calibration_puzzles":cluster,"mean_effect":mean,
                      "qualified":count>=4 and cluster>=3 and mean>.02})
    qualified=sorted((v for v in stats if v["qualified"]),
                     key=lambda x:(-x["mean_effect"],x["feature"]))[:8]
    return {"frozen_ids":[x["feature"] for x in qualified],
            "qualified_before_cap":sum(x["qualified"] for x in stats),
            "feature_calibration_statistics":sorted(stats,key=lambda x:x["feature"])}


def _previous_smc16_targets():
    """Reconstruct SMC-16 original seeded puzzle and target roster for exclusion."""
    old_targets=set()
    old_inputs=set()
    for base,holes in OLDER_SMC16.values():
        for i in range(PUZZLES):
            solution=new_full_solution(base+i)
            puzzle=dig_unique(solution,base+i+1000000,holes)
            old_targets.add(tuple(solution));old_inputs.add(tuple(puzzle))
    return old_inputs,old_targets


def generate_fresh(old):
    old15,manifest15=smc15_fresh(old)
    seen_y={tuple(r["target"]) for group in (list(old.values())+list(old15.values()))
            for r in group}
    seen_x={tuple(r["x"]) for group in (list(old.values())+list(old15.values()))
            for r in group}
    old16_x,old16_y=_previous_smc16_targets()
    seen_x.update(old16_x);seen_y.update(old16_y)
    n_excluded=len(seen_y)
    out={}
    for family,(base,holes) in FAMILIES.items():
        entries=[]
        for i in range(PUZZLES):
            seed_id=base+i
            solution=new_full_solution(seed_id)
            if not valid_solution(solution) or rectangle_count(solution)<=0:
                raise ValueError("new solution in canonical orbit or invalid")
            puzzle=dig_unique(solution,seed_id+1000000,holes)
            if puzzle.count(0)!=holes or count_solutions(puzzle,2)!=1:
                raise ValueError("new puzzle not source-unique or correct holes")
            if tuple(solution) in seen_y or tuple(puzzle) in seen_x:
                raise ValueError("SMC23 duplicated a prior source solution or puzzle")
            if any(value and value!=solution[j] for j,value in enumerate(puzzle)):
                raise ValueError("clue disagrees with target")
            seen_x.add(tuple(puzzle));seen_y.add(tuple(solution))
            entries.append({"seed":seed_id,"x":puzzle,"target":solution,
                            "holes":holes,"rectangles":rectangle_count(solution)})
        out[family]=entries
    roster_hashes={fam:fingerprint(rows) for fam,rows in out.items()}
    manifest={"schema":"SMC23/source-disjoint-novel-sudoku-v1",
              "frozen_seeds":FAMILIES,"n_excluded_old_targets":n_excluded,
              "n_per_family":PUZZLES,"roster_hashes":roster_hashes,
              "min_rectangle":{name:min(x["rectangles"] for x in rows)
                               for name,rows in out.items()},
              "input_sha256":{name:sha(json.dumps([r["x"] for r in rows],
                         sort_keys=True,separators=(",",":")).encode())
                         for name,rows in out.items()},
              "target_sha256":{name:sha(json.dumps([r["target"] for r in rows],
                         sort_keys=True,separators=(",",":")).encode())
                         for name,rows in out.items()},
              "source_disjoint":True}
    return out,manifest


def load_locked(original,independent,conditioned,fetch_source):
    if sha((Path(original)/"joint-full.json").read_bytes())!=ORIGINAL_SOURCE_RECEIPT_SHA256:
        raise ValueError("original SMC11C source receipt differs")
    if sha((Path(independent)/"full.json").read_bytes())!=NEW_TEACHER_RECEIPT_SHA256:
        raise ValueError("independent teacher original receipt differs")
    if sha(Path(conditioned).read_bytes())!=CONDITIONED_JSON_SHA256:
        raise ValueError("SMC13C correspondence receipt differs")
    models,components,rosters,frozen=load_and_lock_inputs(
        original,independent,conditioned,fetch_source)
    if len(frozen)!=4 or any(not x["hashes"]["matched_to_original_smc13c_receipt"]
                             for x in frozen.values()):
        raise ValueError("source calibration correspondences have changed")
    return models,components,rosters,frozen


def run(stage,original,independent,conditioned,calibration=None,fetch_source=False):
    if stage not in ("calibration","test"):
        raise ValueError("unknown registered stage")
    torch.set_num_threads(2)
    models,components,prior,frozen=load_locked(original,independent,conditioned,fetch_source)
    freeze_sha={str(i):x["hashes"] for i,x in frozen.items()}
    if stage=="calibration":
        rosters={"feature_calibration":prior["feature_calibration"]}
        keys=("feature_calibration",)
        original_calibration=None
    else:
        if calibration is None:
            raise ValueError("stage test requires earlier calibration-only artifact")
        original_calibration=json.loads(Path(calibration).read_bytes())
        if (original_calibration["schema"]!=CAL_SCHEMA or
            original_calibration["protocol_sha256"]!=protocol_digest() or
            original_calibration["calibration_mapping_hashes"]!=freeze_sha or
            tuple(original_calibration["components"])!=COMPONENTS):
            raise ValueError("frozen calibration source, protocol or map differs")
        rosters,manifest=generate_fresh(prior)
        keys=tuple(FAMILIES)
    ctx,_=source_group_context(models,components,rosters,keys)
    results={}
    for family in keys:
        comps={}
        for component in COMPONENTS:
            teacher_id,sae,mean,scale=components[component]
            cohort=ctx[teacher_id]["cohorts"][family]
            rows=outcome_rows(ctx[teacher_id]["model64"],sae,mean,scale,
                              cohort,component,family,rosters[family])
            primary=score(rows,component,family)
            comp={"summary":primary,"per_puzzle":rows}
            if stage=="calibration":
                comp["frozen_feature_ranking"]=pick_calibration_features(rows)
            else:
                chosen=set(original_calibration["data"]["feature_calibration"]
                           [component]["frozen_feature_ranking"]["frozen_ids"])
                comp["calibration_beneficial_secondary"]=score(
                    rows,component,family,chosen=chosen)
            comps[component]=comp
            print("SMC23",stage,family,component,
                  {"eligible":primary["eligible_goals"],
                   "mean":primary.get("token_weighted_margin_difference"),
                   "status":primary["status"],
                   "registered_pass":primary["registered_pass"]},flush=True)
        results[family]=comps
    report={"schema":CAL_SCHEMA if stage=="calibration" else SCHEMA,
            "stage":stage,"protocol_sha256":protocol_digest(),
            "source_artifacts":BOUNDED_SOURCE_OLD_HASHES,
            "calibration_mapping_hashes":freeze_sha,
            "components":COMPONENTS,"torch":torch.__version__,
            "runtime":"CPU eager, source FP32 stage1 + FP64 three native blocks",
            "full_source_artifacts_checked":True,
            "data":results}
    if stage=="calibration":
        report["source_calibration_roster_sha256"]=fingerprint(
            rosters["feature_calibration"])
    else:
        report["calibration_receipt_sha256"]=sha(Path(calibration).read_bytes())
        report["fresh_roster_manifest"]=manifest
        report["fresh_rosters"]=rosters
        decisions={component:all(results[f][component]["summary"]["registered_pass"]
                                  for f in FAMILIES) for component in B_COMPONENTS}
        report["pair_success"]=decisions
        report["program_success"]=any(decisions[n] for n in B_COMPONENTS[:2]) and any(
            decisions[n] for n in B_COMPONENTS[2:])
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",required=True,choices=("calibration","test"))
    p.add_argument("--original-artifact",type=Path,required=True)
    p.add_argument("--independent-artifact",type=Path,required=True)
    p.add_argument("--conditioned-receipt",type=Path,required=True)
    p.add_argument("--calibration",type=Path)
    p.add_argument("--fetch-source",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run(a.stage,a.original_artifact,a.independent_artifact,
               a.conditioned_receipt,a.calibration,a.fetch_source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("SMC23_COMPLETE",json.dumps({
        "stage":a.stage,"out":str(a.out),
        "per_component":{fam:{name:info["summary"]["status"]
                         for name,info in cells.items()}
                         for fam,cells in result["data"].items()},
        "program_success":result.get("program_success"),
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
