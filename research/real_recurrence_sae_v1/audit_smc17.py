"""SMC-17 independent stdlib replay of inactive-matched sham receipts.

Reconstructs source assignment hashes, heldout Sudoku/goal masks, 96-row
preactivation disturbance measures and registered clustered success gates.
It deliberately does not import the Torch experiment or replay the networks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

from smc13_family import valid_solution,rectangle_count,count_solutions
from audit_smc16 import independent_score

FAMILIES={"smc17_novel_30":(202617030000,30),
          "smc17_novel_34":(202617040000,34)}
PAIRS=(("pretrainedA_71","newB_501_SAE_71"),
       ("pretrainedA_71","newB_501_SAE_89"),
       ("pretrainedA_71","newB_701_SAE_71"),
       ("pretrainedA_71","newB_701_SAE_89"))
SOURCE_SHA="7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087"
NEW_SHA="1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3"
MAPPING_SHA="3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07"
PRECOMMIT="59bad2ad422faaf3b7bfe7410f7b48656cb279f0"


def close(a,b,tol=4e-8):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol*max(1,abs(a),abs(b))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def norm(xs):
    return math.sqrt(sum(x*x for x in xs))


def compare_stats(recomputed,reported,what):
    if set(recomputed)!=set(reported):
        raise ValueError("score keys differ: "+what)
    for key,x in recomputed.items():
        y=reported[key]
        if isinstance(x,list):
            if not isinstance(y,list) or len(x)!=len(y) or any(not close(a,b) for a,b in zip(x,y)):
                raise ValueError("bootstrap confidence bounds changed: "+what+"/"+key)
        elif isinstance(x,float) or x is None:
            if not close(x,y):
                raise ValueError("registered mean/proportion changed: "+what+"/"+key)
        elif x!=y:
            raise ValueError("registered count/gate changed: "+what+"/"+key)


def verify_rosters(r):
    count=64 if r["full_protocol"] else 4
    inputs=set();solutions=set()
    for fam,(base,holes) in FAMILIES.items():
        rows=r["fresh_rosters"][fam]
        if (len(rows)!=count or r["puzzles_per_family"]!=count
            or r["test_seed_bases"][fam]!=base):
            raise ValueError("nonregistered fresh family size or base")
        data=json.dumps(rows,sort_keys=True,separators=(",",":")).encode()
        if digest(data)!=r["fresh_manifest_sha256"][fam]:
            raise ValueError("fresh full-puzzle manifest SHA changed")
        for i,case in enumerate(rows):
            if case["seed"]!=base+i or case["holes"]!=holes:
                raise ValueError("unexpected Sudoku seed or hole count")
            x,y=case["x"],case["target"]
            if (len(x)!=81 or len(y)!=81 or x.count(0)!=holes or
                not valid_solution(y) or not count_solutions(x,2)==1):
                raise ValueError("Sudoku uniqueness/validity changed")
            if any(d!=0 and d!=y[j] for j,d in enumerate(x)):
                raise ValueError("wrong visible clues")
            if case["rectangles"]!=rectangle_count(y) or case["rectangles"]<=0:
                raise ValueError("canonical or corrupted completed grid")
            inp,sol=tuple(x),tuple(y)
            if inp in inputs or sol in solutions:
                raise ValueError("duplicate new puzzle")
            inputs.add(inp);solutions.add(sol)


def verify_frozen_assignments(r,calibration):
    items=r["all_seven_original_assignment_receipts"]
    if len(items)!=7 or r["seven_calibration_assignments_frozen_before_heldout"] is not True:
        raise ValueError("missing seven calibration source locks")
    for i,v in enumerate(items):
        match=v["matched"]
        if sorted(match)!=list(range(96)):
            raise ValueError("nonbijective 96-feature assignment")
        if digest(json.dumps(match).encode())!=v["hashes"]["matched_assignment_sha256"]:
            raise ValueError("matched SHA no longer reproducible")
        if v["hashes"]["matched_to_original_smc13c_receipt"] is not True:
            raise ValueError("unverified original conditioned mapping")
        if calibration is not None:
            src=calibration["results"][i]
            if (v["source"],v["target"])!=(src["source"],src["target"]):
                raise ValueError("source/target calibration assignment changed")
            p=src["conditional_protocol"]["position_digit"]
            if v["hashes"]["matched_assignment_sha256"]!=p["permutation_sha256"]:
                raise ValueError("original SMC13C matched feature SHA mismatch")
            if v["hashes"]["clue_shuffled_assignment_sha256"]!=p["within_group_shuffled_assignment_sha256"]:
                raise ValueError("original SMC13C within-group shuffled SHA mismatch")


def verify_candidate(info):
    g=info["candidate_geometry"]
    if not isinstance(g,dict) or len(g["candidate_receipts"])!=131:
        raise ValueError("registered 131 sham candidates absent")
    active=g["active_ids"];inactive=g["inactive_ids"]
    if len(active)!=4 or len(inactive)!=92 or set(active)&set(inactive) or sorted(active+inactive)!=list(range(96)):
        raise ValueError("invalid active/inactive split")
    j=g["matched_id"]
    if j not in active or not close(g["overshoot"],.02):
        raise ValueError("matched active identity changed")
    k=g["all96_kill_preactivation_delta"]
    b=g["all96_baseline_preactivation_delta"]
    t=g["all96_best_preactivation_delta"]
    if any(len(v)!=96 or not all(math.isfinite(x) for x in v) for v in (k,b,t)):
        raise ValueError("full 96d preactivation displacement missing/nonfinite")
    if any(abs(b[i])>2e-6 or abs(t[i])>2e-6 for i in active):
        raise ValueError("sham has changed an original active encoder row")
    if any(abs(k[i])>2e-6 for i in active if i!=j):
        raise ValueError("kill has changed one of the other active encoder rows")
    if not close(k[j],g["target_clean_coefficient"]+.02,2e-6):
        raise ValueError("target affine kill overshoot differs")
    eps_b=norm([k[i]-b[i] for i in inactive])
    eps_t=norm([k[i]-t[i] for i in inactive])
    if not close(eps_b,g["epsilon_baseline"],1e-6) or not close(eps_t,g["epsilon_best"],1e-6):
        raise ValueError("inactive discrepancy misreported")
    if g["inactive_residual_floor"]<0 or g["inactive_residual_floor"]>eps_t+1e-6:
        raise ValueError("least-squares floor exceeds achieved residual")
    if eps_t>eps_b+1e-6:
        raise ValueError("best sham worsens original four-seed control")
    doses=g["physical_doses"]
    if len(doses)!=3 or any(not close(doses[0],x,1e-7) for x in doses[1:]):
        raise ValueError("physical doses no longer equal")
    if g["tangent_dimension"]!=44 or not (0<=g["full_inactive_row_rank"]<=48):
        raise ValueError("unregistered tangent or inactive matrix ranks")
    outcomes=g["candidate_receipts"]
    if outcomes[0]["kind"]!="baseline" or not outcomes[0]["valid"]:
        raise ValueError("original SMC16 control excluded from candidate set")
    if outcomes[1]["kind"]!="ls_positive" or outcomes[2]["kind"]!="ls_negative":
        raise ValueError("signed least-squares controls missing")
    for i,candidate in enumerate(outcomes[3:]):
        if candidate["kind"]!=f"random_{i:03d}":
            raise ValueError("random matching candidate order changed")
    ok=[x for x in outcomes if x["valid"]]
    if not any(x["kind"]==g["chosen_candidate"] for x in ok):
        raise ValueError("selected best control not in valid search set")
    if any(x["inactive_epsilon"] is None or x["inactive_epsilon"]<0 for x in ok):
        raise ValueError("valid candidate lacks inactive discrepancy")
    if not close(eps_t,min(x["inactive_epsilon"] for x in ok),1e-6):
        raise ValueError("winner is not least-inactive matched candidate")
    ratio=eps_t/eps_b if eps_b>1e-12 else None
    if not close(g["epsilon_ratio"],ratio,1e-6):
        raise ValueError("inactive residual ratio changed")


def verify_raw(r):
    for pair_index,group in enumerate(r["comparison_roster"]):
        if pair_index>=4 or (group["source"],group["target"])!=PAIRS[pair_index]:
            raise ValueError("pair order differs from SMC17 preregistration")
        frozen=r["all_seven_original_assignment_receipts"][pair_index]
        if (group["frozen_matched_feature_mapping"]!=frozen["matched"]
            or group["source_mapping_sha256"]!=frozen["hashes"]):
            raise ValueError("pairwise matched feature assignment differs")
        assign=frozen["matched"]
        for fam in FAMILIES:
            rows=group["heldout"][fam]["per_puzzle"]
            roster=r["fresh_rosters"][fam]
            if len(rows)!=len(roster):
                raise ValueError("heldout raw puzzle count differs")
            for p,(row,puzzle) in enumerate(zip(rows,roster)):
                if row["puzzle_index"]!=p or row["seed"]!=puzzle["seed"]:
                    raise ValueError("heldout puzzle index/seed changed")
                blank=[i for i,v in enumerate(puzzle["x"]) if v==0]
                positions=[blank[i*len(blank)//8] for i in range(8)]
                if (row["goal_positions"]!=positions or
                    row["goal_truth_digits"]!=[puzzle["target"][i] for i in positions]):
                    raise ValueError("goal selection or Sudoku truth changed")
                for name in ("eligible_mask","edit_diagnostics","source_top_feature_ids",
                             "target_matched_feature_ids","kill_margin_drop",
                             "sham_margin_drop","baseline_sham_margin_drop",
                             "clean_correct","kill_correct_to_wrong",
                             "sham_correct_to_wrong","baseline_sham_correct_to_wrong"):
                    if len(row[name])!=8:
                        raise ValueError("goal-level receipt missing field "+name)
                for j,flag in enumerate(row["eligible_mask"]):
                    src=row["source_top_feature_ids"][j]
                    target=row["target_matched_feature_ids"][j]
                    if not 0<=src<96 or assign[src]!=target:
                        raise ValueError("heldout source/target matched mapping drift")
                    info=row["edit_diagnostics"][j]
                    if bool(flag)!=(info["status"]=="eligible"):
                        raise ValueError("outcome eligibility mask drift")
                    if flag:
                        verify_candidate(info)
                        for key in ("kill_margin_drop","sham_margin_drop","baseline_sham_margin_drop"):
                            if not math.isfinite(row[key][j]):
                                raise ValueError("nonfinite selected task outcome")
                        if any(row[key][j] for key in ("kill_correct_to_wrong",
                                "sham_correct_to_wrong","baseline_sham_correct_to_wrong")) and not row["clean_correct"][j]:
                            raise ValueError("clean-incorrect token falsely counted as flip")
                    else:
                        if info["candidate_geometry"] is not None:
                            raise ValueError("ineligible goal received optimized geometry")
                        if any(row[key][j] is not None for key in (
                                "kill_margin_drop","sham_margin_drop",
                                "baseline_sham_margin_drop","kill_correct_to_wrong",
                                "sham_correct_to_wrong","baseline_sham_correct_to_wrong")):
                            raise ValueError("ineligible goal received measured task outcome")


def replay_stats(raw,seed):
    best=independent_score(raw,seed)
    baseline=[]
    eb=[];eo=[];floor=[];selected=Counter()
    for row in raw:
        c=dict(row)
        c["sham_margin_drop"]=row["baseline_sham_margin_drop"]
        c["sham_correct_to_wrong"]=row["baseline_sham_correct_to_wrong"]
        baseline.append(c)
        for item in row["edit_diagnostics"]:
            if item["status"]=="eligible":
                g=item["candidate_geometry"]
                eb.append(g["epsilon_baseline"]);eo.append(g["epsilon_best"])
                floor.append(g["inactive_residual_floor"])
                selected[g["chosen_candidate"]]+=1
    older=independent_score(baseline,seed)
    total=sum(eb)
    ratio=sum(eo)/total if total>0 else None
    improved=ratio is not None and ratio<=.90
    return {
     "best_sham_causal_score":best,
     "original_four_seed_sham_causal_score":older,
     "inactive_epsilon_baseline_sum":total,
     "inactive_epsilon_best_sum":sum(eo),
     "inactive_matching_ratio":ratio,
     "mean_residual_floor":sum(floor)/len(floor) if floor else None,
     "best_candidate_choice_counts":dict(selected),
     "improved_inactive_disturbance_by_10pct":improved,
     "registered_success":bool(best["registered_success"] and improved),
    }


def audit(r,calibration=None):
    if r["schema"]!="SMC-17/source-locked-inactive-matching-sham-v1":
        raise ValueError("wrong SMC17 source receipt version")
    if r["preregistration_commit"]!=PRECOMMIT or r["candidate_random_budget"]!=128:
        raise ValueError("preregistration or candidate budget modified")
    if any(r[name]!=value for name,value in (
          ("source_receipt_sha256",SOURCE_SHA),
          ("independent_training_receipt_sha256",NEW_SHA),
          ("conditioned_mapping_receipt_sha256",MAPPING_SHA))):
        raise ValueError("source pretrained or conditioned data fingerprint changed")
    full=r["puzzles_per_family"]==64 and len(r["comparison_roster"])==4
    if r["full_protocol"]!=full:
        raise ValueError("full protocol flag inconsistent with data")
    verify_rosters(r)
    verify_frozen_assignments(r,calibration)
    verify_raw(r)
    for i,group in enumerate(r["comparison_roster"]):
        for fam in FAMILIES:
            stats=replay_stats(group["heldout"][fam]["per_puzzle"],
                202617130+100*i+len(fam))
            prior=group["heldout"][fam]["statistics"]
            if set(stats)!=set(prior):
                raise ValueError("SMC17 statistics keys differ")
            for key,val in stats.items():
                if isinstance(val,dict):
                    if key.endswith("causal_score"):
                        compare_stats(val,prior[key],fam+"/"+key)
                    elif val!=prior[key]:
                        raise ValueError("SMC17 geometry choice counts drift")
                elif isinstance(val,float) or val is None:
                    if not close(val,prior[key]):
                        raise ValueError("SMC17 inactive ratio or residual changed: "+key)
                elif val!=prior[key]:
                    raise ValueError("registered SMC17 gate drift: "+key)
        both=bool(full and all(group["heldout"][fam]["statistics"]["registered_success"]
                               for fam in FAMILIES))
        if bool(group["registered_success_both_families"])!=both:
            raise ValueError("two-cohort SMC17 primary result differs")
    return {
      "schema":"SMC-17/independent-stdlib-audit-v1",
      "original_seven_calibration_assignment_hashes_checked":True,
      "new_solution_and_visible_goal_mask_verified":True,
      "all96_inactive_residuals_and_finite_budget_controls_checked":True,
      "paired_puzzle_bootstrap_and_registered_gate_recomputed":True,
      "full_protocol":full,
      "pair_puzzle_family_cases":sum(len(group["heldout"][fam]["per_puzzle"])
          for group in r["comparison_roster"] for fam in FAMILIES),
      "limit":"Numerical receipt replay only; original PyTorch teacher/SAE weights are verified through the source-locked experiment rather than independently rerun here.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--conditioned-receipt",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/smc17/audit.json"))
    args=p.parse_args()
    src=args.conditioned_receipt.read_bytes()
    if digest(src)!=MAPPING_SHA:
        raise ValueError("original SMC13C mapping receipt SHA differs")
    raw=args.receipt.read_bytes()
    report=audit(json.loads(raw),json.loads(src))
    report["raw_source_sha256"]=digest(raw)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
