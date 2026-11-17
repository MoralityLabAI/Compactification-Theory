"""SMC-23 standard-library independent saved-receipt audit.

Checks frozen chronology/provenance, all recorded eligibility, physical dose
controls, per-puzzle denominators, calibration feature ranking and full
cluster-bootstrap gates. Does not regenerate/checkpoint-run native Torch models.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


COMPONENTS=("pretrainedA_71","newB_501_SAE_71","newB_501_SAE_89",
            "newB_701_SAE_71","newB_701_SAE_89")
B_COMPONENTS=COMPONENTS[1:]
FAMILIES={"smc23_novel30":(202633030000,30),
          "smc23_novel34":(202633040000,34)}
EXPECTED_SOURCE=[
"7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087",
"1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3",
"3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07",
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stable_seed(*args):
    mat="SMC23|"+"|".join(map(str,args))
    return int.from_bytes(hashlib.sha256(mat.encode()).digest()[:8],"big") & ((1<<63)-1)


def fingerprint(rows):
    return digest(json.dumps(rows,sort_keys=True,separators=(",",":")).encode())


def check_sudoku(row,holes,seed):
    x=row["x"];y=row["target"]
    if row["seed"]!=seed or len(x)!=81 or len(y)!=81 or x.count(0)!=holes:
        raise ValueError("wrong frozen puzzle source shape/seed/holes")
    full=set(range(1,10))
    for r in range(9):
        if set(y[9*r:9*r+9])!=full or {y[9*s+r] for s in range(9)}!=full:
            raise ValueError("saved full solution invalid")
    for rr in (0,3,6):
        for cc in (0,3,6):
            if {y[9*(rr+a)+cc+b] for a in range(3) for b in range(3)}!=full:
                raise ValueError("invalid Sudoku box")
    if any(a and a!=b for a,b in zip(x,y)):
        raise ValueError("saved clue disagrees with solution")
    if row["rectangles"]<=0:
        raise ValueError("saved source solution lacks noncanonical invariant")


def check_inputs(cal,full,raw_cal):
    if cal.get("stage")!="calibration" or full.get("stage")!="test":
        raise ValueError("calibration and fresh test have wrong stage")
    if cal["schema"]!="SMC-23/calibration-only-feature-list-20261117-v1":
        raise ValueError("wrong calibration schema")
    if full["schema"]!="SMC-23/target-native-encoder-erasure-20261117-v1":
        raise ValueError("wrong fresh-test schema")
    protocol_digest=digest((Path(__file__).parent/"SMC23_PROTOCOL.md").read_bytes())
    if cal["protocol_sha256"]!=protocol_digest or full["protocol_sha256"]!=protocol_digest:
        raise ValueError("registered protocol changed")
    if full["calibration_receipt_sha256"]!=digest(raw_cal):
        raise ValueError("test result was not linked to frozen calibration artifact")
    if cal["calibration_mapping_hashes"]!=full["calibration_mapping_hashes"]:
        raise ValueError("source conditional mapping changed between stages")
    if list(cal["source_artifacts"])!=EXPECTED_SOURCE or list(full["source_artifacts"])!=EXPECTED_SOURCE:
        raise ValueError("source teacher/SAE artifact fingerprints differ")
    if tuple(cal["components"])!=COMPONENTS or tuple(full["components"])!=COMPONENTS:
        raise ValueError("teacher/SAE component roster changed")
    if set(cal["data"])!={"feature_calibration"}:
        raise ValueError("calibration access exceeds preexisting registered cohort")
    if set(full["data"])!=set(FAMILIES):
        raise ValueError("test cohort differs from frozen novel families")
    for stage in (cal,full):
        for index in range(4):
            x=stage["calibration_mapping_hashes"][str(index)]
            for label,prefix in (("matched_assignment_sha256","matched"),
                                 ("clue_shuffled_assignment_sha256","clue_shuffled")):
                if len(x[label])!=64:
                    raise ValueError("malformed historical source map SHA")
            if x["matched_to_original_smc13c_receipt"] is not True:
                raise ValueError("SMC13C source map failed calibration check")
    all_targets=set();all_inputs=set()
    roster=full["fresh_rosters"]
    manifest=full["fresh_roster_manifest"]
    if manifest["schema"]!="SMC23/source-disjoint-novel-sudoku-v1" or not manifest["source_disjoint"]:
        raise ValueError("missing frozen source-disjoint manifest")
    if manifest["n_per_family"]!=64 or manifest["n_excluded_old_targets"]<512:
        raise ValueError("missing prior source exclusion")
    for family,(base,holes) in FAMILIES.items():
        rows=roster[family]
        if len(rows)!=64:
            raise ValueError("wrong independent test-puzzle count")
        if manifest["roster_hashes"][family]!=fingerprint(rows):
            raise ValueError("test roster SHA256 mismatches actual archived raw puzzles")
        for i,row in enumerate(rows):
            check_sudoku(row,holes,base+i)
            y=tuple(row["target"]);x=tuple(row["x"])
            if x in all_inputs or y in all_targets:
                raise ValueError("source target/input duplicated across test families")
            all_targets.add(y);all_inputs.add(x)
    return protocol_digest


def check_goals(rows,roster,component,family):
    """A calibration receipt has no source roster bytes; validate slot shape only."""
    if len(rows)!=64 or {r["puzzle_index"] for r in rows}!=set(range(64)):
        raise ValueError("missing/duplicate independent test puzzles")
    feature_ids=defaultdict(list)
    counter=Counter()
    for row in rows:
        i=row["puzzle_index"]
        if len(row["goals"])!=8 or len(row["goal_positions"])!=8:
            raise ValueError("wrong frozen goal selection/count")
        positions=row["goal_positions"]
        if len(set(positions))!=8 or not all(0<=p<81 for p in positions):
            raise ValueError("duplicate or invalid goal token")
        if roster is not None:
            if row["grid_seed"]!=roster[i]["seed"]:
                raise ValueError("frozen case seed differs from archived roster")
            blanks=[t for t,x in enumerate(roster[i]["x"]) if x==0]
            expected=[blanks[(j*len(blanks))//8] for j in range(8)]
            if positions!=expected:
                raise ValueError("selected goal is not input-only SMC14 blank rule")
        elif row["grid_seed"]!=202613020000+i:
            raise ValueError("calibration case seed differs from original SMC13 registration")
        for j,g in enumerate(row["goals"]):
            if g["goal_index"]!=j or g["position"]!=positions[j]:
                raise ValueError("incorrect input-only goal position")
            p=g["input_plan"]
            if p["status"]!=g["status"] or p["feature"]!=g["feature"]:
                raise ValueError("incompatible model-input plan/score")
            counter[g["status"]]+=1
            if g["status"]!="eligible":
                if "paired_margin_effect" in g:
                    raise ValueError("an inadmissible goal was scored")
                continue
            if p["n_active"]!=4 or len(p["active_ids"])!=4 or p["active_ids"][0]!=g["feature"]:
                raise ValueError("scored feature was not target-native active highest")
            if p["top5_activities"][0]<=p["top5_activities"][1] or p["top5_activities"][3]<=0:
                raise ValueError("no unique strongest positive feature")
            if p["candidate_count"]!=33 or p["admitted_candidates"]<1:
                raise ValueError("sham search size/eligibility changed")
            if p["erased_code"]>1e-7 or p["protected_max_recode_drift"]>1e-4:
                raise ValueError("SAE selective target-erasure reencoding failed")
            if p["sham_active_max_recode_drift"]>1e-4:
                raise ValueError("matched sham lost original top-four code")
            if p["dose_multiple"]>6.0+1e-12:
                raise ValueError("native decoder-dose cap violated")
            if p["inactive_covar_rank"]<0 or p["inactive_covar_rank"]>48:
                raise ValueError("bad inactive encoder rank")
            if p["inactive_preweighted_mismatch"]+1e-8<p["inactive_preweighted_mismatch_floor"]:
                raise ValueError("sham beats unconstrained residual floor")
            m=g["margins"]
            if m["clean"]!=m["self"]:
                raise ValueError("zero-dose self-patch drifted original logits")
            delta=m["sham"]-m["kill"]
            if abs(g["paired_margin_effect"]-delta)>1e-10:
                raise ValueError("paired native effect cannot be reconstructed")
            if abs((g["kill_margin_damage"]-g["sham_margin_damage"])-delta)>1e-10:
                raise ValueError("margins disagree with recorded native damage")
            if roster is not None and g["truth_digit"]!=roster[i]["target"][positions[j]]:
                raise ValueError("task digit not true Sudoku target")
            if not 1<=g["truth_digit"]<=9:
                raise ValueError("test digit invalid")
            if not math.isfinite(delta):
                raise ValueError("nonfinite task effect")
            if g["kill_clean_correct_to_wrong"] and not g["clean_correct"]:
                raise ValueError("clean-incorrect item counted as flipped")
            feature_ids[g["feature"]].append((i,delta))
    if sum(counter.values())!=64*8:
        raise ValueError("missing one or more registered goal cases")
    return feature_ids,counter


def independent_score(rows,component,family,chosen=None,draws=2000):
    amounts=[];counts=[];damagek=0;damages=0
    for row in rows:
        eligible=[g for g in row["goals"] if g["status"]=="eligible" and
                  (chosen is None or g["feature"] in chosen)]
        amounts.append(sum(g["paired_margin_effect"] for g in eligible))
        counts.append(len(eligible))
        damagek+=sum(g["kill_clean_correct_to_wrong"] for g in eligible)
        damages+=sum(g["sham_clean_correct_to_wrong"] for g in eligible)
    n=sum(counts);clusters=sum(x>0 for x in counts)
    if n==0:
        return {"eligible_goals":0,"eligible_puzzles":0,"registered_pass":False}
    rnd=random.Random(stable_seed("bootstrap",20262317,family,component,
                                "all" if chosen is None else "selected"))
    sampled=[]
    for _ in range(draws):
        index=[rnd.randrange(len(rows)) for _ in range(len(rows))]
        size=sum(counts[i] for i in index)
        if size:
            sampled.append(sum(amounts[i] for i in index)/size)
    sampled.sort()
    length=len(sampled)
    ci95=[sampled[int(.025*length)],sampled[int(.975*length)-1]] if length>=1990 else None
    corrected=([sampled[int(.003125*length)],sampled[min(length-1,int(.996875*length))]]
               if length>=1990 and chosen is None and component in B_COMPONENTS else None)
    informative=(n >= (60 if chosen is None else 24) and
                 clusters >= (30 if chosen is None else 20) and length>=1990)
    pass_gate=bool(chosen is None and component in B_COMPONENTS and informative and
                   corrected is not None and corrected[0]>.02 and damagek>damages)
    return {"eligible_goals":n,"eligible_puzzles":clusters,
            "token_weighted_margin_difference":sum(amounts)/n,
            "cluster_bootstrap_draws":length,
            "puzzle_cluster_ci95":ci95,
            "bonferroni_99_375_ci":corrected,
            "kill_correct_to_wrong":damagek,
            "sham_correct_to_wrong":damages,
            "registered_pass":pass_gate,
            "status":"measured" if informative else "underpowered_coverage"}


def _approx(a,b,name):
    if a is None or b is None:
        if a is not None or b is not None:
            raise ValueError("mismatched null in "+name)
    elif abs(a-b)>1e-9:
        raise ValueError("score mismatch in "+name+f": {a} vs {b}")


def compare_score(recomputed,recorded):
    for key in ("eligible_goals","eligible_puzzles","registered_pass","status"):
        if recomputed[key]!=recorded[key]:
            raise ValueError("independent decision/count mismatch: "+key)
    if recomputed["eligible_goals"]==0:
        return
    for key in ("cluster_bootstrap_draws","kill_correct_to_wrong","sham_correct_to_wrong"):
        if recomputed[key]!=recorded[key]:
            raise ValueError("independent decision/count mismatch: "+key)
    for key in ("token_weighted_margin_difference",):
        _approx(recomputed[key],recorded[key],key)
    for key in ("puzzle_cluster_ci95","bonferroni_99_375_ci"):
        a,b=recomputed[key],recorded[key]
        if a is None or b is None:
            if a is not b:
                raise ValueError("different bootstrap interval eligibility")
        else:
            for aa,bb in zip(a,b):
                _approx(aa,bb,key)


def audit(cal,full,raw_cal):
    prot=check_inputs(cal,full,raw_cal)
    calib={}
    for comp in COMPONENTS:
        d=cal["data"]["feature_calibration"][comp]
        case_feature,_=check_goals(d["per_puzzle"],None,comp,"feature_calibration")
        # Calibration source case-score validation uses source frozen feature
        # lists separately because source calibration roster is absent here.
        values=[]
        for feature,entries in case_feature.items():
            avg=sum(v for _,v in entries)/len(entries)
            qualified=(len(entries)>=4 and len({i for i,_ in entries})>=3 and avg>.02)
            values.append({"feature":feature,"mean_effect":avg,"qualified":qualified})
        qualified=sorted((x for x in values if x["qualified"]),
                         key=lambda x:(-x["mean_effect"],x["feature"]))[:8]
        expected=[e["feature"] for e in qualified]
        actual=d["frozen_feature_ranking"]["frozen_ids"]
        if actual!=expected:
            raise ValueError("calibration-beneficial feature IDs changed")
        calib[comp]=set(actual)
    summaries=[]
    for family in FAMILIES:
        for component in COMPONENTS:
            record=full["data"][family][component]
            rows=record["per_puzzle"]
            ids,counts=check_goals(rows,full["fresh_rosters"][family],component,family)
            recomputed=independent_score(rows,component,family)
            compare_score(recomputed,record["summary"])
            sub=independent_score(rows,component,family,calib[component])
            compare_score(sub,record["calibration_beneficial_secondary"])
            summaries.append({"family":family,"component":component,
                              "eligible_goals":recomputed["eligible_goals"],
                              "eligible_puzzles":recomputed["eligible_puzzles"],
                              "mean_D":recomputed.get("token_weighted_margin_difference"),
                              "ci95":recomputed.get("puzzle_cluster_ci95"),
                              "bonferroni":recomputed.get("bonferroni_99_375_ci"),
                              "kill_flips":recomputed.get("kill_correct_to_wrong"),
                              "sham_flips":recomputed.get("sham_correct_to_wrong"),
                              "registered_pass":recomputed["registered_pass"],
                              "status":record["summary"]["status"],
                              "eligibility":dict(counts)})
    decisions={comp:all(next(x for x in summaries
                             if x["family"]==family and x["component"]==comp
                             )["registered_pass"] for family in FAMILIES)
               for comp in B_COMPONENTS}
    if decisions!=full["pair_success"]:
        raise ValueError("recorded both-family decision differs")
    success=any(decisions[x] for x in B_COMPONENTS[:2]) and any(
        decisions[x] for x in B_COMPONENTS[2:])
    if success!=full["program_success"]:
        raise ValueError("recorded teacher-level success differs")
    return {"schema":"SMC-23/independent-stdlib-receipt-audit-v1",
            "audit_status":"PASS","scientific_program_success":success,
            "source_protocol_sha256":prot,
            "calibration_sha256":digest(raw_cal),
            "fresh_manifest_sha256":digest(json.dumps(full["fresh_roster_manifest"],
                                         sort_keys=True).encode()),
            "teacher_pair_success":decisions,
            "new_test_puzzles":128,
            "total_preselected_goal_events":128*8*5,
            "results":summaries,
            "limitations":"Model logits, original top-four tensors and SMC13/15/16 source-target overlap require independent source-weight reruns; this auditor checks archived numeric consistency and input manifests, not hidden model executions."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    raw=a.calibration.read_bytes()
    report=audit(json.loads(raw),json.loads(a.full.read_bytes()),raw)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC23_INDEPENDENT_AUDIT_PASS",json.dumps({
        "program_success":report["scientific_program_success"],
        "new_puzzles":report["new_test_puzzles"],
        "goal_events":report["total_preselected_goal_events"],
        "out":str(a.out)},sort_keys=True))


if __name__=="__main__":
    main()
