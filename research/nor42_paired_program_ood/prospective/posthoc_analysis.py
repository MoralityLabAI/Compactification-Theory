"""NOR-42 read-only first-grade source-block audit, with no source generation.

Consumes frozen original first graded JSON or ZIP, preserving first science
PASS/FAIL. The unit of analysis is the 64 public source PAIRS nested in
16 independent replicate blocks, not 128 independent world observations.
"""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
from zipfile import ZipFile
from itertools import product

REGISTERED="9bf353b7c76535e581ff026098cd682914a90ef0"
POLICIES=("uniform16","original_mix","deterministic_mix","global7_mix")
DIAG="atom_random_h50"
FAMILIES=("wide_dense_small","wide_regime_blocks")
REPS=tuple(range(100,116))
T15=2.1314495455593


def read_grade(path):
    if path.suffix==".zip":
        with ZipFile(path) as z:
            names=[name for name in z.namelist()
                   if name.endswith("nor42_full.json")]
            assert len(names)==1
            return json.loads(z.read(names[0]))
    return json.loads(path.read_bytes())


def block_describe(rows,key,with_signflip=True):
    byrep=[statistics.mean(r[key] for r in rows if r["replicate"]==rep)
           for rep in REPS]
    assert len(byrep)==16
    mean=statistics.mean(byrep)
    half=T15*statistics.stdev(byrep)/4
    result={
        "mean":mean,
        "replicate_means":byrep,
        "descriptive_t95_interval":[mean-half,mean+half],
        "positive_replicates":sum(v>0 for v in byrep),
        "positive_source_pairs":sum(r[key]>0 for r in rows)
    }
    if with_signflip:
        observed=abs(sum(byrep))
        result["two_sided_signflip_p_exploratory"]=sum(
            abs(sum(sign*v for sign,v in zip(signs,byrep)))>=observed-1e-12
            for signs in product((-1,1),repeat=16))/65536
    return result


def audit(grade,with_signflip=True):
    assert grade["schema"]=="nor42.paired_program_ood_authenticated_witness.v1"
    assert grade["registration_first_commit"]==REGISTERED
    assert grade["fresh_public_source_pairs"]==64
    assert grade["two_world_targets"]==128
    assert grade["excluded_historical_sources"]==388
    assert grade["auditor_policies"]==list(POLICIES)
    assert grade["diagnostic_auditor"]==DIAG
    assert grade["audit_budget"]==16 and grade["heldout_schedules_per_source"]==1024
    rows=grade["source_level_primary"]
    rec=grade["source_records"]
    assert len(rows)==len(rec)==64
    assert {r["replicate"] for r in rows}==set(REPS)
    assert len({r["public_sha256"] for r in rows})==64
    assert len({r["public_sha256"] for r in rec})==64
    assert {r["public_sha256"] for r in rows}=={r["public_sha256"] for r in rec}
    assert {r["family"] for r in rows}==set(FAMILIES)
    assert {r["degree"] for r in rows}=={512,1024}
    assert all(sum(r["replicate"]==rep for r in rows)==4 for rep in REPS)
    assert all(r["hidden_true_world_witnessfree_false_accept"]
               and r["clean_true_world_authentic_accept"]
               and r["paired_public_prediction_accuracy"]==[1,2]
               for r in rows)
    assert all(r["paired_public_demos_equal"] and r["public_predictor_correct_worlds"]==1
               and r["paired_worlds"]==2 for r in rec)
    model={}
    for p in (*POLICIES,DIAG):
        model[p]=statistics.mean(
            row["mixture_detection_numerators"][p]/
            row["mixture_detection_denominator"] for row in rec)
    subgroup={}
    for attr in ("degree","family"):
        subgroup[attr]={}
        for value in sorted({r[attr] for r in rows}):
            group=[r for r in rows if r[attr]==value]
            receipts=[r for r in rec if r[attr]==value]
            subgroup[attr][str(value)]={
                "source_pairs":len(group),
                "mean_hidden_difference_support":statistics.mean(
                    r["difference_support_size"] for r in group),
                "analytic_uniform16_hit_mean":statistics.mean(
                    r["exact_uniform_probability"] for r in group),
                "source_mean_uniform16_calibration_abs_error":statistics.mean(
                    r["absolute_uniform_calibration_error"] for r in group),
                "empirical_witnessed_detection":{
                    p:statistics.mean(
                        r["mixture_detection_numerators"][p]/
                        r["mixture_detection_denominator"] for r in receipts)
                    for p in (*POLICIES,DIAG)
                },
                "global7_minus_uniform16_mean":statistics.mean(
                    r["global7_minus_uniform"] for r in group)
            }
    measurements=(
        "global7_detection","uniform_detection","global7_minus_uniform",
        "global7_minus_exact_floor","absolute_uniform_calibration_error",
        "exact_uniform_probability","difference_support_size"
    )
    blocks={k:block_describe(rows,k,with_signflip) for k in measurements}
    for key,v in grade["source_means"].items():
        if key=="absolute_uniform_calibration_error":
            assert math.isclose(blocks[key]["mean"],v,abs_tol=1e-12)
        elif key=="global7_hidden_detection":
            assert math.isclose(blocks["global7_detection"]["mean"],v,abs_tol=1e-12)
        elif key=="global7_minus_exact_floor":
            assert math.isclose(blocks[key]["mean"],v,abs_tol=1e-12)
    return {
        "schema":"nor42.prospective_original_grade_posthoc.v1",
        "original_registration":REGISTERED,
        "original_science_pass_unchanged":grade["registered_primary_pass"],
        "original_five_gates_unchanged":grade["registered_five_gates"],
        "original_source_means_unchanged":grade["source_means"],
        "original_source_wins_unchanged":grade["source_direction_wins"],
        "independent_pair_units":64,
        "replicate_blocks":16,
        "true_target_worlds":128,
        "witnessfree_hidden_false_acceptance_rate":1.0,
        "public_only_exact_prediction_pair_mean":0.5,
        "absolute_detection_means":model,
        "subgroup_means":subgroup,
        "replicate_block_descriptive":blocks,
        "scope":"Synthetic hidden suffix and hypothetical independently authenticated "
                "future coefficients. No new source/grade produced by this audit."
    }


def markdown(report):
    out=["# NOR-42 first original paired-world posthoc audit","",
         "The first registered science PASS/FAIL and five gates are preserved without "
         "retrospective threshold changes. Each of 64 independent public-source "
         "pairs represents two possible target worlds with the same public evidence.",
         "","## Absolute hidden-world detection under hypothetical authentic output coefficients","",
         "| Auditor | Mean detection % |",
         "| --- | ---: |"]
    for p,v in report["absolute_detection_means"].items():
        out.append(f"| \`{p}\` | {100*v:.4f} |")
    out+=["","## Public pair/source strata","",
          "| Stratum | Pair units | Difference support mean | Exact uniform16 % | Empirical uniform16 % | Global7 % | Global7−uniform16 pp |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for attr,groups in report["subgroup_means"].items():
        for value,rec in groups.items():
            e=rec["empirical_witnessed_detection"]
            out.append(f"| {attr}: {value} | {rec['source_pairs']} | "
                       f"{rec['mean_hidden_difference_support']:.2f} | "
                       f"{100*rec['analytic_uniform16_hit_mean']:.3f} | "
                       f"{100*e['uniform16']:.3f} | {100*e['global7_mix']:.3f} | "
                       f"{100*rec['global7_minus_uniform16_mean']:+.3f} |")
    out+=["","## Descriptive 16-replicate-block uncertainty","",
          "| Paired source contrast | Mean | Descriptive 95% t interval | Positive blocks / 16 |",
          "| --- | ---: | --- | ---: |"]
    for key in ("global7_detection","uniform_detection","global7_minus_uniform",
                "global7_minus_exact_floor","absolute_uniform_calibration_error"):
        d=report["replicate_block_descriptive"][key]
        lo,hi=d["descriptive_t95_interval"]
        out.append(f"| \`{key}\` | {100*d['mean']:+.3f}% | "
                   f"[{100*lo:+.3f}, {100*hi:+.3f}]% | "
                   f"{d['positive_replicates']}/16 |")
    out+=["","Under a witness-free public-model self-consistency check, the "
          "clean proposal is accepted in both worlds even if the hidden truth "
          "differs, so this constructed control has 100% false acceptance on "
          "the hidden world. A hypothetical authentic true-output witness "
          "allows rejection whenever the k16 plan intersects the hidden "
          "difference support. This is a conditional synthetic comparison, "
          "not a deployed authenticated-witness implementation.",
          "","The original RRC25 admitted class is public-model consistency "
          "evidence. Actual truth may be a hidden off-grammar suffix. Exact "
          "uniform16 support hit probabilities and Global7 full-output "
          "uniform contribution are separate from the finite EVAL sample "
          "frequencies and from malicious radius-four attack results."]
    return "\n".join(out)+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",required=True,type=Path)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--markdown-out",type=Path)
    args=p.parse_args()
    result=audit(read_grade(args.full))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True,exist_ok=True)
        args.json_out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True,exist_ok=True)
        args.markdown_out.write_text(markdown(result))
    print(markdown(result))

if __name__=="__main__":
    main()
