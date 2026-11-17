"""NOR-41 read-only post-data audit with 16 replicate-block sensitivity.

Consumes the byte-frozen first graded receipt; imports no experiment/source
generator. Reports the registered source/degree/family metrics as originally
emitted and recomputes nested attacker-budget detection and CAL brackets.

No scientific gate or source-selection parameter is modified by this module.
"""
from __future__ import annotations
import argparse,json,math,statistics
from itertools import product
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

REGISTERED="87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f"
AMENDMENT="e5a95cad1ffdcd019f153cd09dd4d48b40848ef9"
POLICIES=("uniform16","original_mix","deterministic_mix","global7_mix")
DIAGNOSTIC="atom_random_h50"
BUDGETS=(256,512,2048,4096)
FAMILIES=("wide_dense_small","wide_regime_blocks")
REPS=tuple(range(84,100))
TCRIT15=2.1314495455593
CONTRASTS=(
    "global7_256_minus_4096_eval",
    "global7_strong_self_minus_deterministic",
    "global7_strong_worst_minus_original",
    "global7_top_minus_atom_random",
    "deterministic_top_minus_atom_random"
)


def load(path):
    if path.suffix==".zip":
        with ZipFile(path) as archive:
            keys=[name for name in archive.namelist()
                  if name.endswith("nor41_full.json")]
            assert len(keys)==1
            return json.loads(archive.read(keys[0]))
    return json.loads(path.read_bytes())


def t_interval(blocks):
    n=len(blocks)
    assert n==16
    m=statistics.mean(blocks)
    se=statistics.stdev(blocks)/math.sqrt(n)
    return [m-TCRIT15*se,m+TCRIT15*se]


def signflip_two_sided(blocks):
    observed=abs(sum(blocks))
    return sum(abs(sum(s*x for s,x in zip(signs,blocks)))>=observed-1e-13
               for signs in product((-1,1),repeat=len(blocks)))/2**len(blocks)


def analyze(grade,with_signflips=True):
    assert grade["schema"]=="nor41.input_ood_nested_cal_transfer.v1"
    assert grade["registration_first_commit"]==REGISTERED
    assert grade["predata_amendment_01_commit"]==AMENDMENT
    recs=grade["source_records"]
    rows=grade["source_level_primary"]
    assert len(recs)==len(rows)==grade["fresh_source_units"]==64
    assert set(r["replicate"] for r in rows)==set(REPS)
    assert len({r["public_sha256"] for r in rows})==64
    assert len({r["public_sha256"] for r in recs})==64
    assert {r["public_sha256"] for r in recs}=={r["public_sha256"] for r in rows}
    assert all(sum(r["replicate"]==rep for r in rows)==4 for rep in REPS)
    assert {r["family"] for r in rows}==set(FAMILIES)
    assert {r["degree"] for r in rows}=={512,1024}
    den=grade["source_records"][0]["crossplay_denominator"]
    assert all(rec["crossplay_denominator"]==den for rec in recs)
    budget_means={}
    for p in POLICIES:
        budget_means[p]={}
        for n in BUDGETS:
            budget_means[p][str(n)]={
                "own_selfplay":statistics.mean(
                    rec["crossplay_weighted_numerators"][f"cal{n}__{p}"][p]/den
                    for rec in recs),
                "worst_four_calibrated_class":statistics.mean(
                    min(rec["crossplay_weighted_numerators"][f"cal{n}__{att}"][p]/den
                        for att in POLICIES) for rec in recs),
                "cal_optimum_bracket_width":statistics.mean(
                    rec["attack_certificates"][str(n)][p]["global_optimality_gap"]/
                    (n*256) for rec in recs),
                "unique_selected_attacks":statistics.mean(
                    rec["attack_certificates"][str(n)][p]["selected_attack_uniques"]
                    for rec in recs)
            }
    distinct=sorted({r["replicate"] for r in rows})
    block_info={}
    for key in CONTRASTS:
        vals=[statistics.mean(row[key] for row in rows
                              if row["replicate"]==rep) for rep in distinct]
        report={
            "mean":statistics.mean(vals),
            "block_means":vals,
            "descriptive_95_t_interval":t_interval(vals),
            "positive_blocks":sum(v>0 for v in vals),
            "positive_sources":sum(row[key]>0 for row in rows),
            "source_mean":statistics.mean(row[key] for row in rows)}
        if with_signflips:
            report["two_sided_exact_signflip_p_exploratory"]=signflip_two_sided(vals)
        block_info[key]=report
    delta_blocks=[
        statistics.mean(row["global7_256_minus_4096_eval"] for row in rows
                        if row["replicate"]==rep and row["degree"]==512)-
        statistics.mean(row["global7_256_minus_4096_eval"] for row in rows
                        if row["replicate"]==rep and row["degree"]==1024)
        for rep in distinct]
    scale={
        "mean":statistics.mean(delta_blocks),
        "block_differences":delta_blocks,
        "descriptive_95_t_interval":t_interval(delta_blocks),
        "positive_blocks":sum(v>0 for v in delta_blocks),
    }
    if with_signflips:
        scale["two_sided_exact_signflip_p_exploratory"]=signflip_two_sided(delta_blocks)
    for key in ("global7_budget_transfer","degree512_minus_degree1024_transfer",
                "ood_top_gain","global7_strong_self_minus_deterministic",
                "global7_strong_worst_minus_original"):
        assert key in grade["source_means"]
    assert math.isclose(scale["mean"],grade["source_means"][
        "degree512_minus_degree1024_transfer"],abs_tol=1e-12)
    assert math.isclose(block_info["global7_256_minus_4096_eval"]["mean"],
                        grade["source_means"]["global7_budget_transfer"],abs_tol=1e-12)
    subgroups={}
    for attr in ("degree","family"):
        subgroups[attr]={}
        for value in sorted({r[attr] for r in rows}):
            subset=[r for r in rows if r[attr]==value]
            subgroups[attr][str(value)]={
                "n":len(subset),
                "gain_by_cal_budget":{
                    str(n):statistics.mean(r["global7_budget_self"][str(n)]
                                          for r in subset) for n in BUDGETS},
                "contrasts":{k:statistics.mean(r[k] for r in subset)
                             for k in CONTRASTS},
                "original_native_mask_cardinality": {
                    str(j):sum(r["original_native_mask_cardinality"]==j
                                for r in subset)
                    for j in sorted({r["original_native_mask_cardinality"]
                                      for r in subset})}
            }
    classes=[f"cal{n}__{p}" for n in BUDGETS for p in POLICIES]+list(
        grade["blind_control_families"])
    assert len(classes)==22
    cross={
        attacker:{p:statistics.mean(
            rec["crossplay_weighted_numerators"][attacker][p]/den
            for rec in recs) for p in POLICIES}
        for attacker in classes
    }
    return {
        "schema":"nor41.posthoc_source_block_analysis.v1",
        "original_registration":REGISTERED,
        "predata_amendment":AMENDMENT,
        "source_units":64,"replicate_blocks":16,
        "calibration_prefix_budgets":list(BUDGETS),
        "original_registered_pass_unchanged":grade["registered_primary_pass"],
        "original_five_gates_unchanged":grade["registered_five_gates"],
        "source_level_means_as_scored":grade["source_means"],
        "source_level_wins_as_scored":grade["source_direction_wins"],
        "ratio_ood_top_gain":grade["retained_ood_top_gain_ratio"],
        "budget_policy_means":budget_means,
        "block_sensitivity":block_info,
        "degree_scale_block_contrast":scale,
        "subgroups":subgroups,
        "crossplay_22x4":cross,
        "limitation":"16 finite replicate blocks; signflip exchangeability and t-block normality are descriptive. Input-family OOD only; program grammar unchanged. No EVAL-based model selection."
    }


def markdown(report):
    lines=[
        "# NOR-41 original first-grade posthoc audit","",
        "The recorded first-run PASS/FAIL remains immutable. All descriptions here "
        "are computed after the first grader from its SHA-locked receipts.","",
        "## Registered source/replicate-block contrasts","",
        "| Contrast | Mean pp | Descriptive 95% block t interval pp | Positive blocks/16 |",
        "| --- | ---: | --- | ---: |"]
    for name,obj in report["block_sensitivity"].items():
        lo,hi=obj["descriptive_95_t_interval"]
        lines.append(f"| \`{name}\` | {100*obj['mean']:+.4f} | "
                     f"[{100*lo:+.4f}, {100*hi:+.4f}] | {obj['positive_blocks']}/16 |")
    s=report["degree_scale_block_contrast"];lo,hi=s["descriptive_95_t_interval"]
    lines.extend(["",f"Degree512-minus-1024 CAL transfer contrast: "
                  f"{100*s['mean']:+.4f}pp; descriptive 95% block interval "
                  f"[{100*lo:+.4f},{100*hi:+.4f}]pp; "
                  f"{s['positive_blocks']}/16 positive blocks.",
                  "","## Four nested attacker budget curves","",
                  "| Auditor | CAL256 self % | CAL512 self % | CAL2048 self % | CAL4096 self % | CAL4096 worst-class % |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for p,v in report["budget_policy_means"].items():
        lines.append(f"| \`{p}\` | "+" | ".join(
            f"{100*v[str(n)]['own_selfplay']:.4f}" for n in BUDGETS)+
            f" | {100*v['4096']['worst_four_calibrated_class']:.4f} |")
    lines.extend(["","## Source input family and length strata","",
                  "| Group | Cases | CAL256→4096 heldout gain pp | TOP exact gain pp |",
                  "| --- | ---: | ---: | ---: |"])
    for attr,groups in report["subgroups"].items():
        for group,vals in groups.items():
            d=vals["contrasts"]
            lines.append(f"| {attr}: {group} | {vals['n']} | "
                         f"{100*d['global7_256_minus_4096_eval']:+.4f} | "
                         f"{100*d['global7_top_minus_atom_random']:+.4f} |")
    lines.extend(["","## Complete calibrated 22×4 crossplay","",
                  "| Attacker | "+" | ".join(POLICIES)+" |",
                  "| --- | "+" | ".join("---:" for _ in POLICIES)+" |"])
    for att,vals in report["crossplay_22x4"].items():
        lines.append(f"| {att} | "+" | ".join(
            f"{100*vals[p]:.4f}" for p in POLICIES)+" |")
    lines.extend(["","All pool16 exact solutions certify minima only in their "
                  "CAL-selected coordinate subsets. Global CAL singleton lower bounds "
                  "and hidden-EVAL test rates are separately reported. An authentic "
                  "target-output witness and truly hidden audit coins remain external "
                  "premises. NOR41 is input-family OOD within fixed RRC25 program grammar."])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--markdown-out",type=Path)
    a=p.parse_args()
    result=analyze(load(a.full))
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    if a.markdown_out:
        a.markdown_out.parent.mkdir(parents=True,exist_ok=True)
        a.markdown_out.write_text(markdown(result))
    print(markdown(result))


if __name__=="__main__":
    main()
