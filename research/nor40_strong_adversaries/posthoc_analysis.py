"""NOR-40 post-data descriptive audit; consumes only frozen first graded JSON.

Never imports NOR40 experiment or modifies prospective gates. Reports source-level
and eight-replicate-block statistics, CAL/EVAL generalization, exact feature
conditional performance and honest global-bound gaps. Source identities are
the 48 study units and six family×degree rows share each replicate block.
"""
from __future__ import annotations
import argparse,json,math,statistics
from fractions import Fraction
from itertools import product
from pathlib import Path
from zipfile import ZipFile

POLICIES=("uniform16","original_mix","deterministic_mix","stochastic50_mix",
          "atom_random_h50","global7_mix","global8_mix")
REGISTERED="dbc1adbe0e30b9c0e36421cf07cebcf376d6a1ec"
T_CRIT7=2.3646242510103
PRIMARY=(
    "global7_baseline_minus_strong_selfplay",
    "global7_strong_self_minus_deterministic_strong",
    "global7_strong_worst_minus_original_strong_worst",
    "global7_top_minus_atom_random",
    "deterministic_top_minus_atom_random",
)


def load(path):
    if path.suffix==".zip":
        with ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith("nor40_full.json")]
            assert len(names)==1
            return json.loads(z.read(names[0]))
    return json.loads(path.read_bytes())


def describe(full):
    assert full["schema"]=="nor40.strong_calibration_policyaware.v1"
    assert full["registration_first_commit"]==REGISTERED
    recs=full["source_records"]
    assert len(recs)==48 and full["fresh_source_units"]==48
    assert tuple(full["auditor_policies"])==POLICIES
    assert len({r["public_sha256"] for r in recs})==48
    assert {r["replicate"] for r in recs}==set(range(76,84))
    assert all(sum(r["replicate"]==i for r in recs)==6 for i in range(76,84))
    per_source=[]
    for r in recs:
        cross=r["crossplay_weighted_numerators"]
        denominator=r["crossplay_denominator"]
        top={p:float(Fraction(*r["exact_top_support_detection"][p]))
             for p in POLICIES}
        policies={}
        for policy in POLICIES:
            cert=r["attack_certificates"][policy]
            policies[policy]={
                "baseline_self":cross["baseline__"+policy][policy]/denominator,
                "strong_self":cross[policy][policy]/denominator,
                "strong_worst":min(cross[attack][policy]/denominator
                                   for attack in POLICIES),
                "top_exact":top[policy],
                "cal256_vs_2048_objective_reduction":(
                    cert["baseline_eight_full_cal_sum"]-
                    cert["strong_eight_full_cal_sum"])/(256*2048*8),
                "global_optimism_gap_upper_bound":(
                    cert["global_optimality_gap_upper_minus_lower"])/(256*2048),
                "cal_global_lower_bound":cert["global_singleton_lower_bound"],
                "cal_global_tested_upper_bound":cert["global_best_found_upper_bound"],
                "restricted_pool_exact_value":cert["restricted_exact_value"],
                "pool_enumerated":cert["restricted_sets_enumerated"],
                "unique_strong_sets":cert["strong_unique_count"],
            }
        scores={
            k:r["crossplay_weighted_numerators"][k] for k in
            ("global7_mix","baseline__global7_mix",
             "deterministic_mix","original_mix","uniform16")}
        per_source.append({
            "replicate":r["replicate"],"degree":r["degree"],
            "family":r["family"],"public_sha256":r["public_sha256"],
            "policies":policies,
            "contrasts":{
                "global7_baseline_minus_strong_selfplay":
                    policies["global7_mix"]["baseline_self"]-
                    policies["global7_mix"]["strong_self"],
                "global7_strong_self_minus_deterministic_strong":
                    policies["global7_mix"]["strong_self"]-
                    policies["deterministic_mix"]["strong_self"],
                "global7_strong_worst_minus_original_strong_worst":
                    policies["global7_mix"]["strong_worst"]-
                    policies["original_mix"]["strong_worst"],
                "global7_top_minus_atom_random":
                    top["global7_mix"]-top["atom_random_h50"],
                "deterministic_top_minus_atom_random":
                    top["deterministic_mix"]-top["atom_random_h50"]
            }
        })
    policy_means={
        p:{k:statistics.mean(s["policies"][p][k] for s in per_source)
           for k in ("baseline_self","strong_self","strong_worst","top_exact",
                     "cal256_vs_2048_objective_reduction",
                     "global_optimism_gap_upper_bound","unique_strong_sets")}
        for p in POLICIES}
    means={k:statistics.mean(s["contrasts"][k] for s in per_source) for k in PRIMARY}
    for k in PRIMARY:
        assert math.isclose(means[k],full["source_means"][k],abs_tol=1e-12)
    block_stats={}
    for k in PRIMARY:
        blocks=[statistics.mean(s["contrasts"][k] for s in per_source
                 if s["replicate"]==i) for i in range(76,84)]
        m=statistics.mean(blocks)
        se=statistics.stdev(blocks)/math.sqrt(8)
        block_stats[k]={
            "mean":m,"replicate_block_means":blocks,
            "descriptive_t95":[m-T_CRIT7*se,m+T_CRIT7*se],
            "block_positive":sum(x>0 for x in blocks),
            "source_positive":sum(s["contrasts"][k]>0 for s in per_source),
            "signflip_two_sided_p_under_symmetry":sum(
                abs(sum(sign*v for sign,v in zip(signs,blocks)))>=
                    abs(sum(blocks))-1e-14
                for signs in product((-1,1),repeat=8))/256,
        }
    classes=list(POLICIES)+["baseline__"+p for p in POLICIES]+list(
        full["blind_control_families"])
    assert len(classes)==20
    cross={attack:{
        p:statistics.mean(r["crossplay_weighted_numerators"][attack][p]/
            r["crossplay_denominator"] for r in recs) for p in POLICIES}
            for attack in classes}
    subgroups={}
    for group in ("degree","family"):
        subgroups[group]={}
        for k in sorted({s[group] for s in per_source}):
            cases=[s for s in per_source if s[group]==k]
            subgroups[group][str(k)]={
                "cases":len(cases),
                "means":{field:statistics.mean(s["contrasts"][field]
                         for s in cases) for field in PRIMARY}}
    return {
        "schema":"nor40.original_posthoc_replicate_block_analysis.v1",
        "scientific_pass_as_frozen":full["registered_primary_pass"],
        "registered_five_gates_as_frozen":full["registered_five_gates"],
        "per_source_units":48,"replicate_blocks":8,
        "policy_means":policy_means,"primary_means":means,
        "replicate_block_descriptors":block_stats,
        "calibrated_20x7_crossplay":cross,"subgroups":subgroups,
        "scope":"Descriptive only; no prospective gate recalibration. Exact 16-pool optimum is NOT a global optimization certificate; gap uses the fourth-smallest singleton global lower bound.",
    }


def markdown(a):
    out=["# NOR-40: first frozen-grade posthoc audit","",
         "Original first-run registered PASS/FAIL and five gates remain byte-for-byte unchanged.",
         "All rows use 48 independent source identities grouped into eight replicate blocks.","",
         "| Auditor | CAL256 baseline self-play % | CAL2048 strong self-play % | Strong worst-class % | Exact TOP % | Mean certified CAL global gap upper, pp |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for p,values in a["policy_means"].items():
        out.append(f"| \`{p}\` | {100*values['baseline_self']:.4f} | "
                   f"{100*values['strong_self']:.4f} | "
                   f"{100*values['strong_worst']:.4f} | "
                   f"{100*values['top_exact']:.4f} | "
                   f"{100*values['global_optimism_gap_upper_bound']:.4f} |")
    out+=["",
        "The CAL gap column is the **width of a valid global minimum bracket** "
        "(tested upper bound minus fourth-smallest singleton lower bound), "
        "expressed as detection percentage points. It is often loose, and "
        "the restricted pool enumeration certifies an optimum only in that pool.","",
        "| Paired contrast | Mean pp | Descriptive 95% replicate-block t interval pp | Positive blocks / 8 |",
        "| --- | ---: | --- | ---: |"]
    for k,d in a["replicate_block_descriptors"].items():
        lo,hi=d["descriptive_t95"]
        out.append(f"| \`{k}\` | {100*d['mean']:+.4f} | "
                   f"[{100*lo:+.4f}, {100*hi:+.4f}] | {d['block_positive']} |")
    out+=["","## Exact 20×7 calibrated attack cross-play","","| Attack class | "+" | ".join(POLICIES)+" |",
          "| --- | "+" | ".join("---:" for _ in POLICIES)+" |"]
    for aclass,cols in a["calibrated_20x7_crossplay"].items():
        out.append("| "+aclass+" | "+" | ".join(
            f"{100*cols[p]:.3f}" for p in POLICIES)+" |")
    out+=["","Unrestricted attack minimax values and source-conditioned analytic "
          "uniform floors remain separate from these CAL/EVAL frequencies. "
          "Sign-flip and t-interval summaries depend on eight finite replicate "
          "blocks and have no registered decision role."]
    return "\n".join(out)+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--md-out",type=Path)
    a=p.parse_args()
    report=describe(load(a.full))
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    if a.md_out:
        a.md_out.parent.mkdir(parents=True,exist_ok=True)
        a.md_out.write_text(markdown(report))
    print(markdown(report))


if __name__=="__main__":
    main()
