"""NOR-39 exploratory source-level inference and baseline audit.

Reads ONLY saved graded receipts. Its outputs are POSTHOC descriptors, never
used to select a NOR39 policy, modify the frozen gates, or generate attacks.
48 new source identities share eight replicate blocks, so intervals use eight
replicate means as the unit. An optional NOR38 prior receipt gives a paired
policy cross-cohort descriptive comparison on disjoint source identities.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from fractions import Fraction
from itertools import product
from pathlib import Path
from zipfile import ZipFile

POLICIES=("uniform16","original_mix","deterministic_mix","stochastic50_mix",
          "atom_random_h50","global7_mix","global8_mix")
CONTRASTS=(
    ("global7_self_minus_deterministic_self","Global7 vs deterministic self-play"),
    ("global7_worst_minus_original_worst","Global7 vs original worst-class"),
    ("global7_top_minus_atom_random","Global7 vs atom-random TOP"),
)
TCRIT7=2.3646242510103


def load(path):
    if path.suffix==".zip":
        with ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith("_full.json")]
            if len(names)!=1:raise ValueError("Expected exactly one graded receipt in ZIP")
            return json.loads(z.read(names[0]))
    return json.loads(path.read_bytes())


def describe(full):
    records=full["source_records"]
    policies=full["auditor_policies"]
    assert tuple(policies)==POLICIES and len(records)==48
    distinct_sources={r["public_sha256"] for r in records}
    assert len(distinct_sources)==48
    reps=sorted(set(r["replicate"] for r in records))
    assert len(reps)==8
    assert all(sum(r["replicate"]==rep for r in records)==6 for rep in reps)

    def rate(rec,policy,attacker):
        return rec["crossplay_weighted_numerators"][attacker][policy]/rec["crossplay_denominator"]

    per_source=[]
    for rec in records:
        top={p:float(Fraction(*rec["exact_top_support_detection"][p])) for p in POLICIES}
        by_policy={p:{
            "self":rate(rec,p,p),
            "worst":min(rate(rec,p,att) for att in POLICIES),
            "top":top[p],
        } for p in POLICIES}
        u=Fraction(*rec["exact_full_uniform_attack_detection_fraction"])
        sample={
            "source":rec["public_sha256"],
            "replicate":rec["replicate"],"degree":rec["degree"],
            "family":rec["family"],"policies":by_policy,
            "uniform_maximin":float(u),
            "global7_ideal_floor":float(Fraction(7,16)*u),
            "global8_ideal_floor":float(Fraction(1,2)*u),
        }
        sample["contrasts"]={
            "global7_self_minus_deterministic_self":
                by_policy["global7_mix"]["self"]-by_policy["deterministic_mix"]["self"],
            "global7_worst_minus_original_worst":
                by_policy["global7_mix"]["worst"]-by_policy["original_mix"]["worst"],
            "global7_top_minus_atom_random":
                by_policy["global7_mix"]["top"]-by_policy["atom_random_h50"]["top"],
        }
        per_source.append(sample)

    aggregate={}
    for p in POLICIES:
        aggregate[p]={key:statistics.mean(r["policies"][p][key]
                     for r in per_source)
                      for key in ("self","worst","top")}
    all_contrasts={}
    for key,label in CONTRASTS:
        by_block=[statistics.mean(r["contrasts"][key] for r in per_source
                    if r["replicate"]==rep) for rep in reps]
        observed=statistics.mean(by_block)
        se=statistics.stdev(by_block)/math.sqrt(len(by_block))
        tail=sum(1 for signs in product((-1,1),repeat=len(by_block))
                 if abs(sum(s*v for s,v in zip(signs,by_block)))>=
                    abs(sum(by_block))-1e-15)/(2**len(by_block))
        all_contrasts[key]={
            "label":label,
            "mean":observed,"block_means":by_block,
            "block_positive":sum(x>0 for x in by_block),
            "source_positive":sum(r["contrasts"][key]>0 for r in per_source),
            "replicate_t95":[observed-TCRIT7*se,observed+TCRIT7*se],
            "exploratory_signflip_p":tail,
        }
    groups={}
    for attr in ("degree","family"):
        groups[attr]={}
        for value in sorted({r[attr] for r in per_source}):
            subset=[r for r in per_source if r[attr]==value]
            groups[attr][str(value)]={
                "count":len(subset),
                "contrasts":{key:statistics.mean(r["contrasts"][key] for r in subset)
                             for key,_ in CONTRASTS},
                "source_wins":{key:sum(r["contrasts"][key]>0 for r in subset)
                               for key,_ in CONTRASTS},
            }
    gate_wins=full["source_direction_wins"]
    for key in ("global7_self_minus_deterministic_self",
                "global7_top_minus_atom_random"):
        assert all_contrasts[key]["source_positive"]==gate_wins[key]
    assert abs(all_contrasts["global7_worst_minus_original_worst"]["mean"]-
               full["source_means"]["global7_worst_minus_original_worst"])<1e-12
    result={
        "schema":"nor39.posthoc_source_block_analysis.v1",
        "source_units":48,"independent_replicate_blocks":8,
        "registered_primary_pass_as_recorded":full["registered_primary_pass"],
        "registered_five_gates_as_recorded":full["registered_five_gates"],
        "policy_rates":aggregate,"contrasts":all_contrasts,
        "strata":groups,
        "narrative_scope":"Exploratory descriptive diagnostics on frozen receipts; "
            "eight replicate-block t intervals and finite sign-flip checks "
            "are not guarantees for unrelated programs. Greedy attacks are approximate.",
    }
    return result


def as_markdown(a,cohort):
    lines=[
        f"# {cohort}: exploratory source-level audit",
        "",
        "Original scientific PASS/FAIL remains exactly as recorded in the frozen grader.",
        "All quantities in this report were computed after first held-out scoring.",
        "The 48 source rows are grouped into eight replicate blocks of six related cases.",
        "",
        "| Auditor | Self-play % | Sourcewise worst-class % | Exact TOP % |",
        "| --- | ---: | ---: | ---: |",
    ]
    for p,x in a["policy_rates"].items():
        lines.append(f"| \`{p}\` | {100*x['self']:.4f} | {100*x['worst']:.4f} | {100*x['top']:.4f} |")
    lines.extend(["",
      "| Source-paired difference | Mean pp | 95% replicate-block t interval pp | "
      "Positive blocks / 8 | Positive source rows / 48 |",
      "| --- | ---: | --- | ---: | ---: |"])
    for val in a["contrasts"].values():
        lower,upper=val["replicate_t95"]
        lines.append(
            f"| {val['label']} | {100*val['mean']:+.4f} | "
            f"[{100*lower:+.4f}, {100*upper:+.4f}] | "
            f"{val['block_positive']} | {val['source_positive']} |")
    for key,values in a["strata"].items():
        lines.extend(["",f"## By {key}","","| Group | N | Self-play gain pp | Worst gap pp | TOP gain pp |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for group,stats in values.items():
            diffs=stats["contrasts"]
            lines.append(f"| {group} | {stats['count']} | "+
                " | ".join(f"{100*diffs[k]:+.3f}" for k,_ in CONTRASTS)+" |")
    lines.extend(["",
      "The unrestricted all-output r4/k16 minimax ceiling is attained by "
      "uniform16, while the explicit global7 branch has the exact theoretical "
      "fixed-support floor of (7/16) times that ceiling. Finite CAL/EVAL "
      "greedy worst-class rates are separate measurements.",
      "",
      "All uncertainty intervals and subgroup readings are exploratory. "
      "The 8-block sign-flip calculations are conditional on blockwise sign "
      "symmetry and are not used as registered gates.",
      ])
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--markdown-out",type=Path)
    p.add_argument("--cohort",default="NOR-39")
    args=p.parse_args()
    a=describe(load(args.full))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True,exist_ok=True)
        args.json_out.write_text(json.dumps(a,sort_keys=True,indent=2)+"\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True,exist_ok=True)
        args.markdown_out.write_text(as_markdown(a,args.cohort))
    print(as_markdown(a,args.cohort))


if __name__=="__main__":
    main()
