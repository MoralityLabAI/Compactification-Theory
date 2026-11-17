"""Source-level/replicate-block uncertainty audit for NOR-39 and later receipts.

Descriptive sensitivity only. This module never generates source identities or
changes a preregistered PASS/FAIL threshold. Eight replicate blocks, each
containing the six fixed degree/family cells, are used as the resampling unit.
The exact sign-flip calculation requires exchangeability of block signs.
"""
from __future__ import annotations
import argparse
from itertools import product
import json
from math import isclose
from pathlib import Path
import random
from statistics import mean, median, stdev

REPS=tuple(range(68,76))
CELLS_PER_REP=6
BOOTSTRAP_SEED=202611390091
BOOTSTRAPS=20000
T_CRITICAL_DF7=2.364624251
COMPARISONS={
    "selfplay_vs_deterministic":("global7_self_minus_deterministic_self",.010),
    "worst_class_vs_original":("global7_worst_minus_original_worst",-.010),
    "exact_top_vs_atom_random":("global7_top_minus_atom_random",.020),
}
DESCRIPTIVE_FIELDS=(
    "global7_self_minus_deterministic_self",
    "global7_worst_minus_original_worst",
    "global7_top_minus_atom_random",
    "deterministic_top_minus_atom_random",
)


def exact_signflip_p_greater(blocks,threshold):
    """One-sided >= observed effect against block-symmetric sign permutations."""
    differences=[v-threshold for v in blocks]
    observed=sum(differences)
    count=sum(
        sum(sign*d for sign,d in zip(signs,differences)) >= observed-1e-12
        for signs in product((-1,1),repeat=len(differences)))
    return count/(1<<len(differences))


def percentile(vals,probability):
    ordered=sorted(vals)
    x=(len(ordered)-1)*probability
    lo=int(x)
    hi=min(lo+1,len(ordered)-1)
    return ordered[lo]*(1-(x-lo))+ordered[hi]*(x-lo)


def block_stats(blocks,threshold,seed):
    avg=mean(blocks)
    s=stdev(blocks)
    interval=[avg-T_CRITICAL_DF7*s/(len(blocks)**.5),
              avg+T_CRITICAL_DF7*s/(len(blocks)**.5)]
    rng=random.Random(seed)
    draws=[mean(rng.choices(blocks,k=len(blocks))) for _ in range(BOOTSTRAPS)]
    return {
        "block_mean":avg,"block_median":median(blocks),
        "block_min":min(blocks),"block_max":max(blocks),
        "block_standard_deviation":s,
        "descriptive_t95_interval":interval,
        "block_bootstrap_95_percentile_interval":[percentile(draws,.025),
                                                   percentile(draws,.975)],
        "block_signflip_one_sided_p_at_frozen_threshold":
            exact_signflip_p_greater(blocks,threshold),
        "threshold":threshold,
        "blocks_positive_to_threshold":sum(v>threshold for v in blocks),
        "leave_one_block_out_means":[mean(blocks[:i]+blocks[i+1:])
                                     for i in range(len(blocks))]
    }


def analyze(result):
    assert result["schema"]=="nor39.clean_global_uniform_policyaware.v1"
    rows=result["source_level_primary"]
    assert result["fresh_source_units"]==len(rows)==48
    assert set(r["replicate"] for r in rows)==set(REPS)
    assert all(sum(r["replicate"]==rep for r in rows)==CELLS_PER_REP
               for rep in REPS)
    assert len({r["public_sha256"] for r in rows})==48
    block={
        str(rep):{key:mean(r[key] for r in rows if r["replicate"]==rep)
                  for key in DESCRIPTIVE_FIELDS}
        for rep in REPS}
    out={}
    for i,(label,(key,threshold)) in enumerate(COMPARISONS.items()):
        unit_means=[block[str(rep)][key] for rep in REPS]
        report=block_stats(unit_means,threshold,BOOTSTRAP_SEED+i)
        report["source_equal_weight_mean"]=mean(r[key] for r in rows)
        report["source_win_count"]=sum(r[key]>0 for r in rows)
        report["source_margin_at_threshold_count"]=sum(
            r[key]>=threshold for r in rows)
        assert isclose(report["source_equal_weight_mean"],report["block_mean"],
                       abs_tol=1e-12)
        assert isclose(report["source_equal_weight_mean"],result["source_means"][key],
                       abs_tol=1e-12)
        out[label]=report
    def group_means(key,group):
        vals=sorted(set(r[group] for r in rows))
        return {str(v):mean(r[key] for r in rows if r[group]==v) for v in vals}
    subgroups={key:{"degree":group_means(key,"degree"),
                    "family":group_means(key,"family")}
               for key in DESCRIPTIVE_FIELDS}
    tops=[block[str(rep)]["global7_top_minus_atom_random"] for rep in REPS]
    dets=[block[str(rep)]["deterministic_top_minus_atom_random"] for rep in REPS]
    base=mean(dets)
    ratio=mean(tops)/base if base>0 else None
    if ratio is not None:
        assert isclose(ratio,
                       result["retained_ratio_global7_to_deterministic_top_gain"],
                       abs_tol=1e-12)
    return {
        "schema":"nor39.exploratory_block_audit.v1",
        "status":"NONREGISTERED descriptive/replicate-block uncertainty only",
        "source_units":48,"source_replicate_blocks":8,
        "within_block_degree_family_cells":6,
        "bootstrap_replicates":BOOTSTRAPS,
        "bootstrap_seed":BOOTSTRAP_SEED,
        "block_mean_ratios":{"global7_retained_fraction_top":ratio},
        "registered_grade_flags_unmodified":result["registered_five_gates"],
        "source_directions":result["source_direction_wins"],
        "comparisons":out,"subgroup_means":subgroups,
        "caveat":"Eight finite replicate blocks. Sign-flip p requires exchangeable block signs; t and bootstrap intervals are descriptive, not population guarantees. Original preregistered PASS/FAIL remains exactly as first scored."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--full",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=analyze(json.loads(args.full.read_bytes()))
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text,encoding="utf-8")
    print(text)


if __name__=="__main__":
    main()
