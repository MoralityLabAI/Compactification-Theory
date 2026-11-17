"""NOR-42 historical-only paired late-suffix smoke gym.

Never generates a NOR-42 source identity. Public inputs come from the already
SHA-archived, independently replayed NOR-40 first fixture. This is a
source-independent prototype for separately preregistered future NOR42 work.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor37_stochastic_curvature import experiment as nor37

ORIGINAL=ROOT/"research/nor40_strong_adversaries/results/nor40_original_first_hosted.zip"
SOURCE_BYTES=9387973
SOURCE_SHA="b51e8031715754c36661b74cff0c26e672fdc21355ff0dc9ab48efded3901de5"
GRADE_BYTES=3492789
GRADE_SHA="42adeebcbf92626c080a6f0f5cb906eed491d6b5ed67250ba7b70bd6d0559092"
AMPLITUDES=(-2,-1,1,2)
OPERATIONS=tuple(corpus.OPS)
AUDIT_K=16
OBSERVED_DEGREES=(512,1024)
OFFSET_AFTER_PUBLIC=32


def canonical(data):
    return (json.dumps(data,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def original_history():
    """Read only the original NOR40 first fixture and grade SHA pins."""
    with ZipFile(ORIGINAL) as z:
        assert z.testzip() is None
        fixture=z.read("nor40_fixture.json")
        grade=z.read("nor40_full.json")
    assert len(fixture)==SOURCE_BYTES and sha256(fixture).hexdigest()==SOURCE_SHA
    assert len(grade)==GRADE_BYTES and sha256(grade).hexdigest()==GRADE_SHA
    data=json.loads(fixture)
    assert data["schema"]=="nor40.strong_adversary_public_only_reps76to83.v1"
    assert data["public_only"] and data["planted_and_private_target_absent"]
    assert data["attack_witness_and_audit_absent"]
    rows=data["cases"]
    assert len(rows)==len({r["full_public_input_sha256"] for r in rows})==48
    return rows


def hidden_future(query,clean,observed_degree,beta,operation):
    """Exact hidden β X^(D+32) T(F), zero through public degree D.

    This constructs a hypothetical true future, not a trusted external
    oracle or an inference from the public clean proposal.
    """
    assert query and query[0]==1 and len(query)==len(clean)
    assert observed_degree in OBSERVED_DEGREES
    assert beta in AMPLITUDES and operation in OPERATIONS
    onset=observed_degree+OFFSET_AFTER_PUBLIC
    assert observed_degree<onset<len(query)
    transformed=corpus.d.operate(query,operation)
    assert len(transformed)==len(query) and transformed[0]==1
    hidden=tuple(
        original if index<onset else
        original+beta*transformed[index-onset]
        for index,original in enumerate(clean))
    assert hidden[:observed_degree+1]==tuple(clean[:observed_degree+1])
    assert hidden[onset]-clean[onset]==beta
    assert all(a==b for a,b in zip(clean[:onset],hidden[:onset]))
    return hidden


def discrepancy_support(clean,hidden):
    assert len(clean)==len(hidden)
    return frozenset(i for i,(a,b) in enumerate(zip(clean,hidden)) if a!=b)


def uniform_query_hit_probability(total,changed,checks=AUDIT_K):
    """Exact probability of detecting a fixed difference set, hidden k-subset."""
    assert 0<=changed<=total and 0<=checks<=total
    if changed==0:return Fraction(0)
    if total-changed<checks:return Fraction(1)
    return Fraction(1)-Fraction(comb(total-changed,checks),comb(total,checks))


def oracle_detects_proposal(clean_proposal,true_target,checked):
    """Conditional on independent authenticated true coefficients."""
    assert len(clean_proposal)==len(true_target)
    assert len(checked)==len(set(checked))==AUDIT_K
    assert all(0<=i<len(clean_proposal) for i in checked)
    return any(clean_proposal[i]!=true_target[i] for i in checked)


def clean_internal_consistency(clean_proposal,inferred_clean,checked):
    """Witness-FREE consistency with a model's OWN inferred clean output."""
    return not oracle_detects_proposal(clean_proposal,inferred_clean,checked)


def historical_smoke():
    """48 previously completed NOR40 source rows; zero future NOR42 identities."""
    rows=original_history()
    records=[]
    for n,row in enumerate(rows):
        source=row["full_public_input_sha256"]
        predicted=nor37.public_proposal(row)
        clean=predicted["pred"]
        assert predicted["mask"]==row["original_native_public_mask"]>0
        query=tuple(row["public_query_input"])
        degree=row["degree"]
        # Fixed deterministic illustrative scenario, not a registered NOR42 draw.
        operation=OPERATIONS[n%len(OPERATIONS)]
        beta=AMPLITUDES[n%len(AMPLITUDES)]
        hidden=hidden_future(query,clean,degree,beta,operation)
        indices=discrepancy_support(clean,hidden)
        onset=degree+OFFSET_AFTER_PUBLIC
        assert min(indices)==onset
        exact=uniform_query_hit_probability(len(clean),len(indices))
        # A uniformly sampled k16 witness schedule is calculated analytically.
        # We also test exact honest equality and the on/off support condition.
        all_ok=all(clean[i]==hidden[i] for i in range(onset))
        assert all_ok and hidden[onset]!=clean[onset]
        records.append({
            "archived_source_sha256":source,
            "archived_replicate":row["replicate"],
            "degree":degree,"future_length":len(clean),
            "source_family":row["family"],
            "hidden_suffix_beta":beta,"hidden_suffix_transform":operation,
            "first_difference_index":onset,
            "last_public_index":degree,
            "difference_support_size":len(indices),
            "native_clean_template_mask_cardinality":predicted["mask"].bit_count(),
            "uniform16_exact_hit_probability_fraction":[
                exact.numerator,exact.denominator],
            "uniform16_exact_hit_probability":float(exact),
            "paired_public_prefix_equal":True,
            "paired_future_targets_equal":False
        })
    assert len(records)==48
    assert all(r["native_clean_template_mask_cardinality"]==1 for r in records)
    assert all(r["paired_public_prefix_equal"] and not r["paired_future_targets_equal"]
               for r in records)
    by_degree={
        str(d):{
            "n":sum(r["degree"]==d for r in records),
            "difference_support_mean":statistics.mean(
                r["difference_support_size"] for r in records if r["degree"]==d),
            "uniform16_exact_hit_mean":statistics.mean(
                r["uniform16_exact_hit_probability"]
                for r in records if r["degree"]==d)
        } for d in OBSERVED_DEGREES}
    return {
        "schema":"nor42.archived_nor40_late_suffix_smoke.v1",
        "source_archive_sha256":SOURCE_SHA,
        "graded_nor40_archive_sha256":GRADE_SHA,
        "historical_source_units":48,
        "new_nor42_source_units":0,
        "policy_audit_budget":AUDIT_K,
        "hidden_suffix_offset_after_public":OFFSET_AFTER_PUBLIC,
        "by_degree":by_degree,
        "records":records,
        "limitation":"Hypothetical exact hidden truth in completed NOR40 fixtures. "
                     "No cryptographic authentication implementation, no "
                     "prospective NOR42 sources, no new registered evaluation."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=historical_smoke()
    buf=canonical(result)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(buf)
    print("NOR42 HISTORICAL SMOKE 48 ARCHIVED SOURCES",sha256(buf).hexdigest())
    print("NOR42 BY DEGREE",result["by_degree"])


if __name__=="__main__":
    main()
