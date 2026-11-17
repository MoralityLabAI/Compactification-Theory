"""NOR-42 permanent read-only first archived historical-smoke SHA verifier.

No new NOR42 sources and no dynamic experiment regeneration. Every original
record is checked against the exact hypergeometric uniform16 probability.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb,isclose
from pathlib import Path

PATH=Path(__file__).resolve().parent/"results/nor42_archived_48_smoke.json"
ORIGINAL_BYTES=36963
ORIGINAL_SHA="411e6e1e30ce262a5604cb60e99b314ff3742413f8d526f4257585a48666f3f9"
PREVIOUS_SOURCE_SHA="b51e8031715754c36661b74cff0c26e672fdc21355ff0dc9ab48efded3901de5"
PREVIOUS_GRADE_SHA="42adeebcbf92626c080a6f0f5cb906eed491d6b5ed67250ba7b70bd6d0559092"


def audit():
    original=PATH.read_bytes()
    assert len(original)==ORIGINAL_BYTES
    assert sha256(original).hexdigest()==ORIGINAL_SHA
    x=json.loads(original)
    assert x["schema"]=="nor42.archived_nor40_late_suffix_smoke.v1"
    assert x["source_archive_sha256"]==PREVIOUS_SOURCE_SHA
    assert x["graded_nor40_archive_sha256"]==PREVIOUS_GRADE_SHA
    assert x["historical_source_units"]==48
    assert x["new_nor42_source_units"]==0
    assert x["policy_audit_budget"]==16
    assert x["hidden_suffix_offset_after_public"]==32
    records=x["records"]
    assert len(records)==len({
        row["archived_source_sha256"] for row in records})==48
    assert Counter(r["degree"] for r in records)=={512:24,1024:24}
    assert Counter(r["source_family"] for r in records)=={
        "wide_dense_small":16,"wide_rational_short":16,"wide_sparse_small":16}
    assert Counter(r["archived_replicate"] for r in records)=={
        r:6 for r in range(76,84)}
    for row in records:
        m=row["future_length"]
        s=row["difference_support_size"]
        assert row["degree"] in (512,1024)
        assert m==3*row["degree"]//2+1
        assert 1<=s<=m
        assert row["first_difference_index"]==row["last_public_index"]+32
        assert row["native_clean_template_mask_cardinality"]==1
        assert row["hidden_suffix_beta"] in (-2,-1,1,2)
        assert row["hidden_suffix_transform"] in ("I","N","R","S")
        assert row["paired_public_prefix_equal"]
        assert not row["paired_future_targets_equal"]
        exact=(Fraction(1) if m-s<16 else
               Fraction(1)-Fraction(comb(m-s,16),comb(m,16)))
        assert row["uniform16_exact_hit_probability_fraction"]==[
            exact.numerator,exact.denominator]
        assert isclose(row["uniform16_exact_hit_probability"],float(exact),
                       abs_tol=1e-16)
    for degree in (512,1024):
        group=[r for r in records if r["degree"]==degree]
        row=x["by_degree"][str(degree)]
        assert row["n"]==24
        assert isclose(row["difference_support_mean"],statistics.mean(
            r["difference_support_size"] for r in group),abs_tol=1e-12)
        assert isclose(row["uniform16_exact_hit_mean"],statistics.mean(
            r["uniform16_exact_hit_probability"] for r in group),abs_tol=1e-15)
    return {
        "schema":"nor42.original_historical_smoke_byte_and_probability_audit.v1",
        "original_sha256":ORIGINAL_SHA,
        "original_bytes":ORIGINAL_BYTES,
        "archived_source_count":48,
        "new_prospective_source_count":0,
        "exact_fraction_reconstructions":48,
        "by_degree":x["by_degree"],
    }


if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
