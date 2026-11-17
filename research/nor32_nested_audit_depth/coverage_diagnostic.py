"""NOR-32 labeled RETROSPECTIVE coverage diagnostic, original first-run bytes.

Loads the first registered source-controlled original ZIP and receipt,
inspects the existing 2,048-schedule outcome rows only, then compares
exact observed batch-coverage counts with ideal uniform coupon-collector
expectations. It never changes the frozen source roster or PASS/FAIL.
"""
from __future__ import annotations
from pathlib import Path
from math import isclose
from hashlib import sha256
import json,zipfile,statistics

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/"results"/"nor32_original_hosted.zip"
RECEIPT=HERE/"results"/"nor32_first_original_receipt.json"
DEPTHS=(512,1024,1536,2048)
ARMS=("hybrid_50","uniform16","static_prior")
M=(769,1537)


def analyze():
    r=json.loads(RECEIPT.read_bytes())
    with zipfile.ZipFile(ARCHIVE) as z:
        info=r["original_raw"]["outcome"]
        original=z.read(info["file"])
        assert len(original)==info["bytes"]
        assert sha256(original).hexdigest()==info["sha256"]
        full=json.loads(original)
    sources=full["source_records"]
    assert len(sources)==24
    by_group={}
    for m in M:
        members=[s for s in sources if s["m"]==m]
        assert len(members)==12
        by_group[str(m)]={}
        for arm in ARMS:
            by_group[str(m)][arm]={}
            for n in DEPTHS:
                seen=[s["never_inspected"][arm][str(n)] for s in members]
                f=sum(v>=4 for v in seen)
                assert f==full["batch_zero_hit_feasibility"][arm][str(n)][str(m)]
                if arm=="hybrid_50":
                    N,u=m-8,8
                elif arm=="uniform16":
                    N,u=m,16
                else:
                    N,u=m-16,0
                expected=N*((1-u/N)**n)
                reported=full["ideal_expected_never_inspected"][str(m)][arm][str(n)]
                assert isclose(expected,reported,abs_tol=1e-9)
                by_group[str(m)][arm][str(n)]={
                    "sources_with_four_untouched":f,
                    "mean_uninspected":statistics.mean(seen),
                    "min":min(seen),"max":max(seen),
                    "ideal_uniform_expected_uninspected":expected,
                }
            for shorter,longer in zip(DEPTHS,DEPTHS[1:]):
                assert all(s["never_inspected"][arm][str(shorter)]>=
                           s["never_inspected"][arm][str(longer)] for s in members)
    assert [by_group["1537"]["hybrid_50"][str(n)]["sources_with_four_untouched"]
            for n in DEPTHS]==[12,11,0,0]
    assert [by_group["769"]["hybrid_50"][str(n)]["sources_with_four_untouched"]
            for n in DEPTHS]==[3,0,0,0]
    print("NOR32 ORIGINAL RETROSPECTIVE COUPON-COLLECTOR COVERAGE PASS")
    print(json.dumps(by_group,sort_keys=True,indent=2))
    return by_group


if __name__=="__main__":
    analyze()
