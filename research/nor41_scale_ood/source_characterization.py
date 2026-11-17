"""Read-only NOR-41 original first public-only source input diagnostics.

Requires exact SHA-pinned original source fixture, consumes neither private
target programs nor CAL/EVAL plans. Used only AFTER the source-only fixture
was frozen; not part of pre-data acceptance or scientific endpoint selection.
"""
from __future__ import annotations
import argparse,collections,json,statistics
from hashlib import sha256
from pathlib import Path

SOURCE_SHA="4073090e5888ed10283d82dc5153758d299ed7cfe9c753be39ba2c2bd4c7119d"
SOURCE_BYTES=9618649


def characterize(raw):
    if len(raw)!=SOURCE_BYTES or sha256(raw).hexdigest()!=SOURCE_SHA:
        raise AssertionError("Original first public-only source bytes changed")
    f=json.loads(raw)
    assert f["schema"]=="nor41.scale_ood_public_only_reps84to99.v1"
    assert f["registered_first_commit"]=="87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f"
    assert f["predata_amendment_01_commit"]=="e5a95cad1ffdcd019f153cd09dd4d48b40848ef9"
    assert f["public_only"] and f["planted_and_private_target_absent"]
    assert f["attack_witness_and_audit_absent"]
    assert f["excluded_historical_sources"]==324
    rows=f["cases"]
    assert len(rows)==len({r["full_public_input_sha256"] for r in rows})==64
    results={family:[] for family in ("wide_dense_small","wide_regime_blocks")}
    for row in rows:
        q=row["public_query_input"]
        n=len(q)-1
        assert n==row["query_degree"]==3*row["degree"]//2
        assert q[0]==1
        result={
            "public_sha256":row["full_public_input_sha256"],
            "replicate":row["replicate"],"degree":row["degree"],
            "native_mask_cardinality":row["original_native_public_mask"].bit_count(),
            "density":sum(x!=0 for x in q[1:])/n
        }
        if row["family"]=="wide_regime_blocks":
            by_kind={k:[] for k in range(4)}
            for start in range(1,len(q),64):
                kind=((start-1)//64)%4
                segment=q[start:min(start+64,len(q))]
                by_kind[kind].append(sum(x!=0 for x in segment)/len(segment))
            result["mean_regime_rates"]={
                str(k):statistics.mean(by_kind[k]) for k in range(4)}
        results[row["family"]].append(result)
    assert all(len(records)==32 for records in results.values())
    return {
        "schema":"nor41.frozen_public_input_characterization.v1",
        "fixture_sha256":SOURCE_SHA,
        "source_count":64,
        "families":{
            family:{
                "n":len(rows),"query_density_mean":statistics.mean(r["density"] for r in rows),
                "query_density_min":min(r["density"] for r in rows),
                "query_density_max":max(r["density"] for r in rows),
                "native_mask_cardinality":dict(collections.Counter(
                    r["native_mask_cardinality"] for r in rows)),
                **({"mean_regime_nonzero_rates":{
                    str(k):statistics.mean(r["mean_regime_rates"][str(k)] for r in rows)
                    for k in range(4)}} if family=="wide_regime_blocks" else {})
            } for family,rows in results.items()
        },
        "scope":"Public-input density only. Input OOD within unchanged RRC25 native program grammar. No private target, CAL attacks or EVAL outcomes read."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    result=characterize(a.fixture.read_bytes())
    output=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(output,encoding="utf-8")
    print(output)


if __name__=="__main__":
    main()
