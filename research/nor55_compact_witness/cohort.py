"""NOR55: new 60 public-only exact formal-series source identities.

Source-only fixture creation is separate from all inverse, witness, root,
signature, session, payload, and CPU timing creation.
"""
from __future__ import annotations
import argparse,json,random,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
REGISTERED="9f7fb42a2e0a8648db63301bed381cb7f3aa0dc2"
SCHEMA="nor55.first_public_only_60_signed_compact_sources.v1"
CASE_SCHEMA="nor55.source_id_from_full_public_coefficients.v1"
SOURCE_DOMAIN="NOR55/SOURCE/v1"
REPS=(0,1)
LENGTHS=(769,1537,3073)
PROFILES=("s1-b2","s4-b2","s8-b2","s4-b8","s4-b16")
FAMILIES=("signed_random","alternating")
PROFILE_PARAMS={"s1-b2":(1,2),"s4-b2":(4,2),"s8-b2":(8,2),
                "s4-b8":(4,8),"s4-b16":(4,16)}
PRIOR_ARCHIVES=(
 ("research/nor51_exact_r_crossover/results/nor51_first_original_hosted.zip",
  "b8771c20b24b7868ff467f9559cba6c551021042943db8d308c204d0d75e39d9",
  "nor51_first_48_source_only.json",
  "0b0de5549de1e1815a40d56302cb01ca104e7ac8141f161bc8cab0b6501d2c93",48),
 ("research/nor52_sparse_inverse/results/nor52_first_original_hosted.zip",
  "1905b3835ed6dea870994a154d8a56942fce0b0fcf83d4058417abc543e206a6",
  "nor52_first_public_only_60_sources.json",
  "d18b795053d81c16fc5316b490316e90fe6b373156a8ee8b295008a13df150f4",60),
 ("research/nor53_native_pack/results/nor53_first_original_hosted.zip",
  "a44856da93943c82a5265bceadad06374f2ccb6050626f10477b6470d6bcd0ff",
  "nor53_first_original_60_public_sources.json",
  "b7e65cdacefced6fcf1084922fdaeaaa4da5af1cf9066681471e4be1cd99ad61",60),
 ("research/nor54_matched_frontier/results/nor54_first_original_hosted.zip",
  "8d5b5745127a5086910f2d4c73b69eb039a6b81e3417060868894366a3129656",
  "nor54_first_original_60_public_only.json",
  "9ac35fdee19a8b9a5b2d08c1ff890bddb3141b69232c4a2687d46195c37a627d",60)
)

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def earlier_source_ids():
    earlier=set()
    for path,archive_sha,member,fixture_sha,count in PRIOR_ARCHIVES:
        p=ROOT/path
        assert sha256(p.read_bytes()).hexdigest()==archive_sha
        with ZipFile(p) as z:
            assert z.testzip() is None
            raw=z.read(member)
        assert sha256(raw).hexdigest()==fixture_sha
        cases=json.loads(raw)["cases"]
        ids={r["source_sha256"] for r in cases}
        assert len(ids)==count and ids.isdisjoint(earlier)
        earlier.update(ids)
    assert len(earlier)==228
    return earlier

def source_seed(rep,m,profile,family):
    assert rep in REPS and m in LENGTHS and profile in PROFILES and family in FAMILIES
    token=f"{SOURCE_DOMAIN}|{rep}|{m}|{profile}|{family}"
    return int.from_bytes(sha256(token.encode("ascii")).digest(),"big")

def source_series(rep,m,profile,family):
    stride,bits=PROFILE_PARAMS[profile]
    rng=random.Random(source_seed(rep,m,profile,family))
    out=[1]
    for i in range(1,m):
        if i%stride:
            out.append(0)
        else:
            mag=rng.randint(1<<(bits-1),(1<<bits)-1)
            sign=(-1 if rng.randrange(2) else 1) if family=="signed_random" else (
                  -1 if (i//stride)%2 else 1)
            out.append(sign*mag)
    return tuple(out)

def source_core(rep,m,profile,family):
    stride,bits=PROFILE_PARAMS[profile]
    return {
        "case_schema":CASE_SCHEMA,
        "replicate":rep,"length":m,"profile":profile,"family":family,
        "stride":stride,"coefficient_magnitude_bits":bits,
        "source_coefficients_decimal":[str(x) for x in source_series(
            rep,m,profile,family)]
    }

def freeze():
    prior=earlier_source_ids()
    rows=[]
    for rep in REPS:
        for m in LENGTHS:
            for profile in PROFILES:
                for family in FAMILIES:
                    core=source_core(rep,m,profile,family)
                    rows.append({**core,"source_sha256":sha256(canonical(core)).hexdigest()})
    current={r["source_sha256"] for r in rows}
    assert len(rows)==len(current)==60 and current.isdisjoint(prior)
    return {
        "schema":SCHEMA,"first_registration_sha":REGISTERED,
        "source_domain":SOURCE_DOMAIN,
        "replicates":list(REPS),"lengths":list(LENGTHS),
        "profiles":list(PROFILES),"families":list(FAMILIES),
        "source_count":60,"earlier_excluded_source_count":228,
        "public_only":True,"no_inverse_root_witness_session_payload_or_timing":True,
        "cases":rows
    }

def read_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    assert canonical(x)==raw
    assert x["schema"]==SCHEMA and x["first_registration_sha"]==REGISTERED
    assert x["source_domain"]==SOURCE_DOMAIN
    assert x["replicates"]==list(REPS) and x["lengths"]==list(LENGTHS)
    assert x["profiles"]==list(PROFILES) and x["families"]==list(FAMILIES)
    assert x["source_count"]==60 and x["earlier_excluded_source_count"]==228
    assert x["public_only"] and x["no_inverse_root_witness_session_payload_or_timing"]
    rows=x["cases"]
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(earlier_source_ids())
    assert Counter(r["length"] for r in rows)=={m:20 for m in LENGTHS}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert {(r["replicate"],r["length"],r["profile"],r["family"])
            for r in rows}=={
        (rep,m,profile,family) for rep in REPS for m in LENGTHS
        for profile in PROFILES for family in FAMILIES}
    for row in rows:
        assert set(row)=={
            "case_schema","replicate","length","profile","family",
            "stride","coefficient_magnitude_bits",
            "source_coefficients_decimal","source_sha256"}
        core={k:v for k,v in row.items() if k!="source_sha256"}
        assert core==source_core(row["replicate"],row["length"],row["profile"],row["family"])
        assert sha256(canonical(core)).hexdigest()==row["source_sha256"]
        assert len(row["source_coefficients_decimal"])==row["length"]
        assert row["source_coefficients_decimal"][0]=="1"
        assert all(type(v) is str and str(int(v))==v for v in row["source_coefficients_decimal"])
    return x,sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only-out",type=Path,required=True)
    args=p.parse_args()
    raw=canonical(freeze())
    args.source_only_out.parent.mkdir(parents=True,exist_ok=True)
    args.source_only_out.write_bytes(raw)
    _,source_sha=read_fixture(args.source_only_out)
    print("NOR55 FIRST 60 PUBLIC-ONLY SOURCE RAW BYTES SHA256",len(raw),source_sha,flush=True)

if __name__=="__main__":
    main()
