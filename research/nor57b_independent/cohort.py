"""NOR57B exact first public-only 60-case SHA source fixture generator.

This module never creates the inverse, witness, output root, nonce, signature,
compact wire or CPU timing. Its input identities exclude the first NOR51–55
source SHA records from the original permanently archived first ZIPs.
"""
from __future__ import annotations
import argparse,json,random,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
REGISTERED="0a9467ae18df3296d8fdf04971f0ffbbb4c6e3d7"
SCHEMA="nor57b.first_public_only_60_authenticated_router_sources.v1"
CASE_SCHEMA="nor57b.public_authenticated_selector_source.v1"
SOURCE_DOMAIN="NOR57B/SOURCE/v1"
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
  "9ac35fdee19a8b9a5b2d08c1ff890bddb3141b69232c4a2687d46195c37a627d",60),
 ("research/nor55_compact_witness/results/nor55_first_original_hosted.zip",
  "f16ed7e1397b863c0dd00175f64a31a15ba61fbfc0667cf002990f732f7d946a",
  "nor55_first_original_60_public_only.json",
  "d1e6b2f5f37edc37f4b996b42e10768b47b4c199ed274cf53643be34fddddd59",60),
 ("research/nor56_batch_codec/results/nor56_first_original_hosted.zip",
  "73a0822822594e1229b4fababee1400b22bddb4e5a06ae87d57e1385788833c3",
  "nor56_first_original_60_public_only.json",
  "f1326f4ac9cac314564c96eb82008177f30280b38ab8972eab696f51996b421c",60),
 ("research/nor57_authenticated_router/results/nor57_first_original_failure_hosted.zip",
  "d6feae134d340a4f15535b4bcc14e9800da3d559213f51203895c1c0c70b72c5",
  "nor57_first_original_60_public_only.json",
  "3eed34d3b80fbae3111d0fcad80df0865df3be78f0077460ac9349647616abe2",60)
)
def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")
def historical_ids():
    prior=set()
    for path,archive_sha,member,source_sha,count in PRIOR_ARCHIVES:
        p=ROOT/path
        assert sha256(p.read_bytes()).hexdigest()==archive_sha
        with ZipFile(p) as z:
            assert z.testzip() is None
            raw=z.read(member)
        assert sha256(raw).hexdigest()==source_sha
        cases=json.loads(raw)["cases"]
        ids={r["source_sha256"] for r in cases}
        assert len(ids)==count and ids.isdisjoint(prior)
        prior.update(ids)
    assert len(prior)==408
    return prior
def cell_seed(rep,m,profile,family):
    assert rep in REPS and m in LENGTHS
    assert profile in PROFILES and family in FAMILIES
    token=f"{SOURCE_DOMAIN}|{rep}|{m}|{profile}|{family}"
    return int.from_bytes(sha256(token.encode("ascii")).digest(),"big")
def source_series(rep,m,profile,family):
    s,b=PROFILE_PARAMS[profile]
    rng=random.Random(cell_seed(rep,m,profile,family))
    a=[1]
    for i in range(1,m):
        if i%s:a.append(0)
        else:
            mag=rng.randint(1<<(b-1),(1<<b)-1)
            sign=(-1 if rng.randrange(2) else 1) if family=="signed_random" else (
                -1 if (i//s)%2 else 1)
            a.append(sign*mag)
    return tuple(a)
def case_core(rep,m,profile,family):
    s,b=PROFILE_PARAMS[profile]
    return {
        "case_schema":CASE_SCHEMA,"replicate":rep,"length":m,
        "profile":profile,"family":family,"stride":s,
        "coefficient_magnitude_bits":b,
        "source_coefficients_decimal":[str(v) for v in source_series(
            rep,m,profile,family)]
    }
def make_fixture():
    old=historical_ids()
    rows=[]
    for rep in REPS:
        for m in LENGTHS:
            for p in PROFILES:
                for fam in FAMILIES:
                    core=case_core(rep,m,p,fam)
                    rows.append({**core,"source_sha256":sha256(canonical(core)).hexdigest()})
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(old)
    return {"schema":SCHEMA,"first_registration_sha":REGISTERED,
            "source_domain":SOURCE_DOMAIN,"replicates":list(REPS),
            "lengths":list(LENGTHS),"profiles":list(PROFILES),
            "families":list(FAMILIES),"public_only":True,
            "no_inverse_root_payload_signature_session_or_timing":True,
            "source_count":60,"excluded_original_source_IDs":408,"cases":rows}
def read_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    assert canonical(x)==raw
    assert x["schema"]==SCHEMA and x["first_registration_sha"]==REGISTERED
    assert x["source_domain"]==SOURCE_DOMAIN
    assert x["replicates"]==list(REPS) and x["lengths"]==list(LENGTHS)
    assert x["profiles"]==list(PROFILES) and x["families"]==list(FAMILIES)
    assert x["source_count"]==60 and x["excluded_original_source_IDs"]==408
    assert x["public_only"] and x["no_inverse_root_payload_signature_session_or_timing"]
    rows=x["cases"]
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(historical_ids())
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert {(r["replicate"],r["length"],r["profile"],r["family"]) for r in rows}=={
        (rep,m,p,f) for rep in REPS for m in LENGTHS for p in PROFILES for f in FAMILIES}
    for r in rows:
        assert set(r)=={"case_schema","replicate","length","profile","family","stride",
                        "coefficient_magnitude_bits","source_coefficients_decimal",
                        "source_sha256"}
        core={k:v for k,v in r.items() if k!="source_sha256"}
        assert core==case_core(r["replicate"],r["length"],r["profile"],r["family"])
        assert sha256(canonical(core)).hexdigest()==r["source_sha256"]
        vals=r["source_coefficients_decimal"]
        assert len(vals)==r["length"] and vals[0]=="1"
        assert all(type(v) is str and str(int(v))==v for v in vals)
    return x,sha256(raw).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-only-out",type=Path,required=True)
    a=p.parse_args()
    raw=canonical(make_fixture())
    a.public_only_out.parent.mkdir(parents=True,exist_ok=True)
    a.public_only_out.write_bytes(raw)
    _,h=read_fixture(a.public_only_out)
    print("NOR57B FIRST ORIGINAL 60 PUBLIC-ONLY RAW BYTES SHA256",len(raw),h,flush=True)
if __name__=="__main__":main()
