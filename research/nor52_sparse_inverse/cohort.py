"""NOR52 first 60 public-only sparse/dense source identities.

No target, inverse, trace, or benchmark timing is generated here.
Source-only fixture is SHA-frozen in an upstream CI job.
"""
from __future__ import annotations
import argparse,json,random,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

REGISTERED="9526627e21b429e7fea1928b3ddfb0e1dadc5e74"
SCHEMA="nor52.first_public_only_60_sparse_dense_sources.v1"
CASE_SCHEMA="nor52.source_id_from_public_coefficients.v1"
SOURCE_DOMAIN="NOR52/SOURCE/v1"
REPS=(0,1)
LENGTHS=(769,1537,3073)
PROFILES=("s1-b2","s4-b2","s8-b2","s4-b8","s4-b16")
FAMILIES=("signed_random","alternating")
PROFILE_PARAMS={
    "s1-b2":(1,2),"s4-b2":(4,2),"s8-b2":(8,2),
    "s4-b8":(4,8),"s4-b16":(4,16)
}
PRIOR_NOR51_PUBLIC_SHA="0b0de5549de1e1815a40d56302cb01ca104e7ac8141f161bc8cab0b6501d2c93"
PRIOR_NOR51_ARCHIVE_SHA="b8771c20b24b7868ff467f9559cba6c551021042943db8d308c204d0d75e39d9"

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def first_prior_source_ids():
    zip_path=ROOT/"research/nor51_exact_r_crossover/results/nor51_first_original_hosted.zip"
    assert sha256(zip_path.read_bytes()).hexdigest()==PRIOR_NOR51_ARCHIVE_SHA
    with ZipFile(zip_path) as z:
        raw=z.read("nor51_first_48_source_only.json")
    assert sha256(raw).hexdigest()==PRIOR_NOR51_PUBLIC_SHA
    rows=json.loads(raw)["cases"]
    ids={r["source_sha256"] for r in rows}
    assert len(ids)==48
    return ids

def seed(rep,m,profile,family):
    assert rep in REPS and m in LENGTHS
    assert profile in PROFILES and family in FAMILIES
    token=f"{SOURCE_DOMAIN}|{rep}|{m}|{profile}|{family}"
    return int.from_bytes(sha256(token.encode("ascii")).digest(),"big")

def source_series(rep,m,profile,family):
    stride,bits=PROFILE_PARAMS[profile]
    rng=random.Random(seed(rep,m,profile,family))
    out=[1]
    for i in range(1,m):
        if i%stride:
            out.append(0)
        else:
            magnitude=rng.randint(1<<(bits-1),(1<<bits)-1)
            if family=="signed_random":
                sign=-1 if rng.randrange(2) else 1
            else:
                sign=-1 if (i//stride)%2 else 1
            out.append(sign*magnitude)
    return tuple(out)

def source_core(rep,m,profile,family):
    stride,bits=PROFILE_PARAMS[profile]
    return {
        "case_schema":CASE_SCHEMA,
        "replicate":rep,"length":m,"profile":profile,"family":family,
        "stride":stride,"coefficient_magnitude_bits":bits,
        "source_coefficients_decimal":[str(v) for v in source_series(
            rep,m,profile,family)]
    }

def make_fixture():
    cases=[]
    for rep in REPS:
        for m in LENGTHS:
            for profile in PROFILES:
                for family in FAMILIES:
                    core=source_core(rep,m,profile,family)
                    cases.append({**core,"source_sha256":sha256(canonical(core)).hexdigest()})
    ids={row["source_sha256"] for row in cases}
    assert len(cases)==len(ids)==60
    assert ids.isdisjoint(first_prior_source_ids())
    return {
        "schema":SCHEMA,"first_registration_sha":REGISTERED,
        "source_domain":SOURCE_DOMAIN,
        "replicates":list(REPS),"lengths":list(LENGTHS),
        "profiles":list(PROFILES),"families":list(FAMILIES),
        "source_count":60,
        "public_only":True,"no_inverse_trace_root_or_timing":True,
        "prior_NOR51_first_public_source_sha256":PRIOR_NOR51_PUBLIC_SHA,
        "cases":cases
    }

def read_fixture(path):
    raw=Path(path).read_bytes()
    fixture=json.loads(raw)
    assert fixture["schema"]==SCHEMA
    assert fixture["first_registration_sha"]==REGISTERED
    assert fixture["source_domain"]==SOURCE_DOMAIN
    assert fixture["replicates"]==list(REPS) and fixture["lengths"]==list(LENGTHS)
    assert fixture["profiles"]==list(PROFILES) and fixture["families"]==list(FAMILIES)
    assert fixture["source_count"]==60 and fixture["public_only"]
    assert fixture["no_inverse_trace_root_or_timing"]
    assert fixture["prior_NOR51_first_public_source_sha256"]==PRIOR_NOR51_PUBLIC_SHA
    cases=fixture["cases"]
    ids={c["source_sha256"] for c in cases}
    assert len(cases)==len(ids)==60
    assert ids.isdisjoint(first_prior_source_ids())
    assert Counter(c["length"] for c in cases)=={n:20 for n in LENGTHS}
    assert Counter(c["stride"] for c in cases)=={1:12,4:36,8:12}
    assert {(c["replicate"],c["length"],c["profile"],c["family"])
            for c in cases}=={
                (rep,m,p,fam) for rep in REPS for m in LENGTHS
                for p in PROFILES for fam in FAMILIES}
    for case in cases:
        assert set(case)=={
            "case_schema","replicate","length","profile","family","stride",
            "coefficient_magnitude_bits","source_coefficients_decimal","source_sha256"}
        core={k:v for k,v in case.items() if k!="source_sha256"}
        assert core==source_core(case["replicate"],case["length"],case["profile"],case["family"])
        assert sha256(canonical(core)).hexdigest()==case["source_sha256"]
        vals=case["source_coefficients_decimal"]
        assert len(vals)==case["length"] and vals[0]=="1"
        assert all(type(v) is str and str(int(v))==v for v in vals)
    assert raw==canonical(fixture)
    return fixture,sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-only-out",required=True,type=Path)
    args=p.parse_args()
    raw=canonical(make_fixture())
    args.public_only_out.parent.mkdir(parents=True,exist_ok=True)
    args.public_only_out.write_bytes(raw)
    _,h=read_fixture(args.public_only_out)
    print("NOR52 FIRST SOURCE ONLY RAW BYTES SHA256",len(raw),h,flush=True)

if __name__=="__main__":main()
