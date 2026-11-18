"""NOR54: first public-only SHA-distinct source cohort, without any inverse.

Registration is frozen before this source generator or CPU benchmark runs.
The upstream Actions job uploads these original public bytes before timing.
"""
from __future__ import annotations
import argparse,json,random,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

REGISTERED="10e7ac3c3fdefec89d90d7a98a947099db8af86e"
SCHEMA="nor54.first_public_only_60_matched_exact_series.v1"
CASE_SCHEMA="nor54.exact_public_source_coordinates.v1"
SOURCE_DOMAIN="NOR54/SOURCE/v1"
REPS=(0,1)
LENGTHS=(769,1537,3073)
PROFILES=("s1-b2","s4-b2","s8-b2","s4-b8","s4-b16")
FAMILIES=("signed_random","alternating")
PROFILE_PARAMS={"s1-b2":(1,2),"s4-b2":(4,2),"s8-b2":(8,2),
                "s4-b8":(4,8),"s4-b16":(4,16)}
ORIGINAL_ARCHIVES=(
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
   "b7e65cdacefced6fcf1084922fdaeaaa4da5af1cf9066681471e4be1cd99ad61",60)
)

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def first_previous_ids():
    seen=set()
    for path,zip_sha,member,raw_sha,n in ORIGINAL_ARCHIVES:
        filename=ROOT/path
        assert sha256(filename.read_bytes()).hexdigest()==zip_sha
        with ZipFile(filename) as z:
            assert z.testzip() is None
            raw=z.read(member)
        assert sha256(raw).hexdigest()==raw_sha
        cases=json.loads(raw)["cases"]
        ids={r["source_sha256"] for r in cases}
        assert len(ids)==n and seen.isdisjoint(ids)
        seen.update(ids)
    assert len(seen)==168
    return seen

def source_seed(rep,m,profile,family):
    assert rep in REPS and m in LENGTHS
    assert profile in PROFILES and family in FAMILIES
    key=f"{SOURCE_DOMAIN}|{rep}|{m}|{profile}|{family}"
    return int.from_bytes(sha256(key.encode("ascii")).digest(),"big")

def source_series(rep,m,profile,family):
    stride,bits=PROFILE_PARAMS[profile]
    rng=random.Random(source_seed(rep,m,profile,family))
    vals=[1]
    for i in range(1,m):
        if i%stride:
            vals.append(0)
        else:
            mag=rng.randint(1<<(bits-1),(1<<bits)-1)
            sign=(-1 if rng.randrange(2) else 1) if family=="signed_random" else (
                -1 if (i//stride)%2 else 1)
            vals.append(sign*mag)
    return tuple(vals)

def case_core(rep,m,profile,family):
    stride,bits=PROFILE_PARAMS[profile]
    return {
        "source_schema":CASE_SCHEMA,"replicate":rep,
        "length":m,"profile":profile,"family":family,
        "stride":stride,"coefficient_magnitude_bits":bits,
        "source_coefficients_decimal":[str(v) for v in source_series(rep,m,profile,family)]
    }

def make_fixture():
    excluded=first_previous_ids()
    rows=[]
    for rep in REPS:
        for m in LENGTHS:
            for profile in PROFILES:
                for family in FAMILIES:
                    core=case_core(rep,m,profile,family)
                    rows.append({**core,"source_sha256":sha256(canonical(core)).hexdigest()})
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(excluded)
    return {
        "schema":SCHEMA,"first_registration_sha":REGISTERED,
        "source_domain":SOURCE_DOMAIN,
        "replicates":list(REPS),"lengths":list(LENGTHS),
        "profiles":list(PROFILES),"families":list(FAMILIES),
        "source_count":60,"original_excluded_source_count":168,
        "public_only":True,"no_inverse_witness_root_or_timing":True,
        "cases":rows
    }

def read_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    assert raw==canonical(x)
    assert x["schema"]==SCHEMA and x["first_registration_sha"]==REGISTERED
    assert x["source_domain"]==SOURCE_DOMAIN
    assert x["replicates"]==list(REPS) and x["lengths"]==list(LENGTHS)
    assert x["profiles"]==list(PROFILES) and x["families"]==list(FAMILIES)
    assert x["source_count"]==60 and x["original_excluded_source_count"]==168
    assert x["public_only"] and x["no_inverse_witness_root_or_timing"]
    rows=x["cases"]
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(first_previous_ids())
    assert Counter(r["length"] for r in rows)=={m:20 for m in LENGTHS}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert {(r["replicate"],r["length"],r["profile"],r["family"])
             for r in rows}=={
        (rep,m,p,fam) for rep in REPS for m in LENGTHS
        for p in PROFILES for fam in FAMILIES}
    for r in rows:
        assert set(r)=={
            "source_schema","replicate","length","profile","family",
            "stride","coefficient_magnitude_bits",
            "source_coefficients_decimal","source_sha256"}
        core={k:v for k,v in r.items() if k!="source_sha256"}
        assert core==case_core(r["replicate"],r["length"],r["profile"],r["family"])
        assert sha256(canonical(core)).hexdigest()==r["source_sha256"]
        coeffs=r["source_coefficients_decimal"]
        assert len(coeffs)==r["length"] and coeffs[0]=="1"
        assert all(type(v) is str and str(int(v))==v for v in coeffs)
    return x,sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-only-out",type=Path,required=True)
    a=p.parse_args()
    raw=canonical(make_fixture())
    a.public_only_out.parent.mkdir(parents=True,exist_ok=True)
    a.public_only_out.write_bytes(raw)
    _,sha=read_fixture(a.public_only_out)
    print("NOR54 FIRST ORIGINAL 60 PUBLIC-ONLY SOURCE BYTES SHA256",len(raw),sha,flush=True)

if __name__=="__main__":
    main()
