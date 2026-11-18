"""NOR53 prospective public-only 60-source generator, no inverse/timing.

New NOR53/SOURCE/v1 identities exclude first original NOR51 and NOR52 cases,
read from immutable source-controlled archives and verified with their SHAs.
"""
from __future__ import annotations
import argparse,json,random,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
REGISTERED="ae56f414251a8485d127fc8f9899a24237870adf"
SCHEMA="nor53.first_60_public_only_native_gmp_sources.v1"
CASE_SCHEMA="nor53.public_exact_series_source.v1"
SOURCE_DOMAIN="NOR53/SOURCE/v1"
REPS=(0,1)
LENGTHS=(769,1537,3073)
PROFILES=("s1-b2","s4-b2","s8-b2","s4-b8","s4-b16")
FAMILIES=("signed_random","alternating")
PROFILE_PARAMS={"s1-b2":(1,2),"s4-b2":(4,2),
                "s8-b2":(8,2),"s4-b8":(4,8),"s4-b16":(4,16)}
PRIOR_ARCHIVES=(
  ("research/nor51_exact_r_crossover/results/nor51_first_original_hosted.zip",
   "b8771c20b24b7868ff467f9559cba6c551021042943db8d308c204d0d75e39d9",
   "nor51_first_48_source_only.json",
   "0b0de5549de1e1815a40d56302cb01ca104e7ac8141f161bc8cab0b6501d2c93",48),
  ("research/nor52_sparse_inverse/results/nor52_first_original_hosted.zip",
   "1905b3835ed6dea870994a154d8a56942fce0b0fcf83d4058417abc543e206a6",
   "nor52_first_public_only_60_sources.json",
   "d18b795053d81c16fc5316b490316e90fe6b373156a8ee8b295008a13df150f4",60)
)

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def original_source_ids():
    previous=set()
    for path,archive_sha,member,source_sha,count in PRIOR_ARCHIVES:
        archive=(ROOT/path).read_bytes()
        assert sha256(archive).hexdigest()==archive_sha
        with ZipFile(ROOT/path) as z:
            raw=z.read(member)
        assert sha256(raw).hexdigest()==source_sha
        rows=json.loads(raw)["cases"]
        ids={r["source_sha256"] for r in rows}
        assert len(ids)==count and previous.isdisjoint(ids)
        previous.update(ids)
    assert len(previous)==108
    return previous

def source_seed(rep,m,profile,family):
    assert rep in REPS and m in LENGTHS and profile in PROFILES and family in FAMILIES
    raw=f"{SOURCE_DOMAIN}|{rep}|{m}|{profile}|{family}"
    return int.from_bytes(sha256(raw.encode("ascii")).digest(),"big")

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

def case_core(rep,m,profile,family):
    stride,bits=PROFILE_PARAMS[profile]
    return {
        "case_schema":CASE_SCHEMA,
        "replicate":rep,"length":m,"profile":profile,"family":family,
        "stride":stride,"coefficient_magnitude_bits":bits,
        "source_coefficients_decimal":[str(v) for v in source_series(rep,m,profile,family)]
    }

def make_fixture():
    rows=[]
    for rep in REPS:
        for m in LENGTHS:
            for profile in PROFILES:
                for family in FAMILIES:
                    core=case_core(rep,m,profile,family)
                    rows.append({**core,"source_sha256":sha256(canonical(core)).hexdigest()})
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(original_source_ids())
    return {
        "schema":SCHEMA,"first_registration_sha":REGISTERED,
        "source_domain":SOURCE_DOMAIN,
        "replicates":list(REPS),"lengths":list(LENGTHS),
        "profiles":list(PROFILES),"families":list(FAMILIES),
        "public_only":True,"inverse_witness_root_and_cpu_absent":True,
        "source_count":60,"previous_excluded_source_identities":108,
        "cases":rows
    }

def read_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    assert x["schema"]==SCHEMA and x["first_registration_sha"]==REGISTERED
    assert x["source_domain"]==SOURCE_DOMAIN
    assert x["replicates"]==list(REPS) and x["lengths"]==list(LENGTHS)
    assert x["profiles"]==list(PROFILES) and x["families"]==list(FAMILIES)
    assert x["source_count"]==60 and x["previous_excluded_source_identities"]==108
    assert x["public_only"] and x["inverse_witness_root_and_cpu_absent"]
    rows=x["cases"]
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(original_source_ids())
    assert Counter(r["length"] for r in rows)=={m:20 for m in LENGTHS}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert {(r["replicate"],r["length"],r["profile"],r["family"]) for r in rows}=={
      (rep,m,p,fam) for rep in REPS for m in LENGTHS for p in PROFILES for fam in FAMILIES}
    for row in rows:
        assert set(row)=={"case_schema","replicate","length","profile","family",
                         "stride","coefficient_magnitude_bits",
                         "source_coefficients_decimal","source_sha256"}
        core={k:v for k,v in row.items() if k!="source_sha256"}
        assert core==case_core(row["replicate"],row["length"],row["profile"],row["family"])
        assert sha256(canonical(core)).hexdigest()==row["source_sha256"]
        assert row["source_coefficients_decimal"][0]=="1"
        assert all(type(v) is str and str(int(v))==v
                   for v in row["source_coefficients_decimal"])
    assert canonical(x)==raw
    return x,sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-only-out",required=True,type=Path)
    args=p.parse_args()
    raw=canonical(make_fixture())
    args.public_only_out.parent.mkdir(parents=True,exist_ok=True)
    args.public_only_out.write_bytes(raw)
    _,h=read_fixture(args.public_only_out)
    print("NOR53 FIRST SOURCE-ONLY RAW BYTES SHA256",len(raw),h,flush=True)

if __name__=="__main__":main()
