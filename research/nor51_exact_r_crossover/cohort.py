"""NOR51 first public-only exact-series source cohort, without any inverse.

Do not import the source-inverse producer or run CPU benchmarks from this file.
The first complete fixture is SHA-frozen in an upstream CI job.
"""
from __future__ import annotations
import argparse,json,random,sys
from hashlib import sha256
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

REGISTERED="83fabe1c34c2bedbba35faa928349f41b964115a"
SCHEMA="nor51.first_public_only_48_exact_reciprocal_sources.v1"
REPS=(0,1)
LENGTHS=(769,1537,3073)
PROFILES=("s4-b2","s4-b8","s4-b16","s1-b2")
FAMILIES=("signed_random","alternating")
PROFILE_PARAMETERS={"s4-b2":(4,2),"s4-b8":(4,8),
                    "s4-b16":(4,16),"s1-b2":(1,2)}
SOURCE_DOMAIN="NOR51/SOURCE/v1"
CASE_SCHEMA="nor51.source_id_from_public_coefficients.v1"

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def input_seed(rep,m,profile,family):
    assert rep in REPS and m in LENGTHS
    assert profile in PROFILES and family in FAMILIES
    token=f"{SOURCE_DOMAIN}|{rep}|{m}|{profile}|{family}"
    return int.from_bytes(sha256(token.encode("ascii")).digest(),"big")

def series(rep,m,profile,family):
    stride,bits=PROFILE_PARAMETERS[profile]
    rng=random.Random(input_seed(rep,m,profile,family))
    source=[1]
    for i in range(1,m):
        if i%stride:
            source.append(0)
        else:
            magnitude=rng.randint(1<<(bits-1),(1<<bits)-1)
            sign=(-1 if rng.randrange(2) else 1) if family=="signed_random" else (
                -1 if (i//stride)%2 else 1)
            source.append(sign*magnitude)
    return tuple(source)

def source_core(rep,m,profile,family):
    stride,bits=PROFILE_PARAMETERS[profile]
    f=series(rep,m,profile,family)
    return {
        "source_schema":CASE_SCHEMA,"replicate":rep,"length":m,
        "profile":profile,"family":family,"stride":stride,
        "coefficient_magnitude_bits":bits,
        "source_coefficients_decimal":[str(v) for v in f]
    }

def make_fixture():
    rows=[]
    for rep in REPS:
        for m in LENGTHS:
            for profile in PROFILES:
                for family in FAMILIES:
                    core=source_core(rep,m,profile,family)
                    row={**core,"source_sha256":sha256(canonical(core)).hexdigest()}
                    rows.append(row)
    ids=[r["source_sha256"] for r in rows]
    assert len(rows)==len(set(ids))==48
    return {
        "schema":SCHEMA,"registration_first_commit":REGISTERED,
        "source_domain":SOURCE_DOMAIN,
        "replicates":list(REPS),"lengths":list(LENGTHS),
        "profiles":list(PROFILES),"sign_families":list(FAMILIES),
        "public_only":True,"no_inverse_root_trace_or_timings":True,
        "source_count":48,
        "cases":rows
    }

def validate_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    assert x["schema"]==SCHEMA and x["registration_first_commit"]==REGISTERED
    assert x["source_domain"]==SOURCE_DOMAIN
    assert x["replicates"]==list(REPS) and x["lengths"]==list(LENGTHS)
    assert x["profiles"]==list(PROFILES) and x["sign_families"]==list(FAMILIES)
    assert x["source_count"]==48 and x["public_only"] and x["no_inverse_root_trace_or_timings"]
    rows=x["cases"]
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==48
    assert Counter(r["length"] for r in rows)=={m:16 for m in LENGTHS}
    assert {(r["replicate"],r["length"],r["profile"],r["family"]) for r in rows}=={
        (rep,m,profile,family)
        for rep in REPS for m in LENGTHS
        for profile in PROFILES for family in FAMILIES}
    for row in rows:
        assert set(row)=={
            "source_schema","replicate","length","profile","family",
            "stride","coefficient_magnitude_bits",
            "source_coefficients_decimal","source_sha256"}
        core={k:v for k,v in row.items() if k!="source_sha256"}
        assert sha256(canonical(core)).hexdigest()==row["source_sha256"]
        assert core==source_core(row["replicate"],row["length"],row["profile"],row["family"])
        f=core["source_coefficients_decimal"]
        assert len(f)==row["length"] and f[0]=="1"
        assert all(type(v) is str and str(int(v))==v for v in f)
    assert canonical(x)==raw
    return x,sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-only-out",required=True,type=Path)
    args=p.parse_args()
    raw=canonical(make_fixture())
    args.public_only_out.parent.mkdir(parents=True,exist_ok=True)
    args.public_only_out.write_bytes(raw)
    x,h=validate_fixture(args.public_only_out)
    print("NOR51 FIRST SOURCE-ONLY BYTES SHA256",len(raw),h,flush=True)
    print("NOR51 GRID COUNTS",len(x["cases"]),Counter(r["length"] for r in x["cases"]))

if __name__=="__main__":main()
