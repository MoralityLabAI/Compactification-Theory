"""NOR59 PUBLIC ONLY source/program roster; no output, root or timing."""
from __future__ import annotations
import argparse,json,random,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
REGISTERED="8f6eb3dccc4d4c4d9d58955387ea6f154ad05d84"
SCHEMA="nor59.first_public_only_60_signed_authorized_program_sources.v1"
CASE_SCHEMA="nor59.public_exact_signed_NOR45_source.v1"
DOMAIN="NOR59/SOURCE/v1"
DEGREES=(512,1024)
STRIDES=(1,4,8)
PROFILES=(0,1,2,3,4)
REPLICATES=(0,1)
PROFILE_BITS=(2,8,16,2,8)
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
  "3eed34d3b80fbae3111d0fcad80df0865df3be78f0077460ac9349647616abe2",60),
 ("research/nor57b_independent/results/nor57b_first_original_hosted.zip",
  "ec8476d8fae5831a2338e779b4d1e0c1fd2175dc341cb5478975c4f6d3a79e1a",
  "nor57b_first_original_60_public_only.json",
  "bca8cddca58a4c8a234df4f6bba2b31c44c7e114c2f07f67a545638dad5b9a63",60)
,
 ("research/nor58_static_merkle/results/nor58_first_original_hosted.zip",
  "5b0c2b18004cb6f04e29283a8b4dd4342e43705f4c65c09df80df767950d218d",
  "nor58_first_original_60_public_only.json",
  "6b34e964bc170f3e8efcdfae23403fead030992cf1fc08640ed13c2e34c6b5b3",60)
)
def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")
def historical_ids():
    ids=set()
    for path,zsha,member,rawsha,count in PRIOR_ARCHIVES:
        p=ROOT/path
        assert sha256(p.read_bytes()).hexdigest()==zsha
        with ZipFile(p) as z:
            assert z.testzip() is None
            raw=z.read(member)
        assert sha256(raw).hexdigest()==rawsha
        rows=json.loads(raw)["cases"]
        new={r["source_sha256"] for r in rows}
        assert len(new)==count and ids.isdisjoint(new)
        ids.update(new)
    assert len(ids)==528
    return ids
def program(profile):
    profiles=(
        dict(base="I",u="R",v="N",delay=64,spacing=24,amplitude=1,hidden_beta=0,hidden_transform=None),
        dict(base="N",u="S",v="I",delay=80,spacing=32,amplitude=-2,hidden_beta=1,hidden_transform="R"),
        dict(base="R",u="I",v="S",delay=96,spacing=24,amplitude=2,hidden_beta=-1,hidden_transform="S"),
        dict(base="S",u="N",v="R",delay=112,spacing=32,amplitude=-1,hidden_beta=0,hidden_transform=None),
        dict(base="R",u="R",v="R",delay=80,spacing=24,amplitude=1,hidden_beta=2,hidden_transform="N")
    )
    if type(profile) is not int or profile not in PROFILES:raise ValueError("unregistered NOR59 program")
    return profiles[profile].copy()
def source_seed(degree,stride,profile,replicate):
    assert degree in DEGREES and stride in STRIDES and profile in PROFILES and replicate in REPLICATES
    return int.from_bytes(sha256(f"{DOMAIN}|{degree}|{stride}|{profile}|{replicate}".encode("ascii")).digest(),"big")
def source_series(degree,stride,profile,replicate):
    rng=random.Random(source_seed(degree,stride,profile,replicate))
    m=3*degree//2+1
    bits=PROFILE_BITS[profile]
    vals=[1]
    for i in range(1,m):
        if i%stride:vals.append(0)
        else:
            mag=rng.randint(1<<(bits-1),(1<<bits)-1)
            sign=(-1 if rng.randrange(2) else 1) if replicate==0 else (-1 if (i//stride)%2 else 1)
            vals.append(sign*mag)
    return tuple(vals)
def case_core(degree,stride,profile,replicate):
    return {
        "case_schema":CASE_SCHEMA,
        "observed_degree":degree,
        "full_length":3*degree//2+1,
        "source_stride":stride,
        "program_profile":profile,
        "coefficient_magnitude_bits":PROFILE_BITS[profile],
        "replicate":replicate,
        "family":"signed_random" if replicate==0 else "alternating",
        "authorized_program":program(profile),
        "public_query_coefficients_decimal":[str(v) for v in source_series(degree,stride,profile,replicate)]
    }
def make_fixture():
    old=historical_ids()
    rows=[]
    for d in DEGREES:
        for s in STRIDES:
            for p in PROFILES:
                for rep in REPLICATES:
                    core=case_core(d,s,p,rep)
                    rows.append({**core,"source_sha256":sha256(canonical(core)).hexdigest()})
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(old)
    return {
        "schema":SCHEMA,"first_registration_sha":REGISTERED,
        "source_domain":DOMAIN,"degrees":list(DEGREES),
        "strides":list(STRIDES),"profiles":list(PROFILES),
        "replicates":list(REPLICATES),"public_only":True,
        "source_count":60,"excluded_first_source_IDs":528,
        "no_output_root_witness_session_signatures_templates_or_timings":True,
        "cases":rows
    }
def read_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    assert canonical(x)==raw
    assert x["schema"]==SCHEMA and x["first_registration_sha"]==REGISTERED
    assert x["source_domain"]==DOMAIN
    assert x["degrees"]==list(DEGREES) and x["strides"]==list(STRIDES)
    assert x["profiles"]==list(PROFILES) and x["replicates"]==list(REPLICATES)
    assert x["public_only"] and x["no_output_root_witness_session_signatures_templates_or_timings"]
    assert x["source_count"]==60 and x["excluded_first_source_IDs"]==528
    rows=x["cases"]
    ids={r["source_sha256"] for r in rows}
    assert len(rows)==len(ids)==60 and ids.isdisjoint(historical_ids())
    assert Counter(r["observed_degree"] for r in rows)=={512:30,1024:30}
    assert Counter(r["source_stride"] for r in rows)=={1:20,4:20,8:20}
    assert Counter(r["program_profile"] for r in rows)=={p:12 for p in PROFILES}
    assert Counter(r["replicate"] for r in rows)=={0:30,1:30}
    assert {(r["observed_degree"],r["source_stride"],r["program_profile"],r["replicate"]) for r in rows}=={(d,s,p,rep) for d in DEGREES for s in STRIDES for p in PROFILES for rep in REPLICATES}
    for row in rows:
        core={k:v for k,v in row.items() if k!="source_sha256"}
        assert set(row)==set(core)|{"source_sha256"}
        assert core==case_core(row["observed_degree"],row["source_stride"],row["program_profile"],row["replicate"])
        assert sha256(canonical(core)).hexdigest()==row["source_sha256"]
        values=row["public_query_coefficients_decimal"]
        assert len(values)==row["full_length"] and values[0]=="1"
        assert all(type(v) is str and str(int(v))==v for v in values)
    return x,sha256(raw).hexdigest()
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--public-only-out",type=Path,required=True)
    args=a.parse_args()
    raw=canonical(make_fixture())
    args.public_only_out.parent.mkdir(parents=True,exist_ok=True)
    args.public_only_out.write_bytes(raw)
    x,h=read_fixture(args.public_only_out)
    print("NOR59 FIRST ORIGINAL 60 PUBLIC-ONLY SOURCE SHA256",len(raw),h,flush=True)
if __name__=="__main__":main()
