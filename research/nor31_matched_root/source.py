"""NOR-31 first stage: 24 public-only cases and 768 salted blind commitments.

Registration first commit 2447248. This stage never obtains a challenge root.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import argparse
import json
import random
import secrets
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

OFFSET=150031000
REPS=(22,23,24,25)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
N_SOURCES=24
REPLICAS=32
RADIUS=4
TRIALS=512
ARMS=("hybrid_50","uniform16","static_prior")
KINDS=("independent_calibration","leaked_evaluation_root")
FIXTURE_SCHEMA="nor31.public_only_24_fresh_sources.v1"
EARLY_SCHEMA="nor31.original_768_salted_commitments.v1"
EARLY_MANIFEST="nor31.public_768_precommitments.v1"
PRIOR_NUMERATORS=(295,246,153,78)
PRIOR_DENOMINATOR=772
assert n25.PRIOR_NUMERATORS==PRIOR_NUMERATORS
assert n25.PRIOR_DENOMINATOR==PRIOR_DENOMINATOR


def canonical(obj):
    return n24.canonical(obj)


def hash_int(*fields):
    return n24.stream_seed(*fields)


def source_seed(rep,degree,family):
    if rep not in REPS or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("source outside NOR31 frozen registration")
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(degree)+10007*FAMILIES.index(family))


def generate_source(rep,degree,family,seen):
    q=3*degree//2
    rng=random.Random(source_seed(rep,degree,family))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,family,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        case=corpus.Case("nor31-public",family,demos,query,(),None)
        headers=corpus.possible_headers(case)
        if len(headers)!=1:continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        digest=nor21.public_digest(case,degree)
        if digest in seen:continue
        mask=nor21.native_mask(case,h,degree)
        if mask==0:continue
        predictions=nor21.candidate_predictions(query,h,q)
        if len({predictions[j] for j in range(16) if mask&(1<<j)})!=1:continue
        planted_j=corpus.OPS.index(planted[4])*4+corpus.OPS.index(planted[5])
        assert mask&(1<<planted_j)
        seen.add(digest)
        return {
            "split":"TEST","replicate":rep,"observation_degree":degree,
            "query_degree":q,"family":family,"source_seed":source_seed(rep,degree,family),
            "selected_attempt":attempt,"full_public_input_sha256":digest,
            "public_header":{"base":h.base,"delay":h.delay,
                             "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(x[:degree+1]),list(y[:degree+1])] for x,y in demos],
            "public_query_input":list(query),
            "original_public_accept_mask":mask,
        }
    raise RuntimeError("NOR31 exhausted first 100 admissible public sources")


def make_fixture():
    seen=set()
    rows=[generate_source(rep,d,f,seen) for rep in REPS for d in DEGREES
          for f in FAMILIES]
    assert len(rows)==len(seen)==N_SOURCES
    return {"schema":FIXTURE_SCHEMA,"source_offset":OFFSET,
            "replicates":list(REPS),"source_only":True,
            "hidden_truth_absent":True,"attack_sets_absent":True,
            "challenge_entropy_absent":True,"blind_attacks_per_source":REPLICAS,
            "cases":rows}


def load_fixture(path):
    raw=Path(path).read_bytes()
    obj=json.loads(raw)
    assert obj["schema"]==FIXTURE_SCHEMA
    assert obj["source_offset"]==OFFSET
    assert obj["replicates"]==list(REPS)
    assert obj["source_only"] and obj["hidden_truth_absent"]
    assert obj["attack_sets_absent"] and obj["challenge_entropy_absent"]
    assert obj["blind_attacks_per_source"]==REPLICAS
    assert len(obj["cases"])==N_SOURCES
    assert len({x["full_public_input_sha256"] for x in obj["cases"]})==N_SOURCES
    assert {(x["replicate"],x["observation_degree"],x["family"]) for x in obj["cases"]}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    assert all(x["split"]=="TEST" and not any(
        bad in x for bad in ("program","future","nonce","attack","root","witness"))
        for x in obj["cases"])
    return obj,sha256(raw).hexdigest()



def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--validate",type=Path)
    a=p.parse_args()
    if bool(a.make_fixture)==bool(a.validate):
        p.error("choose exactly one of --make-fixture/--validate")
    if a.make_fixture:
        raw=canonical(make_fixture())
        a.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        a.make_fixture.write_bytes(raw)
        print("NOR31 ORIGINAL SOURCE-ONLY FIXTURE",len(raw),sha256(raw).hexdigest())
    else:
        f,d=load_fixture(a.validate)
        print("NOR31 PUBLIC SOURCE-ONLY PARITY",len(f["cases"]),d)


if __name__=="__main__":main()
