"""NOR-29 first stage: 24 public-only cases and 768 salted blind commitments.

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

OFFSET=139329000
REPS=(12,13,14,15)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
N_SOURCES=24
REPLICAS=32
RADIUS=4
TRIALS=512
ARMS=("hybrid_50","uniform16","static_prior")
KINDS=("blind_uniform","early_root_aware")
FIXTURE_SCHEMA="nor29.public_only_24_fresh_sources.v1"
EARLY_SCHEMA="nor29.original_768_salted_commitments.v1"
EARLY_MANIFEST="nor29.public_768_precommitments.v1"
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
        raise ValueError("source outside NOR29 frozen registration")
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
        case=corpus.Case("nor29-public",family,demos,query,(),None)
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
    raise RuntimeError("NOR29 exhausted first 100 admissible sources")


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


def commit(opening):
    return sha256(b"NOR29-attack-commit-v1\x00"+canonical(opening)).hexdigest()


def blind_set(digest,m,replica):
    assert 0<=replica<REPLICAS
    rng=random.Random(hash_int(202611290101,digest,replica))
    return tuple(sorted(rng.sample(range(m),RADIUS)))


def rank_order(pred,header,digest):
    return n24.static_schedule(n24.supports(pred,header),n25.PRIOR,digest,0)


def prepare(fixture_path,sealed_path,manifest_path):
    fixture,fixture_sha=load_fixture(fixture_path)
    proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
               for r in fixture["cases"]}
    assert len(proposals)==N_SOURCES
    openings=[]
    entries=[]
    for source in fixture["cases"]:
        digest=source["full_public_input_sha256"]
        m=len(proposals[digest]["predicted"])
        for rep in range(REPLICAS):
            positions=blind_set(digest,m,rep)
            preimage={"source_sha256":digest,
                      "attack_class":"blind_uniform","replica":rep,
                      "changed_indices":list(positions),
                      "nonce_hex":secrets.token_bytes(32).hex()}
            openings.append(preimage)
            entries.append({"source_sha256":digest,"attack_class":"blind_uniform",
                            "replica":rep,"commitment_sha256":commit(preimage)})
    assert len(openings)==len(entries)==N_SOURCES*REPLICAS==768
    sealed={"schema":EARLY_SCHEMA,"fixture_sha256":fixture_sha,"openings":openings}
    sealed_raw=canonical(sealed)
    public={"schema":EARLY_MANIFEST,"fixture_sha256":fixture_sha,
            "sealed_sha256":sha256(sealed_raw).hexdigest(),
            "sources":N_SOURCES,"attack_count":768,
            "challenge_entropy_absent":True,"commitments":entries}
    for path,raw in ((sealed_path,sealed_raw),(manifest_path,canonical(public))):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
    verify_openings(fixture_path,sealed_path,manifest_path)
    print("NOR29 PREP LOCKED 768 BLIND ATTACKS",
          "fixture",fixture_sha,"sealed",public["sealed_sha256"],
          "manifest",sha256(canonical(public)).hexdigest())


def verify_manifest(fixture_path,manifest_path):
    fixture,fhash=load_fixture(fixture_path)
    raw=Path(manifest_path).read_bytes()
    obj=json.loads(raw)
    assert obj["schema"]==EARLY_MANIFEST and obj["fixture_sha256"]==fhash
    assert obj["sources"]==N_SOURCES and obj["attack_count"]==768
    assert obj["challenge_entropy_absent"] is True
    entries=obj["commitments"]
    assert len(entries)==N_SOURCES*REPLICAS
    expected={(r["full_public_input_sha256"],"blind_uniform",j)
              for r in fixture["cases"] for j in range(REPLICAS)}
    actual={(x["source_sha256"],x["attack_class"],x["replica"])
            for x in entries}
    assert expected==actual and len({x["commitment_sha256"] for x in entries})==len(entries)
    return obj,sha256(raw).hexdigest()


def verify_openings(fixture_path,sealed_path,manifest_path):
    fixture,_=load_fixture(fixture_path)
    manifest,mhash=verify_manifest(fixture_path,manifest_path)
    raw=Path(sealed_path).read_bytes()
    assert sha256(raw).hexdigest()==manifest["sealed_sha256"]
    sealed=json.loads(raw)
    assert sealed["schema"]==EARLY_SCHEMA
    assert sealed["fixture_sha256"]==manifest["fixture_sha256"]
    expected={(e["source_sha256"],e["attack_class"],e["replica"]):
              e["commitment_sha256"] for e in manifest["commitments"]}
    sizes={x["full_public_input_sha256"]:x["query_degree"]+1 for x in fixture["cases"]}
    seen=set()
    for opening in sealed["openings"]:
        key=(opening["source_sha256"],opening["attack_class"],opening["replica"])
        assert key in expected and key not in seen
        seen.add(key)
        assert set(opening)=={"source_sha256","attack_class","replica",
                             "changed_indices","nonce_hex"}
        positions=opening["changed_indices"]
        assert len(positions)==len(set(positions))==RADIUS
        assert positions==sorted(positions)
        assert all(type(i) is int and 0<=i<sizes[key[0]] for i in positions)
        assert len(opening["nonce_hex"])==64
        assert all(c in "0123456789abcdef" for c in opening["nonce_hex"])
        assert commit(opening)==expected[key]
        tampered=dict(opening)
        tampered["nonce_hex"]=("0" if opening["nonce_hex"][0]!="0"
                               else "1")+opening["nonce_hex"][1:]
        assert commit(tampered)!=expected[key]
        # Also check a single altered coordinate whenever an alternative exists.
        alt=list(positions)
        alt[-1]=next(i for i in range(sizes[key[0]]) if i not in alt)
        alt.sort()
        tampered_positions=dict(opening,changed_indices=alt)
        assert commit(tampered_positions)!=expected[key]
    assert len(seen)==len(sealed["openings"])==768
    return sealed,mhash


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--verify",action="store_true")
    p.add_argument("--fixture",type=Path)
    p.add_argument("--sealed",type=Path)
    p.add_argument("--manifest",type=Path)
    a=p.parse_args()
    if a.make_fixture:
        if a.prepare or a.verify or a.fixture or a.sealed or a.manifest:
            p.error("fixture creation must be an independent first step")
        raw=canonical(make_fixture())
        a.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        a.make_fixture.write_bytes(raw)
        print("NOR29 NEW PUBLIC SOURCES",len(raw),sha256(raw).hexdigest())
        return
    if not (a.fixture and a.sealed and a.manifest):
        p.error("fixture, sealed, manifest required")
    if a.prepare and not a.verify:
        prepare(a.fixture,a.sealed,a.manifest)
    elif a.verify and not a.prepare:
        verify_openings(a.fixture,a.sealed,a.manifest)
        print("NOR29 ORIGINAL BLIND COMMITMENTS VERIFIED 768")
    else:
        p.error("choose --prepare OR --verify")


if __name__=="__main__":main()
