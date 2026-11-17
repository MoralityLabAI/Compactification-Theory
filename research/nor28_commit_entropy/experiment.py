"""NOR-28 source-only generation, salted attack commitments, and verification.

The pre-data protocol is frozen at da296c8. This module never mints
challenge entropy; that happens in a later dependent GitHub Actions job.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import argparse
import json
import random
import secrets
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor26_policy_aware_audits import experiment as n26
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

OFFSET=129328000
REPS=(10,11)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
T=8
U=8
R=4
CAL_TRIALS=256
ATTACK_REPS=4
KINDS=("calibrated","fresh_uniform")
SOURCE_SCHEMA="nor28.public_only_12_sources.v1"
SEALED_SCHEMA="nor28.attack_openings.v1"
MANIFEST_SCHEMA="nor28.public_commitments.v1"


def canonical(obj):
    return n24.canonical(obj)


def hash_int(*fields):
    return n24.stream_seed(*fields)


def source_seed(rep,d,f):
    if rep not in REPS or d not in DEGREES or f not in FAMILIES:
        raise ValueError("outside NOR28 source registration")
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(d)+10007*FAMILIES.index(f))


def new_source(rep,d,family,seen):
    q=3*d//2
    rng=random.Random(source_seed(rep,d,family))
    for attempt in range(100):
        program=nor21.new_header_program(rng)
        demos=tuple((inp:=nor21.public_source(rng,family,q),
                     corpus.execute(inp,program)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        case=corpus.Case("nor28-public",family,demos,query,(),None)
        headers=corpus.possible_headers(case)
        if len(headers)!=1: continue
        header=headers[0]
        if (header.base,header.delay,header.spacing,header.amplitude)!=program[:4]:
            continue
        digest=nor21.public_digest(case,d)
        if digest in seen: continue
        mask=nor21.native_mask(case,header,d)
        if mask==0: continue
        predictions=nor21.candidate_predictions(query,header,q)
        if len({predictions[j] for j in range(16) if mask&(1<<j)})!=1:
            continue
        true_j=corpus.OPS.index(program[4])*4+corpus.OPS.index(program[5])
        assert mask&(1<<true_j)
        seen.add(digest)
        return dict(
            split="TEST",replicate=rep,observation_degree=d,query_degree=q,
            family=family,source_seed=source_seed(rep,d,family),
            selected_attempt=attempt,full_public_input_sha256=digest,
            public_header=dict(base=header.base,delay=header.delay,
                               spacing=header.spacing,amplitude=header.amplitude),
            demos=[[list(x[:d+1]),list(y[:d+1])] for x,y in demos],
            public_query_input=list(query),original_public_accept_mask=mask)
    raise RuntimeError("NOR28 public header selection exhausted 100 attempts")


def make_fixture():
    seen=set()
    rows=[new_source(rep,d,f,seen) for rep in REPS for d in DEGREES
          for f in FAMILIES]
    assert len(rows)==len(seen)==12
    return dict(schema=SOURCE_SCHEMA,source_offset=OFFSET,replicates=list(REPS),
                source_only=True,private_truth_absent=True,
                attacks_absent=True,challenge_entropy_absent=True,cases=rows)


def load_fixture(path):
    raw=Path(path).read_bytes()
    fixture=json.loads(raw)
    assert fixture["schema"]==SOURCE_SCHEMA
    assert fixture["source_offset"]==OFFSET
    assert fixture["replicates"]==list(REPS)
    assert fixture["source_only"] and fixture["private_truth_absent"]
    assert fixture["attacks_absent"] and fixture["challenge_entropy_absent"]
    rows=fixture["cases"]
    assert len(rows)==12
    assert {(r["replicate"],r["observation_degree"],r["family"]) for r in rows}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    assert len({r["full_public_input_sha256"] for r in rows})==12
    assert all(r["split"]=="TEST" and not any(
        bad in r for bad in ("program","future","attack","nonce","entropy","witness"))
        for r in rows)
    return fixture,sha256(raw).hexdigest()


def public_order(pred,header,digest):
    regions=n24.supports(pred,header)
    order=n24.static_schedule(regions,n25.PRIOR,digest,0)
    assert len(order)==len(set(order))==64
    return order


def attack_sets(pred,header,digest):
    order=public_order(pred,header,digest)
    m=len(pred)
    cal_digest=sha256(("nor28-calibration|"+digest).encode("ascii")).hexdigest()
    schedules=[
        n25.hybrid_schedule(order,m,cal_digest,"hybrid_50",16,trial)
        for trial in range(CAL_TRIALS)]
    exposure=n26.exposure_bitsets(schedules,m)
    greedy=[]
    for rep in range(ATTACK_REPS):
        tie=list(range(m))
        random.Random(hash_int(202611280091,digest,rep)).shuffle(tie)
        used=set()
        seen=0
        for _ in range(R):
            j=min((i for i in tie if i not in used),
                  key=lambda i:(seen|exposure[i]).bit_count())
            used.add(j)
            seen|=exposure[j]
        greedy.append(tuple(sorted(used)))
    fresh=[
        tuple(sorted(random.Random(hash_int(202611280092,digest,rep)).sample(
            range(m),R)))
        for rep in range(ATTACK_REPS)]
    assert all(len(set(p))==R for p in greedy+fresh)
    return {"calibrated":greedy,"fresh_uniform":fresh}


def commitment(preimage):
    return sha256(b"NOR28-attack-commit-v1\x00"+canonical(preimage)).hexdigest()


def make_commitments(fixture_path,sealed_path,manifest_path):
    public,fixture_hash=load_fixture(fixture_path)
    proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
               for r in public["cases"]}
    assert len(proposals)==12
    openings=[]
    entries=[]
    for row in public["cases"]:
        digest=row["full_public_input_sha256"]
        p=proposals[digest]
        for kind,attacks in attack_sets(
                p["predicted"],p["header"],digest).items():
            for rep,positions in enumerate(attacks):
                opening=dict(source_sha256=digest,attack_class=kind,
                             replica=rep,changed_indices=list(positions),
                             nonce_hex=secrets.token_bytes(32).hex())
                openings.append(opening)
                entries.append(dict(source_sha256=digest,attack_class=kind,
                                    replica=rep,commitment_sha256=commitment(opening)))
    assert len(entries)==len(openings)==96
    sealed=dict(schema=SEALED_SCHEMA,fixture_sha256=fixture_hash,openings=openings)
    sealed_bytes=canonical(sealed)
    manifest=dict(schema=MANIFEST_SCHEMA,fixture_sha256=fixture_hash,
                  sealed_sha256=sha256(sealed_bytes).hexdigest(),
                  sources=12,attack_count=96,challenge_entropy_absent=True,
                  commitments=entries)
    for dst,blob in ((sealed_path,sealed_bytes),(manifest_path,canonical(manifest))):
        Path(dst).parent.mkdir(parents=True,exist_ok=True)
        Path(dst).write_bytes(blob)
    verify_openings(fixture_path,sealed_path,manifest_path)
    print("NOR28 PRECOMMITTED",len(openings),"attacks",
          "fixture",fixture_hash,"sealed",manifest["sealed_sha256"],
          "manifest",sha256(canonical(manifest)).hexdigest())


def load_manifest(fixture_path,manifest_path):
    fixture,fixture_hash=load_fixture(fixture_path)
    m=json.loads(Path(manifest_path).read_bytes())
    assert m["schema"]==MANIFEST_SCHEMA
    assert m["fixture_sha256"]==fixture_hash
    assert m["sources"]==12 and m["attack_count"]==96
    assert m["challenge_entropy_absent"] is True
    entries=m["commitments"]
    assert len(entries)==96
    expected={(r["full_public_input_sha256"],k,rep)
              for r in fixture["cases"] for k in KINDS
              for rep in range(ATTACK_REPS)}
    actual={(e["source_sha256"],e["attack_class"],e["replica"])
            for e in entries}
    assert expected==actual and len({e["commitment_sha256"] for e in entries})==96
    return m,sha256(Path(manifest_path).read_bytes()).hexdigest()


def verify_openings(fixture_path,sealed_path,manifest_path):
    fixture,_=load_fixture(fixture_path)
    manifest,manifest_hash=load_manifest(fixture_path,manifest_path)
    raw=Path(sealed_path).read_bytes()
    assert sha256(raw).hexdigest()==manifest["sealed_sha256"]
    sealed=json.loads(raw)
    assert sealed["schema"]==SEALED_SCHEMA
    assert sealed["fixture_sha256"]==manifest["fixture_sha256"]
    entries={ (e["source_sha256"],e["attack_class"],e["replica"]):
              e["commitment_sha256"] for e in manifest["commitments"]}
    sizes={r["full_public_input_sha256"]:r["query_degree"]+1
           for r in fixture["cases"]}
    seen=set()
    for op in sealed["openings"]:
        key=(op["source_sha256"],op["attack_class"],op["replica"])
        assert key in entries and key not in seen
        seen.add(key)
        assert set(op)=={"source_sha256","attack_class","replica",
                        "changed_indices","nonce_hex"}
        positions=op["changed_indices"]
        assert len(positions)==len(set(positions))==4
        assert positions==sorted(positions)
        assert all(type(x) is int and 0<=x<sizes[key[0]] for x in positions)
        assert len(op["nonce_hex"])==64
        assert all(c in "0123456789abcdef" for c in op["nonce_hex"])
        assert commitment(op)==entries[key]
        forged=dict(op)
        forged["nonce_hex"]=("0" if op["nonce_hex"][0]!="0" else "1")+op["nonce_hex"][1:]
        assert commitment(forged)!=entries[key]
    assert len(seen)==len(sealed["openings"])==96
    return sealed,manifest_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--commit-attacks",action="store_true")
    p.add_argument("--verify-openings",action="store_true")
    p.add_argument("--fixture",type=Path)
    p.add_argument("--sealed",type=Path)
    p.add_argument("--manifest",type=Path)
    a=p.parse_args()
    if a.make_fixture:
        assert not (a.fixture or a.sealed or a.manifest)
        raw=canonical(make_fixture())
        a.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        a.make_fixture.write_bytes(raw)
        print("NOR28 SOURCE-ONLY FIXTURE",len(raw),sha256(raw).hexdigest())
        return
    if not (a.fixture and a.sealed and a.manifest):
        p.error("--fixture --sealed --manifest required")
    if a.commit_attacks and not a.verify_openings:
        make_commitments(a.fixture,a.sealed,a.manifest)
    elif a.verify_openings and not a.commit_attacks:
        verify_openings(a.fixture,a.sealed,a.manifest)
        print("NOR28 ALL ORIGINAL ATTACK COMMITMENTS VERIFIED")
    else:
        p.error("exactly one --commit-attacks or --verify-openings required")


if __name__=="__main__": main()
