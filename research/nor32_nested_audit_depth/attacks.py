"""NOR-32 matched greedy attacker at nested audit depths and salted commitments.

Identical four-coordinate greedy strategy with 32 tie restarts per source,
policy and depth. Before-root attacks use prefix n of an independent OS
calibration root. After-root attacks use prefix n of the actual evaluation
root. Sealed attack support is held fixed across the chosen n-batch.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import json
import random
import secrets
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor32_nested_audit_depth import source,sampler

PRE_SCHEMA="nor32.pre_root_9216_salted_openings.v1"
PRE_MANIFEST="nor32.pre_root_9216_commitments.v1"
LATE_SCHEMA="nor32.leaked_root_9216_salted_openings.v1"
LATE_MANIFEST="nor32.leaked_root_9216_commitments.v1"
REPLICAS=32
ATTACK_RADIUS=4
TIE_STREAM=202611310111
LABELS=("independent_calibration","leaked_evaluation_root")
TOTAL=source.N_SOURCES*len(sampler.ARMS)*len(sampler.DEPTHS)*REPLICAS
assert TOTAL==9216


def commitment(obj):
    return sha256(b"NOR32-attack-commit-v1\x00"+source.canonical(obj)).hexdigest()


def exposure_masks(plans,m):
    seen=[0]*m
    for ordinal,checked in enumerate(plans):
        for i in checked:
            seen[i]|=1<<ordinal
    return seen


def greedy(masks,digest,arm,rep):
    if arm not in sampler.ARMS or not 0<=rep<REPLICAS:
        raise ValueError("NOR32 optimizer argument outside registration")
    order=list(range(len(masks)))
    random.Random(source.hash_int(TIE_STREAM,digest,arm,rep)).shuffle(order)
    selected=set()
    union=0
    for _ in range(ATTACK_RADIUS):
        best=min((i for i in order if i not in selected),
                 key=lambda i:(union|masks[i]).bit_count())
        selected.add(best)
        union|=masks[best]
    return tuple(sorted(selected)),union.bit_count()


def verify_manifest(fixture_path,manifest_path,label):
    fixture,fhash=source.load_fixture(fixture_path)
    raw=Path(manifest_path).read_bytes()
    m=json.loads(raw)
    assert label in LABELS
    assert m["schema"]==(PRE_MANIFEST if label==LABELS[0] else LATE_MANIFEST)
    assert m["attacker_class"]==label
    assert m["fixture_sha256"]==fhash
    assert m["source_count"]==source.N_SOURCES
    assert m["attack_count"]==TOTAL
    assert m["depths"]==list(sampler.DEPTHS)
    assert m["root_of_phase_known_before_committing"]==(label==LABELS[1])
    entries=m["commitments"]
    assert len(entries)==TOTAL
    expected={(r["full_public_input_sha256"],arm,d,rep)
              for r in fixture["cases"] for arm in sampler.ARMS
              for d in sampler.DEPTHS for rep in range(REPLICAS)}
    observed={(e["source_sha256"],e["auditor"],e["depth"],e["replica"])
              for e in entries}
    assert expected==observed
    assert len({e["commitment_sha256"] for e in entries})==TOTAL
    return m,sha256(raw).hexdigest()


def verify_openings(fixture_path,phase_path,sealed_path,manifest_path,label,
                    pre_manifest_path=None,calibration_path=None):
    fixture,_=source.load_fixture(fixture_path)
    phase,phase_hash=sampler.verify_phase(fixture_path,phase_path,exact=False)
    manifest,manifest_hash=verify_manifest(fixture_path,manifest_path,label)
    sealed_raw=Path(sealed_path).read_bytes()
    assert sha256(sealed_raw).hexdigest()==manifest["sealed_sha256"]
    sealed=json.loads(sealed_raw)
    assert sealed["schema"]==(PRE_SCHEMA if label==LABELS[0] else LATE_SCHEMA)
    assert manifest["phase_sha256"]==sealed["phase_sha256"]==phase_hash
    assert manifest["calibration_sha256"]==sealed["calibration_sha256"]
    assert manifest["precommit_manifest_sha256"]==sealed["precommit_manifest_sha256"]
    if label==LABELS[0]:
        assert phase["phase"]=="calibration"
        assert manifest["calibration_sha256"]==phase_hash
        assert manifest["precommit_manifest_sha256"] is None
    else:
        assert phase["phase"]=="evaluation"
        assert pre_manifest_path is not None and calibration_path is not None
        pre,pre_hash=verify_manifest(fixture_path,pre_manifest_path,LABELS[0])
        calibration,cal_hash=sampler.verify_phase(fixture_path,calibration_path,exact=False)
        assert calibration["phase"]=="calibration"
        assert pre["calibration_sha256"]==cal_hash
        assert manifest["calibration_sha256"]==cal_hash
        assert manifest["precommit_manifest_sha256"]==pre_hash
        assert phase["precommit_manifest_sha256"]==pre_hash
        assert phase["calibration_sha256"]==cal_hash
    refs={(e["source_sha256"],e["auditor"],e["depth"],e["replica"]):
          e["commitment_sha256"] for e in manifest["commitments"]}
    sizes={r["full_public_input_sha256"]:r["query_degree"]+1
           for r in fixture["cases"]}
    indexed={r["source_sha256"]:r for r in phase["records"]}
    assert len(sealed["openings"])==TOTAL
    seen=set()
    for item in sealed["openings"]:
        key=(item["source_sha256"],item["auditor"],item["depth"],item["replica"])
        assert key in refs and key not in seen
        seen.add(key)
        assert item["attack_class"]==label
        assert set(item)=={"source_sha256","auditor","depth","attack_class",
                           "replica","changed_indices","training_covered_schedules","nonce_hex"}
        changed=item["changed_indices"]
        assert changed==sorted(changed)
        assert len(changed)==len(set(changed))==ATTACK_RADIUS
        assert all(type(i) is int and 0<=i<sizes[key[0]] for i in changed)
        assert len(item["nonce_hex"])==64
        assert all(c in "0123456789abcdef" for c in item["nonce_hex"])
        assert commitment(item)==refs[key]
        tampered=dict(item,nonce_hex=("0" if item["nonce_hex"][0]!="0" else "1")+item["nonce_hex"][1:])
        assert commitment(tampered)!=refs[key]
        alternative=list(changed)
        alternative[-1]=next(i for i in range(sizes[key[0]]) if i not in changed)
        alternative.sort()
        forged=dict(item,changed_indices=alternative)
        assert commitment(forged)!=refs[key]
        plans=indexed[key[0]]["schedules"][key[1]][:key[2]]
        assert len(plans)==key[2]
        assert item["training_covered_schedules"]==sum(bool(set(changed)&set(p)) for p in plans)
    assert len(seen)==TOTAL
    return sealed,manifest_hash


def open_and_commit(fixture_path,phase_path,sealed_path,manifest_path,label,
                    pre_manifest_path=None,calibration_path=None):
    fixture,fixture_sha=source.load_fixture(fixture_path)
    phase,phase_sha=sampler.verify_phase(fixture_path,phase_path,exact=False)
    if label==LABELS[0]:
        assert phase["phase"]=="calibration"
        assert pre_manifest_path is None and calibration_path is None
        pre_hash=None
        calibration_hash=phase_sha
        ss,ms=PRE_SCHEMA,PRE_MANIFEST
    elif label==LABELS[1]:
        assert phase["phase"]=="evaluation"
        assert pre_manifest_path is not None and calibration_path is not None
        pre,pre_hash=verify_manifest(fixture_path,pre_manifest_path,LABELS[0])
        calibration,calibration_hash=sampler.verify_phase(
            fixture_path,calibration_path,exact=False)
        assert calibration["phase"]=="calibration"
        assert phase["precommit_manifest_sha256"]==pre_hash
        assert phase["calibration_sha256"]==calibration_hash
        assert pre["calibration_sha256"]==calibration_hash
        ss,ms=LATE_SCHEMA,LATE_MANIFEST
    else:raise ValueError("unregistered NOR32 attack condition")
    openings=[]
    entries=[]
    for rec in phase["records"]:
        digest=rec["source_sha256"]
        m=rec["m"]
        for arm in sampler.ARMS:
            all_plans=rec["schedules"][arm]
            for depth in sampler.DEPTHS:
                masks=exposure_masks(all_plans[:depth],m)
                for rep in range(REPLICAS):
                    attack,cov=greedy(masks,digest,arm,rep)
                    preimage={"source_sha256":digest,"auditor":arm,"depth":depth,
                              "attack_class":label,"replica":rep,
                              "changed_indices":list(attack),
                              "training_covered_schedules":cov,
                              "nonce_hex":secrets.token_bytes(32).hex()}
                    openings.append(preimage)
                    entries.append({
                        "source_sha256":digest,"auditor":arm,"depth":depth,
                        "replica":rep,"commitment_sha256":commitment(preimage),
                    })
    assert len(openings)==len(entries)==TOTAL
    sealed={"schema":ss,"fixture_sha256":fixture_sha,
            "phase_sha256":phase_sha,"calibration_sha256":calibration_hash,
            "precommit_manifest_sha256":pre_hash,"openings":openings}
    raw=source.canonical(sealed)
    manifest={"schema":ms,"fixture_sha256":fixture_sha,
              "phase_sha256":phase_sha,"calibration_sha256":calibration_hash,
              "precommit_manifest_sha256":pre_hash,
              "sealed_sha256":sha256(raw).hexdigest(),
              "source_count":source.N_SOURCES,"attack_count":TOTAL,
              "attacker_class":label,"depths":list(sampler.DEPTHS),
              "root_of_phase_known_before_committing":label==LABELS[1],
              "commitments":entries}
    for path,b in ((sealed_path,raw),(manifest_path,source.canonical(manifest))):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    verify_openings(fixture_path,phase_path,sealed_path,manifest_path,label,
                    pre_manifest_path,calibration_path)
    print("NOR32 COMMITTED",label,TOTAL,"depth-specific attacks",
          "root",phase["root_commitment_sha256"],"sealed",sha256(raw).hexdigest())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--class",dest="label",choices=LABELS,required=True)
    for key in ("fixture","phase","sealed","manifest","pre-manifest","calibration"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--make",action="store_true")
    p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    if a.make==a.verify:p.error("choose --make or --verify")
    if not all((a.fixture,a.phase,a.sealed,a.manifest)):
        p.error("fixture, phase, sealed, manifest required")
    fn=open_and_commit if a.make else verify_openings
    fn(a.fixture,a.phase,a.sealed,a.manifest,a.label,a.pre_manifest,a.calibration)
    if a.verify:print("NOR32 VERIFIED",a.label,TOTAL,"source-depth commitments")


if __name__=="__main__":main()
