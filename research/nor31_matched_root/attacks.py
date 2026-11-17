"""NOR-31 matched greedy attacker and source-bound salted commitments.

Identical 32 restart tie streams, the same 512-schedule search budget,
and exactly the same algorithm in both arms. The only difference is whether
the attacker optimized on an independent calibration root (PRECOMMIT), or
the realized evaluation root (LEAKED).
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import secrets
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor31_matched_root import source
from research.nor31_matched_root import sampler

PRE_SCHEMA="nor31.pre_root_2304_salted_openings.v1"
PRE_MANIFEST="nor31.pre_root_2304_commitments.v1"
LATE_SCHEMA="nor31.leaked_root_2304_salted_openings.v1"
LATE_MANIFEST="nor31.leaked_root_2304_commitments.v1"
ATTACK_RADIUS=4
REPLICAS=32
TIE_STREAM=202611310111
LABELS=("independent_calibration","leaked_evaluation_root")


def commitment(preimage):
    return sha256(b"NOR31-attack-commit-v1\x00"+source.canonical(preimage)).hexdigest()


def exposure_masks(plans,m):
    masks=[0]*m
    for ordinal,checked in enumerate(plans):
        for i in checked:
            masks[i]|=1<<ordinal
    return masks


def greedy(masks,digest,arm,rep):
    """Same source/policy/rep tie order across both independent-root treatments."""
    assert arm in sampler.ARMS and 0<=rep<REPLICAS
    positions=list(range(len(masks)))
    random.Random(source.hash_int(TIE_STREAM,digest,arm,rep)).shuffle(positions)
    selected=set()
    union=0
    for _ in range(ATTACK_RADIUS):
        best=min((i for i in positions if i not in selected),
                 key=lambda i:(union|masks[i]).bit_count())
        selected.add(best)
        union|=masks[best]
    return tuple(sorted(selected)),union.bit_count()


def open_and_commit(fixture_path,phase_path,sealed_path,manifest_path,
                    label,pre_manifest_path=None,calibration_path=None):
    fixture,fixture_sha=source.load_fixture(fixture_path)
    phase,phase_sha=sampler.verify_phase(fixture_path,phase_path,exact=False)
    if label=="independent_calibration":
        assert phase["phase"]=="calibration"
        assert pre_manifest_path is None and calibration_path is None
        sealed_schema,manifest_schema=PRE_SCHEMA,PRE_MANIFEST
        pre_hash=None
        calibration_sha=phase_sha
    elif label=="leaked_evaluation_root":
        assert phase["phase"]=="evaluation"
        assert pre_manifest_path is not None and calibration_path is not None
        pre,pre_hash=verify_manifest(
            fixture_path,pre_manifest_path,"independent_calibration")
        assert phase["precommit_manifest_sha256"]==pre_hash
        calibration,calibration_sha=sampler.verify_phase(
            fixture_path,calibration_path,exact=False)
        assert calibration["phase"]=="calibration"
        assert calibration_sha==pre["calibration_sha256"]
        assert phase["calibration_sha256"]==calibration_sha
        sealed_schema,manifest_schema=LATE_SCHEMA,LATE_MANIFEST
    else:
        raise ValueError("unregistered NOR31 attacker class")
    openings=[]
    public=[]
    for row in phase["records"]:
        digest=row["source_sha256"]
        m=row["m"]
        for arm in sampler.ARMS:
            masks=exposure_masks(row["schedules"][arm],m)
            for rep in range(REPLICAS):
                A,cov=greedy(masks,digest,arm,rep)
                preimage={
                    "source_sha256":digest,
                    "auditor":arm,
                    "attack_class":label,
                    "replica":rep,
                    "changed_indices":list(A),
                    "training_covered_schedules":cov,
                    "nonce_hex":secrets.token_bytes(32).hex(),
                }
                openings.append(preimage)
                public.append({
                    "source_sha256":digest,"auditor":arm,"replica":rep,
                    "commitment_sha256":commitment(preimage),
                })
    assert len(openings)==len(public)==source.N_SOURCES*len(sampler.ARMS)*REPLICAS==2304
    sealed={"schema":sealed_schema,
            "phase_sha256":phase_sha,
            "fixture_sha256":fixture_sha,
            "calibration_sha256":calibration_sha,
            "precommit_manifest_sha256":pre_hash,
            "openings":openings}
    sealed_raw=source.canonical(sealed)
    manifest={"schema":manifest_schema,
              "fixture_sha256":fixture_sha,
              "phase_sha256":phase_sha,
              "calibration_sha256":calibration_sha,
              "precommit_manifest_sha256":pre_hash,
              "sealed_sha256":sha256(sealed_raw).hexdigest(),
              "source_count":source.N_SOURCES,
              "attack_count":len(public),
              "attacker_class":label,
              "root_of_phase_known_before_committing":phase["phase"]=="evaluation",
              "commitments":public}
    for path,blob in ((sealed_path,sealed_raw),(manifest_path,source.canonical(manifest))):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(blob)
    verify_openings(fixture_path,phase_path,sealed_path,manifest_path,
                    label,pre_manifest_path,calibration_path)
    print("NOR31 ORIGINAL",label,"ATTACKS COMMITTED",
          len(openings),"root",phase["root_commitment_sha256"],
          "sealed SHA256",sha256(sealed_raw).hexdigest())


def verify_manifest(fixture_path,manifest_path,label):
    fixture,fixture_sha=source.load_fixture(fixture_path)
    raw=Path(manifest_path).read_bytes()
    m=json.loads(raw)
    assert label in LABELS
    assert m["schema"]==(PRE_MANIFEST if label=="independent_calibration"
                          else LATE_MANIFEST)
    assert m["attacker_class"]==label
    assert m["fixture_sha256"]==fixture_sha
    assert m["source_count"]==source.N_SOURCES
    assert m["attack_count"]==2304
    assert m["root_of_phase_known_before_committing"]==(label=="leaked_evaluation_root")
    assert len(m["commitments"])==2304
    expected={(r["full_public_input_sha256"],arm,rep)
              for r in fixture["cases"] for arm in sampler.ARMS for rep in range(REPLICAS)}
    actual={(x["source_sha256"],x["auditor"],x["replica"]) for x in m["commitments"]}
    assert expected==actual
    assert len({x["commitment_sha256"] for x in m["commitments"]})==2304
    return m,sha256(raw).hexdigest()


def verify_openings(fixture_path,phase_path,sealed_path,manifest_path,label,
                    pre_manifest_path=None,calibration_path=None):
    fixture,_=source.load_fixture(fixture_path)
    phase,phase_sha=sampler.verify_phase(fixture_path,phase_path,exact=False)
    manifest,manifest_sha=verify_manifest(fixture_path,manifest_path,label)
    raw=Path(sealed_path).read_bytes()
    assert sha256(raw).hexdigest()==manifest["sealed_sha256"]
    sealed=json.loads(raw)
    assert sealed["schema"]==(PRE_SCHEMA if label=="independent_calibration" else LATE_SCHEMA)
    assert sealed["phase_sha256"]==manifest["phase_sha256"]==phase_sha
    assert sealed["calibration_sha256"]==manifest["calibration_sha256"]
    assert sealed["precommit_manifest_sha256"]==manifest["precommit_manifest_sha256"]
    if label=="leaked_evaluation_root":
        prior,prior_sha=verify_manifest(fixture_path,pre_manifest_path,"independent_calibration")
        assert prior_sha==manifest["precommit_manifest_sha256"]
        calib,calib_sha=sampler.verify_phase(fixture_path,calibration_path,exact=False)
        assert calib_sha==manifest["calibration_sha256"] and calib["phase"]=="calibration"
        assert phase["phase"]=="evaluation"
    else:
        assert phase["phase"]=="calibration"
        assert manifest["calibration_sha256"]==phase_sha
        assert manifest["precommit_manifest_sha256"] is None
    refs={(x["source_sha256"],x["auditor"],x["replica"]):
          x["commitment_sha256"] for x in manifest["commitments"]}
    sizes={r["full_public_input_sha256"]:r["query_degree"]+1
           for r in fixture["cases"]}
    phase_by_id={r["source_sha256"]:r for r in phase["records"]}
    assert len(sealed["openings"])==2304
    seen=set()
    for x in sealed["openings"]:
        key=(x["source_sha256"],x["auditor"],x["replica"])
        assert key in refs and key not in seen
        seen.add(key)
        assert x["attack_class"]==label
        assert set(x)=={"source_sha256","auditor","attack_class","replica",
                       "changed_indices","training_covered_schedules","nonce_hex"}
        attacked=x["changed_indices"]
        assert attacked==sorted(attacked)
        assert len(set(attacked))==len(attacked)==ATTACK_RADIUS
        assert all(type(i) is int and 0<=i<sizes[key[0]] for i in attacked)
        assert len(x["nonce_hex"])==64
        assert commitment(x)==refs[key]
        forged=dict(x,nonce_hex=("0" if x["nonce_hex"][0]!="0" else "1")+x["nonce_hex"][1:])
        assert commitment(forged)!=refs[key]
        forged2=dict(x)
        new=list(attacked)
        new[0]=next(i for i in range(sizes[key[0]]) if i not in attacked)
        forged2["changed_indices"]=sorted(new)
        assert commitment(forged2)!=refs[key]
        # Enforce that every approximate best response truly uses
        # only this phase's 512 schedule exposures.
        plans=phase_by_id[key[0]]["schedules"][key[1]]
        assert x["training_covered_schedules"]==sum(bool(set(attacked)&set(s)) for s in plans)
    assert len(seen)==2304
    return sealed,manifest_sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--class",dest="label",choices=LABELS,required=True)
    for key in ("fixture","phase","sealed","manifest","pre-manifest","calibration"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--make",action="store_true")
    p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    if a.make==a.verify:p.error("choose exactly --make or --verify")
    if not all((a.fixture,a.phase,a.sealed,a.manifest)):
        p.error("--fixture --phase --sealed --manifest required")
    args=(a.fixture,a.phase,a.sealed,a.manifest,a.label,a.pre_manifest,a.calibration)
    if a.make:
        open_and_commit(*args)
    else:
        verify_openings(*args)
        print("NOR31 VERIFIED",a.label,"2304 SALTED COMMITMENTS")


if __name__=="__main__":main()
