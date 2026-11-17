"""NOR-32 phase-isolated OS entropy and HMAC-based 512-schedule sampler.

CALIBRATION root is minted inside PREP before attacks are chosen.
EVALUATION root is minted in a separate CHALLENGE job after PREP commits.
This module never imports or accesses attack preimages, nor true targets.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import hmac
import json
from pathlib import Path
import secrets
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor32_nested_audit_depth import source
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor28_commit_entropy import challenge_sampler as nor28

SCHEMA="nor32.nested_independent_os_roots_24x3x2048.v1"
TRIALS=2048
DEPTHS=(512,1024,1536,2048)
ARMS=("hybrid_50","uniform16","static_prior")
PHASES=("calibration","evaluation")
T=8
K=16
ROOT_BYTES=32


def hash_root(root):
    assert isinstance(root,bytes) and len(root)==ROOT_BYTES
    return sha256(b"NOR32-root-v1\x00"+root).hexdigest()


def subkey(root,phase):
    assert phase in PHASES
    assert len(root)==ROOT_BYTES
    return hmac.new(root,("NOR32-v1-"+phase+"-subkey").encode("ascii"),sha256).digest()


def risk_order(pred,header,digest):
    return n24.static_schedule(n24.supports(pred,header),n25.PRIOR,digest,0)


def draw(root,phase,digest,arm,trial,m,order):
    assert phase in PHASES and arm in ARMS and 0<=trial<TRIALS
    return nor28.schedule(subkey(root,phase),digest,arm,trial,m,order)


def make_phase(fixture_path,phase,manifest_path=None,calibration_path=None,root=None):
    assert phase in PHASES
    fixture,source_sha=source.load_fixture(fixture_path)
    manifest_sha=None
    calibration_sha=None
    if phase=="evaluation":
        assert manifest_path is not None and calibration_path is not None
        manifest_bytes=Path(manifest_path).read_bytes()
        manifest=json.loads(manifest_bytes)
        assert manifest["schema"]=="nor32.pre_root_9216_commitments.v1"
        assert manifest["fixture_sha256"]==source_sha
        assert manifest["source_count"]==source.N_SOURCES
        assert manifest["attack_count"]==source.N_SOURCES*len(ARMS)*source.REPLICAS*4
        assert manifest["calibration_sha256"]==sha256(Path(calibration_path).read_bytes()).hexdigest()
        calibration,_=verify_phase(fixture_path,calibration_path)
        assert calibration["phase"]=="calibration"
        manifest_sha=sha256(manifest_bytes).hexdigest()
        calibration_sha=sha256(Path(calibration_path).read_bytes()).hexdigest()
    else:
        assert manifest_path is None and calibration_path is None
    os_origin=root is None
    if root is None:
        # Exactly one OS entropy draw, after prerequisite phase validation.
        root=secrets.token_bytes(ROOT_BYTES)
    assert len(root)==ROOT_BYTES
    if phase=="evaluation":
        assert root.hex()!=calibration["root_hex_disclosed_after_freeze"]
    proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
               for r in fixture["cases"]}
    records=[]
    for row in fixture["cases"]:
        digest=row["full_public_input_sha256"]
        p=proposals[digest]
        m=len(p["predicted"])
        order=risk_order(p["predicted"],p["header"],digest)
        records.append({
            "source_sha256":digest,"m":m,
            "public_mask":p["mask"],"public_prediction_sha256":p["sha256"],
            "risk_prefix":list(order[:T]),
            "schedules":{
                arm:[list(draw(root,phase,digest,arm,trial,m,order))
                     for trial in range(TRIALS)]
                for arm in ARMS},
        })
    return {
        "schema":SCHEMA,"phase":phase,
        "fixture_sha256":source_sha,
        "precommit_manifest_sha256":manifest_sha,
        "calibration_sha256":calibration_sha,
        "source_count":source.N_SOURCES,"trial_count":TRIALS,
        "root_commitment_sha256":hash_root(root),
        "root_hex_disclosed_after_freeze":root.hex(),
        "root_origin":"os_secrets_token_bytes_32" if os_origin else "test_replay_root",
        "sampler":"HMAC-SHA256(NOR32 phase subkey) then NOR28 unbiased Fisher-Yates",
        "records":records,
    }


def verify_phase(fixture_path,phase_path,exact=False):
    fixture,source_sha=source.load_fixture(fixture_path)
    raw=Path(phase_path).read_bytes()
    z=json.loads(raw)
    assert z["schema"]==SCHEMA
    assert z["phase"] in PHASES
    assert z["source_count"]==source.N_SOURCES
    assert z["trial_count"]==TRIALS
    assert z["fixture_sha256"]==source_sha
    assert z["sampler"]=="HMAC-SHA256(NOR32 phase subkey) then NOR28 unbiased Fisher-Yates"
    assert len(z["records"])==source.N_SOURCES
    assert [r["source_sha256"] for r in z["records"]]==[
        r["full_public_input_sha256"] for r in fixture["cases"]]
    root=bytes.fromhex(z["root_hex_disclosed_after_freeze"])
    assert hash_root(root)==z["root_commitment_sha256"]
    assert len(root)==ROOT_BYTES
    if exact:
        proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
                   for r in fixture["cases"]}
    for rec in z["records"]:
        assert rec["m"] in (769,1537)
        assert len(rec["risk_prefix"])==T
        assert set(rec["schedules"])==set(ARMS)
        for arm in ARMS:
            plans=rec["schedules"][arm]
            assert len(plans)==TRIALS
            assert all(len(plan)==len(set(plan))==K
                       and all(0<=j<rec["m"] for j in plan) for plan in plans)
        if exact:
            p=proposals[rec["source_sha256"]]
            assert rec["public_mask"]==p["mask"]
            assert rec["public_prediction_sha256"]==p["sha256"]
            order=risk_order(p["predicted"],p["header"],rec["source_sha256"])
            assert rec["risk_prefix"]==list(order[:T])
            for arm in ARMS:
                for trial,plan in enumerate(rec["schedules"][arm]):
                    assert plan==list(draw(root,z["phase"],rec["source_sha256"],
                                           arm,trial,rec["m"],order))
    return z,sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=PHASES)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--calibration",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--verify",type=Path)
    a=p.parse_args()
    if bool(a.out)==bool(a.verify):p.error("choose --out or --verify")
    if a.out:
        if not a.phase:p.error("--phase required for mint")
        z=make_phase(a.fixture,a.phase,a.manifest,a.calibration)
        assert z["root_origin"]=="os_secrets_token_bytes_32"
        raw=source.canonical(z)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_bytes(raw)
        print("NOR32 FROZEN",a.phase,"OS ROOT",z["root_commitment_sha256"])
        print("NOR32 ORIGINAL",a.phase,len(raw),sha256(raw).hexdigest())
    else:
        z,h=verify_phase(a.fixture,a.verify,exact=True)
        if a.phase:assert z["phase"]==a.phase
        print("NOR32 REPLAYED",z["phase"],"ROOT AND SCHEDULES",h)


if __name__=="__main__":main()
