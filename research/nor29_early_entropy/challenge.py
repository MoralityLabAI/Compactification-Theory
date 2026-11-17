"""NOR-29 fresh OS-root challenge schedules, drawn after early commitments.

This module sees only source-only public inputs and the public PREP manifest.
It does NOT read the sealed attack openings or optimize any late attacks.
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
from research.nor29_early_entropy import prepare
from research.nor24_adaptive_audits import experiment as n24
from research.nor28_commit_entropy import challenge_sampler as prior_sampler

SCHEMA="nor29.single_fresh_os_root_24x3x512_schedules.v1"
TRIALS=prepare.TRIALS
ARMS=prepare.ARMS


def root_commitment(root):
    assert isinstance(root,bytes) and len(root)==32
    return sha256(b"NOR29-root-v1\x00"+root).hexdigest()


def audit_key(root):
    assert isinstance(root,bytes) and len(root)==32
    return hmac.new(root,b"NOR29-v1-schedule-subkey",sha256).digest()


def draw(root,digest,arm,trial,m,order):
    assert arm in ARMS and 0<=trial<TRIALS
    return prior_sampler.schedule(audit_key(root),digest,arm,trial,m,order)


def make_schedules(fixture_path,manifest_path,root=None):
    fixture,fixture_hash=prepare.load_fixture(fixture_path)
    manifest,manifest_hash=prepare.verify_manifest(fixture_path,manifest_path)
    assert manifest["challenge_entropy_absent"]
    os_new=root is None
    if root is None:
        # Exactly one fresh 256-bit root; PREP is an independent completed job.
        root=secrets.token_bytes(32)
    assert isinstance(root,bytes) and len(root)==32
    proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
               for r in fixture["cases"]}
    assert len(proposals)==prepare.N_SOURCES
    records=[]
    for row in fixture["cases"]:
        digest=row["full_public_input_sha256"]
        p=proposals[digest]
        m=len(p["predicted"])
        order=prepare.rank_order(p["predicted"],p["header"],digest)
        schedules={
            arm:[list(draw(root,digest,arm,j,m,order))
                 for j in range(TRIALS)]
            for arm in ARMS}
        assert all(len(plans)==TRIALS for plans in schedules.values())
        records.append({
            "public_sha256":digest,"m":m,
            "public_prediction_sha256":p["sha256"],
            "original_public_mask":p["mask"],
            "risk_prefix":list(order[:8]),
            "schedules":schedules,
        })
    return {
        "schema":SCHEMA,
        "fixture_sha256":fixture_hash,
        "early_manifest_sha256":manifest_hash,
        "source_count":prepare.N_SOURCES,
        "trial_count":TRIALS,
        "root_origin":"os_secrets_token_bytes_32" if os_new else "test_replay_root",
        "root_commitment_sha256":root_commitment(root),
        "root_hex_disclosed_after_freeze":root.hex(),
        "key_derivation":"HMAC-SHA256(root,NOR29-v1-schedule-subkey)",
        "sampler":"NOR28 rejection-sampled HMAC-SHA256 partial Fisher-Yates",
        "records":records,
    }


def verify_schedules(fixture_path,manifest_path,schedules_path,exact=True):
    fixture,fixture_hash=prepare.load_fixture(fixture_path)
    _,manifest_hash=prepare.verify_manifest(fixture_path,manifest_path)
    raw=Path(schedules_path).read_bytes()
    x=json.loads(raw)
    assert x["schema"]==SCHEMA
    assert x["fixture_sha256"]==fixture_hash
    assert x["early_manifest_sha256"]==manifest_hash
    assert x["source_count"]==prepare.N_SOURCES
    assert x["trial_count"]==TRIALS
    assert x["key_derivation"]=="HMAC-SHA256(root,NOR29-v1-schedule-subkey)"
    assert x["sampler"]=="NOR28 rejection-sampled HMAC-SHA256 partial Fisher-Yates"
    assert len(x["records"])==prepare.N_SOURCES
    assert [a["public_sha256"] for a in x["records"]]==[
        r["full_public_input_sha256"] for r in fixture["cases"]]
    root=bytes.fromhex(x["root_hex_disclosed_after_freeze"])
    assert len(root)==32 and root_commitment(root)==x["root_commitment_sha256"]
    if exact:
        proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
                   for r in fixture["cases"]}
    for row in x["records"]:
        assert row["m"] in (769,1537)
        assert len(row["risk_prefix"])==8
        assert set(row["schedules"])==set(ARMS)
        for arm in ARMS:
            plans=row["schedules"][arm]
            assert len(plans)==TRIALS
            assert all(len(p)==len(set(p))==16 and
                       all(0<=i<row["m"] for i in p)
                       for p in plans)
        if exact:
            proposal=proposals[row["public_sha256"]]
            order=prepare.rank_order(proposal["predicted"],proposal["header"],
                                    row["public_sha256"])
            assert row["public_prediction_sha256"]==proposal["sha256"]
            assert row["original_public_mask"]==proposal["mask"]
            assert row["risk_prefix"]==list(order[:8])
            for arm in ARMS:
                for j,expected in enumerate(row["schedules"][arm]):
                    actual=draw(root,row["public_sha256"],arm,j,row["m"],order)
                    assert expected==list(actual),(row["public_sha256"],arm,j)
    return x,sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--verify",type=Path)
    a=p.parse_args()
    if bool(a.out)==bool(a.verify):
        p.error("choose --out or --verify")
    if a.out:
        output=make_schedules(a.fixture,a.manifest)
        assert output["root_origin"]=="os_secrets_token_bytes_32"
        raw=prepare.canonical(output)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_bytes(raw)
        print("NOR29 FRESH OS ROOT AFTER BLIND COMMIT",output["root_commitment_sha256"])
        print("NOR29 SCHEDULES FROZEN",len(raw),sha256(raw).hexdigest())
    else:
        _,digest=verify_schedules(a.fixture,a.manifest,a.verify,exact=True)
        print("NOR29 SCHEDULES INDEPENDENTLY REGENERATED",digest)


if __name__=="__main__":main()
