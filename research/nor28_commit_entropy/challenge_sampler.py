"""NOR-28 challenge sampler. IMPORTANT: this file never opens attack preimages.

PREP is a completed GitHub Actions prerequisite. This CHALLENGE process reads
only the source-only fixture and public commitment manifest, then draws ONE
fresh OS root via secrets.token_bytes(32). Schedules are hashed and written
before a separate grader may inspect the sealed attack openings.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research.nor28_commit_entropy import experiment as base
from research.nor24_adaptive_audits import experiment as n24

SCHEMA="nor28.fresh_os_entropy_512_audits_per_source.v1"
TRIALS=512
ARMS=("hybrid_50","uniform16","static_prior")
LEAK_BITS=(0,8,16,256)
U=8
T=8
R=4
K=16
M=1<<256


def root_commitment(root):
    assert isinstance(root,bytes) and len(root)==32
    return hashlib.sha256(b"NOR28-root-v1\x00"+root).hexdigest()


def uniform_without_replacement(root,digest,arm,trial,eligible,k):
    """HMAC-SHA256 keyed counter draws with rejection to eliminate modulo bias."""
    remaining=list(eligible)
    assert len(set(remaining))==len(remaining)
    assert len(remaining)>=k
    chosen=[]
    for draw in range(k):
        n=len(remaining)
        limit=M-(M%n)
        reject=0
        while True:
            message=(f"NOR28-sampling-v1|{digest}|{arm}|{trial}|"
                     f"{draw}|{reject}").encode("ascii")
            candidate=int.from_bytes(hmac.new(
                root,message,hashlib.sha256).digest(),"big")
            if candidate<limit:
                j=candidate%n
                break
            reject+=1
            assert reject<100
        chosen.append(remaining[j])
        remaining[j]=remaining[-1]
        remaining.pop()
    assert len(chosen)==len(set(chosen))==k
    return tuple(chosen)


def schedule(root,digest,arm,trial,m,static_order):
    assert arm in ARMS and 0<=trial<TRIALS
    if arm=="static_prior":
        return tuple(static_order[:K])
    if arm=="uniform16":
        return uniform_without_replacement(root,digest,arm,trial,range(m),K)
    targeted=tuple(static_order[:T])
    rest=(i for i in range(m) if i not in set(targeted))
    reserve=uniform_without_replacement(root,digest,arm,trial,rest,U)
    return targeted+reserve


def make_challenges(fixture_path,manifest_path,root=None):
    fixture,source_sha=base.load_fixture(fixture_path)
    manifest,manifest_sha=base.load_manifest(fixture_path,manifest_path)
    assert manifest["challenge_entropy_absent"] is True
    # ROOT IS MINTED AFTER THE PUBLIC COMMITMENT MANIFEST IS VALIDATED.
    os_origin = root is None
    if root is None:
        root=secrets.token_bytes(32)
    assert len(root)==32
    proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
               for r in fixture["cases"]}
    records=[]
    for row in fixture["cases"]:
        digest=row["full_public_input_sha256"]
        p=proposals[digest]
        m=len(p["predicted"])
        order=base.public_order(p["predicted"],p["header"],digest)
        records.append({
            "public_sha256":digest,"m":m,
            "public_prediction_sha256":p["sha256"],
            "public_mask":p["mask"],
            "targeted_prefix":list(order[:T]),
            "schedules":{
                arm:[list(schedule(root,digest,arm,trial,m,order))
                     for trial in range(TRIALS)]
                for arm in ARMS},
        })
    return {
        "schema":SCHEMA,"fixture_sha256":source_sha,
        "manifest_sha256":manifest_sha,
        "source_count":12,"trial_count":TRIALS,
        "root_commitment_sha256":root_commitment(root),
        "root_hex_disclosed_after_commit":root.hex(),
        "root_origin":"os_secrets_token_bytes_32" if os_origin else "test_replay_root",
        "sampler":"HMAC-SHA256-counter rejection Fisher-Yates, no replacement",
        "postcommit_leak_bits":list(LEAK_BITS),
        "records":records,
    }


def verify_challenges(fixture_path,manifest_path,challenge_path):
    original,source_sha=base.load_fixture(fixture_path)
    manifest,manifest_sha=base.load_manifest(fixture_path,manifest_path)
    receipt=json.loads(Path(challenge_path).read_bytes())
    assert receipt["schema"]==SCHEMA
    assert receipt["fixture_sha256"]==source_sha
    assert receipt["manifest_sha256"]==manifest_sha
    assert receipt["source_count"]==12 and receipt["trial_count"]==TRIALS
    assert receipt["sampler"]=="HMAC-SHA256-counter rejection Fisher-Yates, no replacement"
    assert receipt["postcommit_leak_bits"]==list(LEAK_BITS)
    root=bytes.fromhex(receipt["root_hex_disclosed_after_commit"])
    assert root_commitment(root)==receipt["root_commitment_sha256"]
    assert len(receipt["records"])==12
    assert [r["public_sha256"] for r in receipt["records"]]==[
        r["full_public_input_sha256"] for r in original["cases"]]
    for entry in receipt["records"]:
        assert len(entry["targeted_prefix"])==T
        for arm in ARMS:
            plans=entry["schedules"][arm]
            assert len(plans)==TRIALS
            assert all(len(p)==len(set(p))==K and
                       all(0<=i<entry["m"] for i in p)
                       for p in plans)
            if arm=="hybrid_50":
                assert all(p[:T]==entry["targeted_prefix"] for p in plans)
    return receipt


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--fixture",type=Path,required=True)
    a.add_argument("--manifest",type=Path,required=True)
    a.add_argument("--out",type=Path)
    a.add_argument("--verify",type=Path)
    args=a.parse_args()
    if bool(args.out)==bool(args.verify):
        a.error("choose exactly --out or --verify")
    if args.out:
        output=make_challenges(args.fixture,args.manifest)
        assert output["root_origin"]=="os_secrets_token_bytes_32"
        raw=base.canonical(output)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(raw)
        print("NOR28 OS ROOT DRAWN AFTER PREP",output["root_commitment_sha256"])
        print("NOR28 SCHEDULES FROZEN",len(raw),hashlib.sha256(raw).hexdigest())
    else:
        verify_challenges(args.fixture,args.manifest,args.verify)
        print("NOR28 COMPLETE AUDIT SCHEDULES HASH VERIFIED",
              hashlib.sha256(args.verify.read_bytes()).hexdigest())


if __name__=="__main__": main()
