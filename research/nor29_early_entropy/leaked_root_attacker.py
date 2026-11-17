"""NOR-29 deliberately compromised stage: root-aware batch-fixed attacks.

This stage can read ALL 512 realized audit schedules and therefore must occur
strictly after CHALLENGE. It chooses one fixed four-position set per
source/auditor/replica and binds it with a NEW post-root commitment. It never
reads precommitted blind attack openings during optimization.
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
from research.nor29_early_entropy import prepare
from research.nor29_early_entropy import challenge

SEALED_SCHEMA="nor29.early_root_leaked_2304_attack_openings.v1"
PUBLIC_SCHEMA="nor29.early_root_leaked_2304_new_commitments.v1"


def coverage(plans,m):
    masks=[0]*m
    for trial,checked in enumerate(plans):
        assert len(checked)==16
        for i in checked:
            masks[i]|=1<<trial
    return masks


def greedy(exposure,digest,arm,rep):
    """Registered approximate best-response using the COMPLETE realized batch."""
    if arm not in prepare.ARMS or not 0<=rep<prepare.REPLICAS:
        raise ValueError("outside registered early-root attack grid")
    indices=list(range(len(exposure)))
    random.Random(prepare.hash_int(202611290102,digest,arm,rep)).shuffle(indices)
    used=set()
    seen=0
    for _ in range(prepare.RADIUS):
        best=None
        best_hits=None
        for i in indices:
            if i in used:continue
            hit=(seen|exposure[i]).bit_count()
            if best_hits is None or hit<best_hits:
                best=i
                best_hits=hit
        assert best is not None
        used.add(best)
        seen|=exposure[best]
    return tuple(sorted(used)),seen.bit_count()


def make_late(fixture_path,early_manifest_path,challenge_path,sealed_path,public_path):
    fixture,_=prepare.load_fixture(fixture_path)
    _,early_manifest_sha=prepare.verify_manifest(fixture_path,early_manifest_path)
    sched,challenge_sha=challenge.verify_schedules(
        fixture_path,early_manifest_path,challenge_path,exact=False)
    assert sched["root_origin"]=="os_secrets_token_bytes_32"
    assert sched["early_manifest_sha256"]==early_manifest_sha
    preimages=[]
    public=[]
    for row in sched["records"]:
        digest=row["public_sha256"]
        m=row["m"]
        for arm in prepare.ARMS:
            expo=coverage(row["schedules"][arm],m)
            for rep in range(prepare.REPLICAS):
                attack,hits=greedy(expo,digest,arm,rep)
                opening={
                    "source_sha256":digest,
                    "auditor":arm,
                    "attack_class":"early_root_aware",
                    "replica":rep,
                    "changed_indices":list(attack),
                    "calibration_batch_hits":hits,
                    "nonce_hex":secrets.token_bytes(32).hex(),
                }
                preimages.append(opening)
                public.append({
                    "source_sha256":digest,"auditor":arm,"replica":rep,
                    "commitment_sha256":prepare.commit(opening),
                })
    assert len(preimages)==len(public)==prepare.N_SOURCES*len(prepare.ARMS)*prepare.REPLICAS==2304
    sealed={
        "schema":SEALED_SCHEMA,
        "early_manifest_sha256":early_manifest_sha,
        "challenge_sha256":challenge_sha,
        "openings":preimages,
    }
    sealed_bytes=prepare.canonical(sealed)
    manifest={
        "schema":PUBLIC_SCHEMA,
        "early_manifest_sha256":early_manifest_sha,
        "challenge_sha256":challenge_sha,
        "sealed_sha256":sha256(sealed_bytes).hexdigest(),
        "attack_count":len(public),
        "source_count":prepare.N_SOURCES,
        "root_disclosed_to_attacker_before_new_commitments":True,
        "commitments":public,
    }
    for target,data in (
        (sealed_path,sealed_bytes),
        (public_path,prepare.canonical(manifest)),
    ):
        path=Path(target)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    verify_late(fixture_path,early_manifest_path,challenge_path,sealed_path,public_path)
    print("NOR29 ROOT-AWARE ATTACKS MINTED AFTER ROOT DISCLOSURE",
          len(preimages),"challenge",challenge_sha,
          "late sealed",sha256(sealed_bytes).hexdigest())


def verify_late(fixture_path,early_manifest_path,challenge_path,sealed_path,public_path):
    fixture,_=prepare.load_fixture(fixture_path)
    _,early_hash=prepare.verify_manifest(fixture_path,early_manifest_path)
    challenge_,challenge_hash=challenge.verify_schedules(
        fixture_path,early_manifest_path,challenge_path,exact=False)
    raw=Path(sealed_path).read_bytes()
    public=json.loads(Path(public_path).read_bytes())
    assert public["schema"]==PUBLIC_SCHEMA
    assert public["source_count"]==prepare.N_SOURCES
    assert public["attack_count"]==2304
    assert public["root_disclosed_to_attacker_before_new_commitments"] is True
    assert public["early_manifest_sha256"]==early_hash
    assert public["challenge_sha256"]==challenge_hash
    assert public["sealed_sha256"]==sha256(raw).hexdigest()
    sealed=json.loads(raw)
    assert sealed["schema"]==SEALED_SCHEMA
    assert sealed["early_manifest_sha256"]==early_hash
    assert sealed["challenge_sha256"]==challenge_hash
    references={(x["source_sha256"],x["auditor"],x["replica"]):
                x["commitment_sha256"] for x in public["commitments"]}
    assert len(references)==2304
    assert len(sealed["openings"])==2304
    by_source={x["public_sha256"]:x for x in challenge_["records"]}
    seen=set()
    for opening in sealed["openings"]:
        key=(opening["source_sha256"],opening["auditor"],opening["replica"])
        assert key in references and key not in seen
        seen.add(key)
        assert set(opening)=={"source_sha256","auditor","attack_class",
                             "replica","changed_indices","calibration_batch_hits","nonce_hex"}
        assert opening["attack_class"]=="early_root_aware"
        m=by_source[key[0]]["m"]
        changed=opening["changed_indices"]
        assert changed==sorted(changed)
        assert len(changed)==len(set(changed))==prepare.RADIUS
        assert all(type(i) is int and 0<=i<m for i in changed)
        assert len(opening["nonce_hex"])==64
        assert prepare.commit(opening)==references[key]
        # Confirm this set is fixed across the batch and recorded hits match.
        plans=by_source[key[0]]["schedules"][key[1]]
        hits=sum(bool(set(changed)&set(p)) for p in plans)
        assert hits==opening["calibration_batch_hits"]
        mutate=dict(opening)
        mutate["nonce_hex"]=("0" if opening["nonce_hex"][0]!="0" else "1")+opening["nonce_hex"][1:]
        assert prepare.commit(mutate)!=references[key]
    assert len(seen)==2304
    return sealed,sha256(Path(public_path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for field in ("fixture","early-manifest","challenge","sealed","public"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--make",action="store_true")
    p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    if a.make==a.verify:p.error("choose --make OR --verify")
    fn=make_late if a.make else verify_late
    fn(a.fixture,a.early_manifest,a.challenge,a.sealed,a.public)


if __name__=="__main__":main()
