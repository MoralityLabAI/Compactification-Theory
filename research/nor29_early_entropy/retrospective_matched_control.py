"""NOR-29 RETROSPECTIVE matched-optimization-budget control.

POSTHOC after seeing first NOR-29 registered result. This diagnostic does
not amend or enter NOR-29's original prospective PASS/FAIL criterion.

The same 32 greedy tie streams per auditor and source optimize on 512
independent HMAC schedules, using a deterministic *calibration* root
derived solely from the public source digest. Each attack then faces the
original 512 OS-rooted evaluation schedules. The attacked coordinates
are fixed across the entire evaluation batch.

Compare this matched 512-schedule calibration arm to the ORIGINAL 512-schedule
in-sample root-aware arm. Both use the same greedy rule, policy and 32 replicas.
Their difference isolates knowledge of the realized audit schedule within
this synthetic sampler, conditional on the chosen retrospective comparison.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import json
import random
import statistics
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor29_early_entropy import prepare
from research.nor29_early_entropy import challenge
from research.nor29_early_entropy import leaked_root_attacker as late
from research.nor29_early_entropy import original_archive_audit as archive
from research.nor24_adaptive_audits import experiment as n24

SCHEMA="nor29.retrospective_matched_schedule_knowledge_control.v1"
CALIBRATION_DOMAIN=b"NOR29-posthoc-independent-calibration-root-v1\x00"


def run(archive_path=None):
    archive.check()
    if archive_path is None:
        archive_path=archive.ARCHIVE
    with zipfile.ZipFile(archive_path) as z:
        fixture_raw=z.read("nor29_fixture.json")
        schedule_raw=z.read("nor29_challenges.json")
        outcome_raw=z.read("nor29_full.json")
        fixture=json.loads(fixture_raw)
        challenge_file=json.loads(schedule_raw)
        original=json.loads(outcome_raw)
    assert fixture["schema"]==prepare.FIXTURE_SCHEMA
    assert len(fixture["cases"])==24
    assert challenge_file["root_origin"]=="os_secrets_token_bytes_32"
    assert len(challenge_file["records"])==24
    assert original["registered_primary_pass"] is True
    assert original["fixture_sha256"]==sha256(fixture_raw).hexdigest()
    original_root=bytes.fromhex(challenge_file["root_hex_disclosed_after_freeze"])
    original_rows={z["public_sha256"]:z for z in original["source_records"]}
    proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
               for r in fixture["cases"]}
    paired={}
    rows=[]
    for public,source in zip(fixture["cases"],challenge_file["records"]):
        digest=public["full_public_input_sha256"]
        assert source["public_sha256"]==digest
        p=proposals[digest]
        order=prepare.rank_order(p["predicted"],p["header"],digest)
        m=len(p["predicted"])
        assert source["m"]==m
        cal_root=sha256(CALIBRATION_DOMAIN+bytes.fromhex(digest)).digest()
        assert cal_root!=original_root
        stats={}
        for arm in prepare.ARMS:
            training=[challenge.draw(cal_root,digest,arm,j,m,order)
                      for j in range(prepare.TRIALS)]
            masks=late.coverage(training,m)
            original_plans=[set(plan) for plan in source["schedules"][arm]]
            assert len(original_plans)==prepare.TRIALS
            counts=[]
            for rep in range(prepare.REPLICAS):
                changed,cal_hits=late.greedy(masks,digest,arm,rep)
                assert cal_hits==sum(bool(set(changed)&set(pl))
                                     for pl in training)
                hits=sum(bool(set(changed)&plan) for plan in original_plans)
                counts.append(hits)
            mean_rate=sum(counts)/(prepare.REPLICAS*prepare.TRIALS)
            stats[arm]={
                "matched_calibration_detected":sum(counts),
                "matched_calibration_rate":mean_rate,
                "in_sample_root_aware_rate":original_rows[digest]["rates"][arm]["early_root_aware"],
                "original_random_precommit_rate":original_rows[digest]["rates"][arm]["blind_uniform"],
                "paired_matched_minus_early_root":
                    mean_rate-original_rows[digest]["rates"][arm]["early_root_aware"],
            }
            paired.setdefault(arm,[]).append(
                stats[arm]["paired_matched_minus_early_root"])
        rows.append({
            "public_sha256":digest,"replicate":public["replicate"],
            "observation_degree":public["observation_degree"],
            "family":public["family"],"source_policy_controls":stats,
        })
    result={}
    for arm in prepare.ARMS:
        group=[x["source_policy_controls"][arm] for x in rows]
        result[arm]={
            "matched_calibration_source_mean":
                statistics.mean(x["matched_calibration_rate"] for x in group),
            "in_sample_root_aware_source_mean":
                statistics.mean(x["in_sample_root_aware_rate"] for x in group),
            "original_random_precommit_source_mean":
                statistics.mean(x["original_random_precommit_rate"] for x in group),
            "mean_matched_minus_early_root":statistics.mean(paired[arm]),
            "positive_source_pairs_matched_minus_early_root":
                sum(x>0 for x in paired[arm]),
            "median_matched_minus_early_root":statistics.median(paired[arm]),
        }
    rng=random.Random(202611290121)
    bs=sorted(statistics.mean(rng.choices(paired["hybrid_50"],k=24))
              for _ in range(10000))
    return {
        "schema":SCHEMA,
        "status":"RETROSPECTIVE_DIAGNOSTIC_AFTER_REGISTERED_NOR29_FIRST_RESULT",
        "source_count":24,"calibration_schedules_per_source_policy":512,
        "evaluation_schedules_per_source_policy":512,
        "attack_replicas_per_source_policy":32,
        "original_fixture_sha256":sha256(fixture_raw).hexdigest(),
        "original_challenge_sha256":sha256(schedule_raw).hexdigest(),
        "original_outcome_sha256":sha256(outcome_raw).hexdigest(),
        "calibration_root_domain":CALIBRATION_DOMAIN.hex(),
        "calibration_public_source_only":True,
        "calibration_root_distinct_from_original_os_root":True,
        "policy_means":result,
        "hybrid_mean_gap_descriptive_source_bootstrap_95pct":[bs[250],bs[9750]],
        "source_records":rows,
        "caution":("Posthoc matched search-budget comparator on the same original public "
                   "sources and held-out root. It is NOT part of the frozen NOR29 "
                   "scientific PASS/FAIL. A prospective fresh-source replication is "
                   "needed for confirmatory root-leakage isolation."),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    x=run()
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_bytes(prepare.canonical(x))
    print("NOR29 RETROSPECTIVE POLICY-AWARE MATCHED CALIBRATION")
    print("NOR29 MATCHED POLICY MEANS",x["policy_means"])
    print("NOR29 MATCHED HYBRID SOURCE BOOTSTRAP",
          x["hybrid_mean_gap_descriptive_source_bootstrap_95pct"])
    print("NOR29 MATCHED RESULT SHA256",
          sha256(prepare.canonical(x)).hexdigest())


if __name__=="__main__":main()
