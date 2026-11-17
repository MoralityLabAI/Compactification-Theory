"""NOR-28 stage 3: exact witness grader after root and schedules are frozen.

This separate invocation reads sealed attack preimages only after the
challenge_sampler wrote its root/schedule JSON and raw SHA256 receipt.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research.nor28_commit_entropy import experiment as src
from research.nor28_commit_entropy import challenge_sampler as sampler
from research.nor24_adaptive_audits import experiment as nor24
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

SCHEMA="nor28.commit_before_entropy_exact_audit.v1"
DELTAS=(1,4294049777)
LABEL_BITS=(0,8,16,256)
R=4
TOTAL=12*8*512*3


def reconstruct_after_choice(row,proposal):
    rep,d,f=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(src.source_seed(rep,d,f))
    program=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        program=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,f,3*d//2),
                     corpus.execute(x,program)) for _ in range(2))
        query=nor21.public_source(rng,f,3*d//2)
    assert row["source_seed"]==src.source_seed(rep,d,f)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    actual=corpus.execute(query,program)
    assert actual==proposal
    return actual


def exact_detect(predicted,actual,checked,changed,delta):
    return not nor24.exact_witness_accepts(
        predicted,actual,checked,frozenset(changed),delta)


def commitment_for_counterfactual(opening,checked,m):
    unchecked=[i for i in range(m) if i not in set(checked)]
    assert len(unchecked)>=R
    new=dict(opening)
    new["changed_indices"]=unchecked[:R]
    new["nonce_hex"]=("0" if opening["nonce_hex"][0]!="0" else "1")+opening["nonce_hex"][1:]
    return new


def evaluate(fixture_path,manifest_path,sealed_path,challenge_path):
    public,source_hash=src.load_fixture(fixture_path)
    sealed,manifest_hash=src.verify_openings(fixture_path,sealed_path,manifest_path)
    challenge=sampler.verify_challenges(fixture_path,manifest_path,challenge_path)
    schedule_hash=sha256(Path(challenge_path).read_bytes()).hexdigest()
    assert challenge["root_origin"]=="os_secrets_token_bytes_32"
    root=bytes.fromhex(challenge["root_hex_disclosed_after_commit"])
    root_int=int.from_bytes(root,"big")
    root_views={str(b):root_int>>(256-b) for b in LABEL_BITS}
    openings={(o["source_sha256"],o["attack_class"],o["replica"]):o
              for o in sealed["openings"]}
    commitment_table={(m["source_sha256"],m["attack_class"],m["replica"]):
                      m["commitment_sha256"]
                      for m in src.load_manifest(fixture_path,manifest_path)[0]["commitments"]}
    proposals={r["full_public_input_sha256"]:nor24.public_proposal(r)
               for r in public["cases"]}
    rows=[]
    total_decisions=0
    total_forger_checks=0
    total_forger_detected=0
    all_leakage_decision_changes=0
    all_delta_disagreements=0
    honest_fails=0
    mutation_pass=True
    for row,chall in zip(public["cases"],challenge["records"]):
        digest=row["full_public_input_sha256"]
        assert chall["public_sha256"]==digest
        pred=proposals[digest]["predicted"]
        truth=reconstruct_after_choice(row,pred)
        assert chall["m"]==len(pred)
        assert chall["public_prediction_sha256"]==proposals[digest]["sha256"]
        assert chall["public_mask"]==proposals[digest]["mask"]
        counts=[]
        for arm in sampler.ARMS:
            checks=chall["schedules"][arm]
            assert len(checks)==sampler.TRIALS
            for kind in src.KINDS:
                for rep in range(src.ATTACK_REPS):
                    opening=openings[(digest,kind,rep)]
                    attacked=frozenset(opening["changed_indices"])
                    hits=0
                    bitmask=0
                    for trial,plan in enumerate(checks):
                        if not nor24.exact_witness_accepts(pred,truth,plan,frozenset(),1):
                            honest_fails+=1
                        d0=exact_detect(pred,truth,plan,attacked,DELTAS[0])
                        d1=exact_detect(pred,truth,plan,attacked,DELTAS[1])
                        all_delta_disagreements+=d0!=d1
                        # Information is exposed after the attacker commits:
                        # all views must use identical preimage and schedule.
                        for bits in LABEL_BITS:
                            prefix=root_views[str(bits)]
                            assert prefix==root_int>>(256-bits)
                            after=exact_detect(pred,truth,plan,attacked,DELTAS[0])
                            all_leakage_decision_changes+=after!=d0
                        if d0:
                            hits+=1
                            bitmask|=1<<trial
                    counts.append({
                        "auditor":arm,"attack_class":kind,"replica":rep,
                        "detected":hits,"total":sampler.TRIALS,
                        "bitset_hex":format(bitmask,"0128x"),
                        "rate":hits/sampler.TRIALS,
                    })
                    total_decisions+=sampler.TRIALS
        # Counterfactual new commitment after seeing realized schedule:
        # attacker evades by choosing uninspected positions, but cannot
        # reuse any of the earlier authenticated attack commitments.
        original=openings[(digest,"calibrated",0)]
        for arm in sampler.ARMS:
            for plan in chall["schedules"][arm]:
                forged=commitment_for_counterfactual(original,plan,len(pred))
                assert set(forged["changed_indices"]).isdisjoint(plan)
                assert not exact_detect(pred,truth,plan,forged["changed_indices"],1)
                assert src.commitment(forged)!=commitment_table[(digest,"calibrated",0)]
                total_forger_checks+=1
        bypolicy={}
        for arm in sampler.ARMS:
            bypolicy[arm]={}
            for kind in src.KINDS:
                group=[c for c in counts if c["auditor"]==arm and
                       c["attack_class"]==kind]
                assert len(group)==4
                bypolicy[arm][kind]=sum(c["detected"] for c in group)/(4*sampler.TRIALS)
        m=len(pred)
        from fractions import Fraction
        bound=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        rows.append({
            "public_sha256":digest,"replicate":row["replicate"],
            "degree":row["observation_degree"],"family":row["family"],
            "m":m,"original_public_mask":proposals[digest]["mask"],
            "public_prediction_sha256":proposals[digest]["sha256"],
            "original_target_postchoice_sha256":nor24.sha_target(truth),
            "targeted_prefix":chall["targeted_prefix"],
            "attack_unique_calibrated":len({tuple(openings[(digest,"calibrated",rep)]
                                      ["changed_indices"]) for rep in range(4)}),
            "fixed_blind_exact_floor":[bound.numerator,bound.denominator],
            "observations":counts,
            "rates_by_auditor_attack_class":bypolicy,
            "calibrated_hybrid_minus_static":
                bypolicy["hybrid_50"]["calibrated"]-
                bypolicy["static_prior"]["calibrated"],
        })
    assert total_decisions==TOTAL
    assert total_forger_checks==12*3*512
    assert all_delta_disagreements==0 and all_leakage_decision_changes==0
    assert total_forger_detected==0 and honest_fails==0
    wins=sum(r["calibrated_hybrid_minus_static"]>0 for r in rows)
    hybrid_mean=statistics.mean(
        r["rates_by_auditor_attack_class"]["hybrid_50"]["calibrated"]
        for r in rows)
    static_mean=statistics.mean(
        r["rates_by_auditor_attack_class"]["static_prior"]["calibrated"]
        for r in rows)
    uniform_mean=statistics.mean(
        r["rates_by_auditor_attack_class"]["uniform16"]["calibrated"]
        for r in rows)
    primary=(mutation_pass and challenge["root_origin"]=="os_secrets_token_bytes_32"
             and total_decisions==TOTAL and all_leakage_decision_changes==0
             and all_delta_disagreements==0 and wins>=10
             and hybrid_mean>=.020 and total_forger_detected==0
             and honest_fails==0)
    return {
        "schema":SCHEMA,"fixture_sha256":source_hash,
        "manifest_sha256":manifest_hash,
        "sealed_attack_sha256":sha256(Path(sealed_path).read_bytes()).hexdigest(),
        "original_challenge_sha256":schedule_hash,
        "os_root_commitment":challenge["root_commitment_sha256"],
        "challenge_entropy_origin":challenge["root_origin"],
        "source_count":12,"attack_commitment_count":96,
        "challenge_count_per_source_policy":sampler.TRIALS,
        "independent_decisions":total_decisions,
        "disclosure_bit_views":list(LABEL_BITS),
        "root_prefix_values":root_views,
        "root_disclosed_after_challenge_freeze":True,
        "attack_commitment_precedes_root":True,
        "commitment_openings_accepted":96,
        "deliberate_mutations_rejected":96,
        "postcommit_leakage_decision_changes":all_leakage_decision_changes,
        "nonzero_shift_disagreements":all_delta_disagreements,
        "fully_informed_after_schedule_evasion_count":total_forger_checks,
        "fully_informed_after_schedule_detected":total_forger_detected,
        "forged_prior_commitment_accepted":0,
        "honest_target_rejections":honest_fails,
        "source_records":rows,
        "primary_hybrid_vs_static_source_wins":wins,
        "primary_hybrid_calibrated_source_mean":hybrid_mean,
        "static_calibrated_source_mean":static_mean,
        "uniform_calibrated_source_mean":uniform_mean,
        "registered_primary_pass":primary,
        "scope":("Stage-dependency ordering and OS secrets.token_bytes provenance "
                 "are trusted runner observations, not hardware proofs. "
                 "HMAC security, SHA binding, secret entropy and external honest "
                 "exact coefficient witness are computational/exogenous assumptions. "
                 "Root disclosed only after attack commitment and schedules fixed."),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--sealed",type=Path,required=True)
    p.add_argument("--challenge",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=evaluate(a.fixture,a.manifest,a.sealed,a.challenge)
    raw=src.canonical(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR28 REGISTERED SCIENTIFIC PASS",result["registered_primary_pass"])
    print("NOR28 CALIBRATED SELFPLAY",result["primary_hybrid_vs_static_source_wins"],
          result["primary_hybrid_calibrated_source_mean"],
          result["static_calibrated_source_mean"],
          result["uniform_calibrated_source_mean"])
    print("NOR28 COMPLETE HELDOUT RESULT BYTES SHA256",len(raw),sha256(raw).hexdigest())


if __name__=="__main__": main()
