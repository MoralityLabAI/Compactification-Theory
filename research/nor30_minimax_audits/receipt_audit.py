"""NOR-30 independent original-interpreter source and minimax-game replay.

No import of NOR30 experiment, model fitter or original audit scorer.
Recomputes historical 7x7 counts, 16-quantum maximin selection, original
RRC25 hidden programs, seven audit plans and greedy attacks using trial-ID
sets (the evaluator uses bitsets), full cross-play and source-level gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations_with_replacement
from math import comb, isclose
from pathlib import Path
from zipfile import ZipFile
import json,random,statistics,sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import receipt_audit as original
from research.nor30_minimax_audits import replay_plans as replay

ARMS=replay.ARMS
ATTACKERS=replay.ATTACKERS
STRUCTURED=("tail","magnitude","lag","uniform","off_template")
MIX_WEIGHTS={"tail":.4,"magnitude":.3,"lag":.2,"uniform":.1}
R=4
K=16
N=512
HIST_RESULT_SHA="a451897b780b024aac49f543833bc5e3ec32a734489d3f7a9b14a929bd11669a"
HIST_SOURCE_SHA="e3f5a3df07d062d8681a39a7ddcba9a981b58af6f149f13bd372daf08ac6606f"


def hash_future(y):
    return sha256(json.dumps(y,separators=(",",":")).encode()).hexdigest()


def independent_training(receipt):
    archive=ROOT/"research/nor26_policy_aware_audits/results/nor26_original_hosted.zip"
    with ZipFile(archive) as z:
        assert z.testzip() is None
        src=z.read("nor26_fixture.json")
        full=z.read("nor26_full.json")
    assert sha256(src).hexdigest()==HIST_SOURCE_SHA
    assert sha256(full).hexdigest()==HIST_RESULT_SHA
    assert len(src)==2838554 and len(full)==217954
    s=json.loads(src);f=json.loads(full)
    assert len(s["cases"])==len(f["source_records"])==12
    counts={(a,b):0 for a in ARMS for b in ARMS}
    for source in f["source_records"]:
        for cell in source["crossplay"]:
            assert cell["total"]==4096
            counts[(cell["attacker_policy"],cell["auditor_policy"])]+=cell["detected"]
    matrix=[[counts[(a,b)] for b in ARMS] for a in ARMS]
    assert receipt["historical_integer_hit_matrix"]==matrix
    assert receipt["historical_public_hashes"]==sorted(
        x["full_public_input_sha256"] for x in s["cases"])
    assert receipt["historical_result_sha256"]==HIST_RESULT_SHA
    assert receipt["historical_fixture_sha256"]==HIST_SOURCE_SHA
    # Independent enumeration via multisets of 16 auditor selections.
    choices=0;best=None;best_key=None;all_best=None;all_key=None
    for selections in combinations_with_replacement(range(7),16):
        weights=tuple(selections.count(j) for j in range(7))
        scores=tuple(sum(w*x for w,x in zip(weights,row)) for row in matrix)
        key=(min(scores),sum(scores),tuple(-w for w in weights))
        if all_key is None or key>all_key:
            all_key=key;all_best=weights
        if 2<=weights[0]<=12:
            choices+=1
            if best_key is None or key>best_key:
                best_key=key;best=weights
    assert choices==receipt["constrained_solution"]["number_of_grid_candidates"]
    assert list(best)==receipt["constrained_solution"]["weights"]
    assert list(all_best)==receipt["unconstrained_diagnostic_solution"]["weights"]
    assert best_key[0]==receipt["constrained_solution"]["minimum_weighted_hits"]
    assert receipt["grid_quanta"]==16
    return best


def independent_source(row):
    rep,d,fam=row["replicate"],row["observation_degree"],row["family"]
    assert rep in (20,21) and d in (512,1024)
    family_order=("wide_dense_small","wide_sparse_small","wide_rational_short")
    expected=(corpus.SEED+130030000+1000003*rep+
              100009*(0 if d==512 else 1)+10007*family_order.index(fam))
    assert expected==row["source_seed"] and 0<=row["selected_attempt"]<100
    rng=random.Random(expected)
    planted=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        planted=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((src:=original.independent_source(rng,fam,3*d//2),
                     corpus.execute(src,planted)) for _ in range(2))
        query=original.independent_source(rng,fam,3*d//2)
    assert planted is not None
    assert row["demos"]==[[list(src[:d+1]),list(y[:d+1])] for src,y in demos]
    assert row["public_query_input"]==list(query)
    header_row=row["public_header"]
    h=corpus.Header(header_row["base"],header_row["delay"],
                    header_row["spacing"],header_row["amplitude"])
    assert planted[:4]==(h.base,h.delay,h.spacing,h.amplitude)
    pub=corpus.Case("nor30-independent-public",fam,
                    tuple((tuple(src),tuple(y)) for src,y in row["demos"]),
                    tuple(query),(),None)
    assert pub.program is None and pub.future==()
    assert original.public_source_hash(pub,d)==row["full_public_input_sha256"]
    mask=original.independent_native_mask(pub,h,d)
    assert mask==row["original_public_accept_mask"] and mask>0
    chosen=next(j for j in range(16) if mask&(1<<j))
    predicted=corpus.execute(query,original.candidate_program(h,chosen))
    for j in range(16):
        if mask&(1<<j):
            assert corpus.execute(query,original.candidate_program(h,j))==predicted
    assert corpus.execute(query,planted)==predicted
    return mask,chosen,predicted,h


def support_regions(pred,h):
    m=len(pred)
    assert m in (769,1537)
    top=set(sorted(range(m),key=lambda i:(-abs(pred[i]),i))[:96])
    tail=set(range(m-96,m))
    lag=set(range(h.delay+h.spacing,h.delay+h.spacing+96))
    reg=dict(tail=tail,magnitude=top,lag=lag,uniform=set(range(m)))
    reg["off_template"]=set(range(m))-(tail|top|lag)
    assert len(reg["off_template"])>=16
    return reg


def exact_floor(m,w):
    score=Fraction()
    for j,a in enumerate(ARMS):
        if a=="uniform":
            p=Fraction(1)-Fraction(comb(m-R,K),comb(m,K))
        elif a.startswith("hybrid_"):
            u={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[a]
            t=K-u
            p=Fraction(1)-Fraction(comb(m-t-R,u),comb(m-t,u))
        else:p=Fraction()
        score+=Fraction(w[j],16)*p
    return score


def replay_original(train_path,fixture_path,out_path):
    train_raw=Path(train_path).read_bytes()
    fixture_raw=Path(fixture_path).read_bytes()
    outcome_raw=Path(out_path).read_bytes()
    training=json.loads(train_raw);fixture=json.loads(fixture_raw);out=json.loads(outcome_raw)
    weights=independent_training(training)
    assert out["schema"]=="nor30.heldout_constrained_minimax_blind_audit.v1"
    assert fixture["schema"]=="nor30.fresh_public_only_12_source.v1"
    assert out["fixture_sha256"]==sha256(fixture_raw).hexdigest()
    assert out["training_receipt_sha256"]==sha256(train_raw).hexdigest()
    assert out["mixture_weights"]==dict(zip(ARMS,weights))
    assert out["training_source_count"]==out["heldout_source_count"]==12
    assert out["test_attacks_frozen_before_evaluation_schedules"]
    assert out["test_public_proposals_frozen_before_private_truth"]
    assert out["training_weights_frozen_before_test_source_creation"]
    assert out["authentic_external_exact_witness_assumed"]
    rows=fixture["cases"]
    assert len(rows)==len(out["source_records"])==len(out["primary_source_pairs"])==12
    assert len({r["full_public_input_sha256"] for r in rows})==12
    assert all(r["split"]=="TEST" for r in rows)
    assert {r["full_public_input_sha256"] for r in rows}.isdisjoint(
        training["historical_public_hashes"])
    primary=[]
    total_decisions=0
    for row,record in zip(rows,out["source_records"],strict=True):
        pub=row["full_public_input_sha256"]
        mask,chosen,pred,h=independent_source(row)
        m=len(pred)
        assert record["public_sha256"]==pub
        assert record["replicate"]==row["replicate"]
        assert record["degree"]==row["observation_degree"]
        assert record["family"]==row["family"]
        assert record["m"]==m and record["public_mask"]==mask
        assert record["chosen_class"]==chosen
        assert record["public_prediction_sha256"]==hash_future(pred)
        assert record["sealed_truth_sha256_postchoice"]==hash_future(pred)
        reg=support_regions(pred,h)
        assert record["support_sizes"]=={k:len(v) for k,v in reg.items()}
        cal=replay.generate_schedules(reg,pub,"calibration")
        cv=replay.cover_by_policy(cal,m)
        attacks={a:[replay.attack_greedy_from_cover(cv,weights,pub,a,rep)
                    for rep in range(8)] for a in ATTACKERS}
        groups={
            **attacks,
            "fresh_uniform":[replay.independent_blind(pub,m,rep) for rep in range(8)],
            **{"template_"+f:[replay.template(pub,f,reg,rep) for rep in range(8)]
               for f in STRUCTURED},
        }
        assert record["attacks"]=={a:[list(x) for x in v] for a,v in groups.items()}
        assert record["calibration_schedule_sha256"]==sha256(
            replay.canonical(cal)).hexdigest()
        evaluation=replay.generate_schedules(reg,pub,"evaluation")
        assert record["evaluation_schedule_sha256"]==sha256(
            replay.canonical(evaluation)).hexdigest()
        choices=[replay.chosen_mixed_arm(weights,pub,t) for t in range(512)]
        assert record["mixed_selected_arm_counts"]==dict(Counter(choices))
        hit=Counter()
        mixed=Counter()
        for arm in ARMS:
            for t,plan in enumerate(evaluation[arm]):
                observed=set(plan)
                assert len(observed)==16
                for name,attack_list in groups.items():
                    for altered in attack_list:
                        detected=bool(observed&set(altered))
                        hit[(name,arm)]+=detected
                        if choices[t]==arm:mixed[name]+=detected
                assert len(reg["uniform"]-observed)>=R
        for z in record["ordinary_arm_hit_counts"]:
            key=(z["attacker"],z["auditor"])
            assert z["detected"]==hit[key] and z["total"]==4096,key
        denom=16*4096
        exp={name:sum(weights[j]*hit[(name,arm)]
                      for j,arm in enumerate(ARMS)) for name in groups}
        assert record["mixed_expected_denominator"]==denom
        assert record["mixed_expected_numerator"]==exp
        assert record["mixed_realized_detected"]==dict(mixed)
        fl=exact_floor(m,weights)
        assert record["mixture_exact_floor_fraction"]==[fl.numerator,fl.denominator]
        assert fl>0
        assert record["legitimate_rejections"]==record["informed_attacker_detections"]==0
        assert record["delta_decision_disagreements"]==0
        total_decisions+=len(groups)*7*8*512
        selfmix=exp["mixed_best_response"]/denom
        selfh=hit[("hybrid_50","hybrid_50")]/4096
        worst_mix=min(exp[a]/denom for a in ATTACKERS)
        worst_h=min(hit[(a,"hybrid_50")]/4096 for a in ATTACKERS)
        im=sum(MIX_WEIGHTS[f]*exp["template_"+f]/denom for f in MIX_WEIGHTS)
        ih=sum(MIX_WEIGHTS[f]*hit[("template_"+f,"hybrid_50")]/4096
               for f in MIX_WEIGHTS)
        primary.append({
            "public_sha256":pub,"degree":row["observation_degree"],"family":row["family"],
            "mixed_self_response":selfmix,"hybrid50_self_response":selfh,
            "mixed_minus_hybrid50_self_response":selfmix-selfh,
            "mixed_worst_over_attacker_classes":worst_mix,
            "hybrid50_worst_over_attacker_classes":worst_h,
            "mixed_minus_hybrid50_worst":worst_mix-worst_h,
            "mixed_in_template_weighted":im,"hybrid50_in_template_weighted":ih,
            "mixed_off_template":exp["template_off_template"]/denom,
            "hybrid50_off_template":hit[("template_off_template","hybrid_50")]/4096,
        })
    for got,expected in zip(out["primary_source_pairs"],primary,strict=True):
        assert got.keys()==expected.keys()
        for key in expected:
            if isinstance(expected[key],float):
                assert isclose(got[key],expected[key],rel_tol=0,abs_tol=1e-13),(key,got[key],expected[key])
            else:assert got[key]==expected[key]
    wu=sum(x["mixed_minus_hybrid50_self_response"]>0 for x in primary)
    ww=sum(x["mixed_minus_hybrid50_worst"]>0 for x in primary)
    mean=statistics.mean(x["mixed_minus_hybrid50_self_response"] for x in primary)
    criterion=wu>=9 and ww>=9 and mean>=.005
    assert out["registered_primary_pass"]==criterion
    assert out["primary_wins_against_hybrid50_self_response"]==wu
    assert out["primary_wins_against_hybrid50_worst"]==ww
    assert isclose(out["primary_mean_self_response_gain"],mean,rel_tol=0,abs_tol=1e-14)
    return {"historical_train_source_count":12,"prospective_test_source_count":12,
            "full_policy_decisions_replayed":total_decisions,
            "candidate_training_grid_replayed":74613,
            "primary_pass":criterion,"mean_self_response_gain":mean,
            "wins_against_hybrid50_self":wu,
            "wins_against_hybrid50_worst":ww}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--training",type=Path,required=True)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    res=replay_original(args.training,args.fixture,args.full)
    print("NOR30 INDEPENDENT MIXED-AUDIT GAME REPLAY",res)
    if args.check:assert res["full_policy_decisions_replayed"]==12*(
        8+1+5)*7*8*512


if __name__=="__main__":
    main()
