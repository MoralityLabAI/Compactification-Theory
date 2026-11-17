"""NOR-30: source-heldout constrained minimax mixture versus policy-aware attacks.

Consumes only TRAIN receipt frozen from NOR-26 original 12-source evidence.
All 12 NOR30 TEST public sources are new (replicates 20/21), source-distinct
from historical training sources. Attack sets are fixed after CAL and before
disjoint evaluation schedules. Honest exact target coefficients are an
additional external witness; no production PRNG or crypto guarantee.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
import json
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21
from research.nor24_adaptive_audits import experiment as prior24
from research.nor25_robust_audits import experiment as robust25

SCHEMA="nor30.heldout_constrained_minimax_blind_audit.v1"
FIXTURE_SCHEMA="nor30.fresh_public_only_12_source.v1"
TRAIN_SCHEMA="nor30.nor26_only_frozen_grid_minimax.v1"
OFFSET=130030000
REPLICATES=(20,21)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
ARMS=("uniform","static_prior","posterior","shuffled_prior",
      "hybrid_25","hybrid_50","hybrid_75")
ATTACKERS=ARMS+("mixed_best_response",)
STRUCTURED=prior24.ALL_KINDS
K=16
R=4
QUANTA=16
ATTACK_REPS=8
CAL_TRIALS=256
EVAL_TRIALS=512
STREAM_CAL=202611300091
STREAM_ATTACK=202611300092
STREAM_EVAL=202611300093
STREAM_MIX=202611300094
STREAM_TEMPLATE=202611300095
DELTAS=(1,4294049777)
EVAL_WEIGHTS=dict(zip(prior24.KINDS,prior24.PRIOR_WEIGHTS))


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def hash_bytes(x):
    return sha256(x).hexdigest()


def seed(*fields):
    return int.from_bytes(sha256("|".join(map(str,fields)).encode("ascii")).digest(),"big")


def source_seed(rep,degree,family):
    if rep not in REPLICATES or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("outside registered NOR30 roster")
    return corpus.SEED+OFFSET+1000003*rep+100009*DEGREES.index(degree)+10007*FAMILIES.index(family)


def draw_source(rep,degree,family,seen):
    q=3*degree//2
    rng=random.Random(source_seed(rep,degree,family))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,family,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        public=corpus.Case("nor30-source",family,demos,query,(),None)
        headers=corpus.possible_headers(public)
        if len(headers)!=1: continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        digest=nor21.public_digest(public,degree)
        if digest in seen:continue
        mask=nor21.native_mask(public,h,degree)
        if not mask:continue
        preds=nor21.candidate_predictions(query,h,q)
        if len({preds[j] for j in range(16) if mask & (1<<j)})!=1:continue
        planted_class=4*corpus.OPS.index(planted[4])+corpus.OPS.index(planted[5])
        assert mask&(1<<planted_class)
        seen.add(digest)
        return {
            "split":"TEST","replicate":rep,"observation_degree":degree,
            "query_degree":q,"family":family,
            "source_seed":source_seed(rep,degree,family),
            "selected_attempt":attempt,
            "full_public_input_sha256":digest,
            "public_header":{"base":h.base,"delay":h.delay,
                             "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(x[:degree+1]),list(y[:degree+1])] for x,y in demos],
            "public_query_input":list(query),
            "original_public_accept_mask":mask,
        }
    raise RuntimeError("NOR30 source selection exhausted all 100 attempts")


def make_fixture():
    seen=set()
    rows=[draw_source(rep,d,fam,seen)
          for rep in REPLICATES for d in DEGREES for fam in FAMILIES]
    assert len(rows)==len(seen)==12
    return {"schema":FIXTURE_SCHEMA,"source_offset":OFFSET,
            "replicates":list(REPLICATES),"observation_degrees":list(DEGREES),
            "families":list(FAMILIES),
            "public_source_only":True,"private_target_absent":True,
            "test_attacks_absent":True,"cases":rows}


def load_fixture(path,history):
    raw=Path(path).read_bytes()
    fixture=json.loads(raw)
    assert fixture["schema"]==FIXTURE_SCHEMA and fixture["source_offset"]==OFFSET
    assert fixture["replicates"]==list(REPLICATES)
    assert fixture["public_source_only"] and fixture["private_target_absent"] and fixture["test_attacks_absent"]
    assert len(fixture["cases"])==12
    ids=[x["full_public_input_sha256"] for x in fixture["cases"]]
    assert len(set(ids))==12
    assert set(ids).isdisjoint(history["historical_public_hashes"])
    assert {(r["replicate"],r["observation_degree"],r["family"])
            for r in fixture["cases"]}=={
                (rep,d,fam) for rep in REPLICATES for d in DEGREES for fam in FAMILIES}
    for r in fixture["cases"]:
        assert r["split"]=="TEST"
        assert all(x not in r for x in ("program","future","target","attacks","witness"))
    return fixture,hash_bytes(raw)


def load_training(path):
    raw=Path(path).read_bytes()
    training=json.loads(raw)
    assert training["schema"]==TRAIN_SCHEMA
    assert training["historical_result_sha256"]=="a451897b780b024aac49f543833bc5e3ec32a734489d3f7a9b14a929bd11669a"
    assert training["historical_fixture_sha256"]=="e3f5a3df07d062d8681a39a7ddcba9a981b58af6f149f13bd372daf08ac6606f"
    assert training["test_information_used_for_training"] is False
    assert training["auditors"]==list(ARMS) and training["attacker_types"]==list(ARMS)
    assert training["historical_source_count"]==12
    assert training["grid_quanta"]==QUANTA
    weights=training["constrained_solution"]["weights"]
    assert len(weights)==7 and sum(weights)==16 and 2<=weights[0]<=12
    assert training["constrained_solution"]["minimum_uniform_quanta"]==2
    assert training["constrained_solution"]["maximum_uniform_quanta"]==12
    return training,hash_bytes(raw),weights


def phase_digest(digest,phase):
    if phase=="calibration":stream=STREAM_CAL
    elif phase=="evaluation":stream=STREAM_EVAL
    else:raise ValueError("unknown challenge phase")
    return sha256(f"{stream}|{phase}|{digest}".encode("ascii")).hexdigest()


def policies(regions,digest,phase):
    tag=phase_digest(digest,phase)
    n=CAL_TRIALS if phase=="calibration" else EVAL_TRIALS
    result={arm:[] for arm in ARMS}
    for trial in range(n):
        plans=robust25.policy_schedules(regions,tag,trial)
        for arm in ARMS:
            checked=tuple(plans[(arm,K)])
            assert len(set(checked))==K
            result[arm].append(checked)
    return result


def exposure(plans,m):
    cov=[0]*m
    for trial,checked in enumerate(plans):
        for i in checked:cov[i]|=1<<trial
    return cov


def best_response(cov_by_arm,weights,digest,attack_kind,rep):
    """Greedy approximation to minimum union exposure, with frozen tie RNG."""
    assert attack_kind in ATTACKERS and rep in range(ATTACK_REPS)
    m=len(cov_by_arm[ARMS[0]])
    rng=random.Random(seed(STREAM_ATTACK,digest,attack_kind,rep))
    positions=list(range(m))
    rng.shuffle(positions)
    unions={a:0 for a in ARMS}
    chosen=set()
    for _ in range(R):
        best=None
        best_cost=None
        for i in positions:
            if i in chosen:continue
            if attack_kind=="mixed_best_response":
                cost=sum(weights[j]*(unions[a]|cov_by_arm[a][i]).bit_count()
                         for j,a in enumerate(ARMS))
            else:
                cost=(unions[attack_kind]|cov_by_arm[attack_kind][i]).bit_count()
            if best_cost is None or cost<best_cost:
                best=i;best_cost=cost
        assert best is not None
        chosen.add(best)
        for a in ARMS:
            if attack_kind=="mixed_best_response" or a==attack_kind:
                unions[a]|=cov_by_arm[a][best]
    assert len(chosen)==R
    return tuple(sorted(chosen))


def pure_blind(digest,m,rep):
    return tuple(sorted(random.Random(seed(STREAM_ATTACK,digest,"fresh",rep)).sample(range(m),R)))


def template_attack(digest,kind,reg,rep):
    assert kind in STRUCTURED
    return tuple(sorted(random.Random(seed(STREAM_TEMPLATE,digest,kind,rep)).sample(sorted(reg[kind]),R)))


def mix_choose(weights,digest,trial):
    rng=random.Random(seed(STREAM_MIX,digest,trial))
    n=rng.randrange(QUANTA)
    total=0
    for j,w in enumerate(weights):
        total+=w
        if n<total:return ARMS[j]
    raise AssertionError("invalid integer mixture")


def freeze_private_program(row,predicted):
    """Only called after the public proposal and calibration/eval schedules."""
    rep,d,fam=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((src:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(src,planted)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,fam)
    assert row["demos"]==[[list(src[:d+1]),list(y[:d+1])] for src,y in demos]
    assert row["public_query_input"]==list(query)
    truth=corpus.execute(query,planted)
    assert truth==predicted
    return truth


def ideal_floor(m,arm):
    if arm=="uniform":return Fraction(1)-Fraction(comb(m-R,K),comb(m,K))
    if arm.startswith("hybrid_"):
        u={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[arm]
        t=K-u
        return Fraction(1)-Fraction(comb(m-t-R,u),comb(m-t,u))
    return Fraction(0)


def verify_hit(pred,true,checked,changed):
    """Use the ORIGINAL exact integer coefficient witness at both deltas.

    An independent set-intersection agrees iff the target is altered only
    at fixed 'changed' coordinates with nonzero registered integer delta.
    """
    indices=frozenset(checked)
    altered=frozenset(changed)
    intersection=bool(indices & altered)
    a=not prior24.exact_witness_accepts(pred,true,checked,altered,DELTAS[0])
    b=not prior24.exact_witness_accepts(pred,true,checked,altered,DELTAS[1])
    assert a==b==intersection
    return a


def evaluate(fixture_path,training_path):
    training,train_sha,weights=load_training(training_path)
    fixture,fixture_sha=load_fixture(fixture_path,training)
    # All public proposals frozen before private program regeneration.
    proposals={row["full_public_input_sha256"]:prior24.public_proposal(row)
               for row in fixture["cases"]}
    assert len(proposals)==12
    outputs=[]
    for row in fixture["cases"]:
        digest=row["full_public_input_sha256"]
        prop=proposals[digest]
        m=len(prop["predicted"])
        reg=prior24.supports(prop["predicted"],prop["header"])
        cal=policies(reg,digest,"calibration")
        cov={a:exposure(cal[a],m) for a in ARMS}
        attacks={at:[best_response(cov,weights,digest,at,rep)
                      for rep in range(ATTACK_REPS)] for at in ATTACKERS}
        fresh=[pure_blind(digest,m,rep) for rep in range(ATTACK_REPS)]
        structural={f:[template_attack(digest,f,reg,rep)
                       for rep in range(ATTACK_REPS)] for f in STRUCTURED}
        # Draw all TEST audit plans before accessing true future.
        evaluations=policies(reg,digest,"evaluation")
        chosen=[mix_choose(weights,digest,t) for t in range(EVAL_TRIALS)]
        truth=freeze_private_program(row,prop["predicted"])
        original_program_valid=prior24.sha_target(truth)==prop["sha256"]
        assert original_program_valid
        assert prop["mask"]==row["original_public_accept_mask"]

        attack_groups={**attacks,"fresh_uniform":fresh,
                       **{"template_"+k:v for k,v in structural.items()}}
        counts={(name,a):0 for name in attack_groups for a in ARMS}
        realized=Counter({name:0 for name in attack_groups})
        legitimate=0
        informed=0
        for arm in ARMS:
            for t,checked in enumerate(evaluations[arm]):
                for group,variants in attack_groups.items():
                    for attack in variants:
                        hit=verify_hit(prop["predicted"],truth,checked,attack)
                        counts[(group,arm)]+=hit
                        if chosen[t]==arm:
                            realized[group]+=hit
                if not prior24.exact_witness_accepts(
                        prop["predicted"],truth,checked,frozenset(),1):
                    legitimate+=1
                # Informed attacker sees a realized full inspection schedule,
                # so r unchecked targets can be altered with zero detection.
                revised=prior24.informed_attack(checked,reg["uniform"],R)
                informed+=bool(frozenset(checked)&revised)
        assert legitimate==informed==0
        for name in attack_groups:
            for a in ARMS:
                assert 0<=counts[(name,a)]<=EVAL_TRIALS*ATTACK_REPS
        expectation={name:sum(weights[j]*counts[(name,a)]
                             for j,a in enumerate(ARMS))
                     for name in attack_groups}
        denom=QUANTA*EVAL_TRIALS*ATTACK_REPS
        theoretical_floor=sum(Fraction(w,QUANTA)*ideal_floor(m,a)
                             for a,w in zip(ARMS,weights))
        assert theoretical_floor>0
        outputs.append({
            "public_sha256":digest,"replicate":row["replicate"],
            "degree":row["observation_degree"],"family":row["family"],
            "m":m,"public_mask":prop["mask"],"chosen_class":prop["chosen_class"],
            "public_prediction_sha256":prop["sha256"],
            "sealed_truth_sha256_postchoice":prior24.sha_target(truth),
            "support_sizes":{k:len(v) for k,v in reg.items()},
            "calibration_schedule_sha256":hash_bytes(canonical(cal)),
            "evaluation_schedule_sha256":hash_bytes(canonical(evaluations)),
            "attacks":{k:[list(v) for v in variants]
                       for k,variants in attack_groups.items()},
            "mixed_selected_arm_counts":dict(Counter(chosen)),
            "mixed_expected_numerator":expectation,
            "mixed_expected_denominator":denom,
            "mixed_realized_detected":dict(realized),
            "ordinary_arm_hit_counts":[
                {"attacker":name,"auditor":arm,"detected":counts[(name,arm)],
                 "total":EVAL_TRIALS*ATTACK_REPS}
                for name in attack_groups for arm in ARMS],
            "mixture_exact_floor_fraction":[theoretical_floor.numerator,
                                            theoretical_floor.denominator],
            "legitimate_rejections":legitimate,
            "informed_attacker_detections":informed,
            "delta_decision_disagreements":0,
        })
    rows=[]
    for s in outputs:
        count={(x["attacker"],x["auditor"]):x["detected"]
               for x in s["ordinary_arm_hit_counts"]}
        denom=s["mixed_expected_denominator"]
        self_mixed=s["mixed_expected_numerator"]["mixed_best_response"]/denom
        self_hybrid=count[("hybrid_50","hybrid_50")]/(EVAL_TRIALS*ATTACK_REPS)
        worst_mixed=min(s["mixed_expected_numerator"][at]/denom
                        for at in ATTACKERS)
        worst_hybrid=min(count[(at,"hybrid_50")]/(EVAL_TRIALS*ATTACK_REPS)
                         for at in ATTACKERS)
        in_mix=sum(EVAL_WEIGHTS[f]*
            s["mixed_expected_numerator"]["template_"+f]/denom
            for f in prior24.KINDS)
        in_hybrid=sum(EVAL_WEIGHTS[f]*
            count[("template_"+f,"hybrid_50")]/(EVAL_TRIALS*ATTACK_REPS)
            for f in prior24.KINDS)
        rows.append({
            "public_sha256":s["public_sha256"],"degree":s["degree"],
            "family":s["family"],
            "mixed_self_response":self_mixed,"hybrid50_self_response":self_hybrid,
            "mixed_minus_hybrid50_self_response":self_mixed-self_hybrid,
            "mixed_worst_over_attacker_classes":worst_mixed,
            "hybrid50_worst_over_attacker_classes":worst_hybrid,
            "mixed_minus_hybrid50_worst":worst_mixed-worst_hybrid,
            "mixed_in_template_weighted":in_mix,
            "hybrid50_in_template_weighted":in_hybrid,
            "mixed_off_template":s["mixed_expected_numerator"]["template_off_template"]/denom,
            "hybrid50_off_template":count[("template_off_template","hybrid_50")]/(EVAL_TRIALS*ATTACK_REPS),
        })
    wins_self=sum(x["mixed_minus_hybrid50_self_response"]>0 for x in rows)
    wins_worst=sum(x["mixed_minus_hybrid50_worst"]>0 for x in rows)
    gain=statistics.mean(x["mixed_minus_hybrid50_self_response"] for x in rows)
    positive_floor=all(Fraction(*x["mixture_exact_floor_fraction"])>0 for x in outputs)
    mandatory=all(x["legitimate_rejections"]==x["informed_attacker_detections"]==
                  x["delta_decision_disagreements"]==0 for x in outputs)
    success=(wins_self>=9 and gain>=0.005 and wins_worst>=9
             and positive_floor and mandatory)
    return {
        "schema":SCHEMA,"fixture_sha256":fixture_sha,"training_receipt_sha256":train_sha,
        "historical_training_result_sha256":training["historical_result_sha256"],
        "training_source_count":12,"heldout_source_count":12,
        "mixture_weights":dict(zip(ARMS,weights)),
        "mixture_quanta":QUANTA,"attack_class_count":len(ATTACKERS),
        "policy_count":len(ARMS),
        "calibration_trials_per_source_policy":CAL_TRIALS,
        "evaluation_trials_per_source_policy":EVAL_TRIALS,
        "attacks_per_class":ATTACK_REPS,
        "radius":R,"budget":K,"deltas":list(DELTAS),
        "test_public_proposals_frozen_before_private_truth":True,
        "test_attacks_frozen_before_evaluation_schedules":True,
        "training_weights_frozen_before_test_source_creation":True,
        "authentic_external_exact_witness_assumed":True,
        "source_records":outputs,"primary_source_pairs":rows,
        "primary_wins_against_hybrid50_self_response":wins_self,
        "primary_wins_against_hybrid50_worst":wins_worst,
        "primary_mean_self_response_gain":gain,
        "positive_conditional_mixture_floor_verified":positive_floor,
        "mandatory_exact_controls_pass":mandatory,
        "registered_primary_pass":success,
        "scope":("Historical NOR26 12-source integer game selects a constrained "
                 "discretized minimax mixture. Heldout sources are fresh; "
                 "policy-aware greedy attackers see policy/calibration only. "
                 "Synthetic deterministic challenge seeds are not secrets, "
                 "and actual target witness authenticity is assumed."),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--training",type=Path,required=True)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:p.error("source generation stage must be separate")
        training,sha,w=load_training(args.training)
        data=canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(data)
        print("NOR30 FIRST PUBLIC SOURCE FIXTURE",len(data),hash_bytes(data))
        print("NOR30 FROZEN TRAIN RECEIPT",sha,w)
        return
    if not args.fixture:p.error("--fixture required")
    result=evaluate(args.fixture,args.training)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(result))
    print("NOR30 FROZEN MIXTURE",result["mixture_weights"])
    print("NOR30 SOURCE PAIRS",[
        (x["degree"],x["family"],
         round(x["mixed_minus_hybrid50_self_response"],6),
         round(x["mixed_minus_hybrid50_worst"],6))
        for x in result["primary_source_pairs"]])
    print("NOR30 PRIMARY",result["registered_primary_pass"],
          "selfwins",result["primary_wins_against_hybrid50_self_response"],
          "worstwins",result["primary_wins_against_hybrid50_worst"],
          "selfmean",round(result["primary_mean_self_response_gain"],7))


if __name__=="__main__":
    main()
