"""NOR-31: prospective six-class constrained minimax RRC25 audit game.

Frozen historical TRAIN is read from separate CI artifact before new TEST
sources exist. The shuffled_prior defender is deleted as a duplicate
uniform-in-law strategy, while its historical greedy attack stays in game.
No test-source labels or target witness values enter weight fitting.
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
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21
from research.nor24_adaptive_audits import experiment as nor24
from research.nor25_robust_audits import experiment as nor25
from research.nor30_minimax_audits import train as historical

SCHEMA="nor31.fresh_source_sixclass_template_constrained_minimax.v1"
FIXTURE_SCHEMA="nor31.source_only_new_12_reps24_25.v1"
TRAIN_SCHEMA=historical.SCHEMA.replace("nor30.nor26_only_frozen_grid_minimax.v1",
    "nor31.historical_two_cohort_alias_collapsed_training.v1")
OFFSET=130031000
REPS=(24,25)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
ARMS=("uniform","static_prior","posterior",
      "hybrid_25","hybrid_50","hybrid_75")
LEGACY_ATTACKS=("uniform","static_prior","posterior","shuffled_prior",
                "hybrid_25","hybrid_50","hybrid_75")
ATTACKERS=LEGACY_ATTACKS+("mixed_best_response",)
KINDS=("tail","magnitude","lag","uniform","off_template")
TEMPLATE_WEIGHTS=(.4,.3,.2,.1)
STREAM_CAL=202611310091
STREAM_ATTACK=202611310092
STREAM_EVAL=202611310093
STREAM_MIX=202611310094
STREAM_TEMPLATE=202611310095
CAL=256
EVAL=512
RESTARTS=8
QUANTA=16
K=16
R=4
SHIFTS=(1,4294049777)
MAX_SOURCE_ATTEMPTS=100


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def source_seed(rep,degree,family):
    if rep not in REPS or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("unregistered NOR31 source")
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(degree)+10007*FAMILIES.index(family))


def seed(*fields):
    return int.from_bytes(
        sha256("|".join(str(z) for z in fields).encode("ascii")).digest(),"big")


def new_public_source(rep,degree,family,seen):
    q=degree*3//2
    rng=random.Random(source_seed(rep,degree,family))
    for attempt in range(MAX_SOURCE_ATTEMPTS):
        planted=nor21.new_header_program(rng)
        demos=tuple((src:=nor21.public_source(rng,family,q),
                     corpus.execute(src,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        c=corpus.Case("nor31-public",family,demos,query,(),None)
        headers=corpus.possible_headers(c)
        if len(headers)!=1:continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        digest=nor21.public_digest(c,degree)
        if digest in seen:continue
        mask=nor21.native_mask(c,h,degree)
        if mask==0:continue
        preds=nor21.candidate_predictions(query,h,q)
        if len({preds[i] for i in range(16) if mask&(1<<i)})!=1:
            continue
        chosen_actual=4*corpus.OPS.index(planted[4])+corpus.OPS.index(planted[5])
        assert mask&(1<<chosen_actual)
        seen.add(digest)
        return {
            "split":"TEST","replicate":rep,"degree":degree,
            "query_degree":q,"family":family,
            "source_seed":source_seed(rep,degree,family),
            "selected_attempt":attempt,"full_public_input_sha256":digest,
            "public_header":{"base":h.base,"delay":h.delay,
                             "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(src[:degree+1]),list(y[:degree+1])]
                     for src,y in demos],
            "public_query_input":list(query),
            "exact_native_public_mask":mask
        }
    raise RuntimeError("NOR31 exhausted 100 public-admissible source attempts")


def load_training(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    assert x["schema"]==TRAIN_SCHEMA
    assert x["preregistration_commit"]=="7e37e8c57a68e7224a67626d2495e9bcf922a424"
    assert x["historical_nor26_outcome_sha256"]==historical.HIST26_FULL
    assert x["historical_nor30_outcome_sha256"]==historical.HIST30_FULL
    assert x["test_source_information_used"] is False
    assert x["auditors"]==list(ARMS) and x["attacker_classes"]==list(LEGACY_ATTACKS)
    assert x["training_public_source_count"]==24
    w=x["fitted_constrained"]["weights"]
    assert len(w)==6 and sum(w)==QUANTA and 2<=w[0]<=12
    assert x["fitted_constrained"]["historical_in_template_rate"]>=.18
    assert x["template_floor_numerator"]==18 and x["template_floor_denominator"]==100
    assert len(set(x["historical_nor26_public_hashes"]+
                   x["historical_nor30_public_hashes"]))==24
    return x,sha256(raw).hexdigest(),tuple(w)


def make_fixture(historical_receipt):
    seen=set(historical_receipt["historical_nor26_public_hashes"]+
             historical_receipt["historical_nor30_public_hashes"])
    rows=[new_public_source(rep,d,fam,seen)
          for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(rows)==12 and len(seen)==36
    return {
        "schema":FIXTURE_SCHEMA,"registered_offset":OFFSET,
        "source_replicates":list(REPS),"observation_degrees":list(DEGREES),
        "source_families":list(FAMILIES),"source_only":True,
        "private_program_and_true_target_absent":True,
        "attacks_and_witness_answers_absent":True,
        "historical_train_public_sha256":historical.HIST26_SOURCE,
        "historical_template_public_sha256":historical.HIST30_SOURCE,
        "cases":rows
    }


def load_fixture(path,training):
    raw=Path(path).read_bytes()
    o=json.loads(raw)
    assert o["schema"]==FIXTURE_SCHEMA
    assert o["registered_offset"]==OFFSET
    assert o["source_replicates"]==list(REPS)
    assert o["observation_degrees"]==list(DEGREES)
    assert o["source_families"]==list(FAMILIES)
    assert o["source_only"] and o["private_program_and_true_target_absent"]
    assert o["attacks_and_witness_answers_absent"]
    assert len(o["cases"])==12
    new={row["full_public_input_sha256"] for row in o["cases"]}
    old=set(training["historical_nor26_public_hashes"]+
            training["historical_nor30_public_hashes"])
    assert len(new)==12 and new.isdisjoint(old)
    assert {(r["replicate"],r["degree"],r["family"]) for r in o["cases"]}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    assert all(row["split"]=="TEST" and
               not any(k in row for k in ("program","future","target","attack","witness"))
               for row in o["cases"])
    return o,sha256(raw).hexdigest()


def public_proposal(row):
    hrow=row["public_header"];d=row["degree"]
    header=corpus.Header(hrow["base"],hrow["delay"],
                         hrow["spacing"],hrow["amplitude"])
    public=corpus.Case("nor31-visible",row["family"],
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),
        tuple(row["public_query_input"]),(),None)
    assert public.program is None and public.future==()
    assert nor21.public_digest(public,d)==row["full_public_input_sha256"]
    choice=nor21.certify(public,header,d,row["query_degree"],"stream16")
    assert choice["mask"]==row["exact_native_public_mask"]>0
    assert choice["unanimous"] and choice["chosen_class"] is not None
    pred=corpus.execute(public.query,nor21.program(header,choice["chosen_class"]))
    return {"pred":pred,"header":header,"mask":choice["mask"],
            "class":choice["chosen_class"],
            "future_sha256":nor24.sha_target(pred)}


def private_future_after_freeze(row,pred):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((src:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(src,planted)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,fam)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    truth=corpus.execute(query,planted)
    assert tuple(pred)==tuple(truth)
    return truth


def phase_tag(public_digest,phase):
    stream={"calibration":STREAM_CAL,"evaluation":STREAM_EVAL}[phase]
    return sha256(f"{stream}|{phase}|{public_digest}".encode("ascii")).hexdigest()


def schedules(regions,digest,phase):
    n=CAL if phase=="calibration" else EVAL
    key=phase_tag(digest,phase)
    # Seven original NOR25 policies used to generate attacker calibration,
    # with shuffled_prior solely an historical ATTACKER reference class.
    plans={arm:[] for arm in LEGACY_ATTACKS}
    for trial in range(n):
        original=nor25.policy_schedules(regions,key,trial)
        for arm in LEGACY_ATTACKS:
            coords=tuple(original[(arm,K)])
            assert len(coords)==len(set(coords))==K
            plans[arm].append(coords)
    return plans


def coverage(plans,m):
    return {arm:cover_one(plan,m) for arm,plan in plans.items()}


def cover_one(plans,m):
    bitsets=[0]*m
    for trial,checked in enumerate(plans):
        for i in checked:
            bitsets[i]|=1<<trial
    return bitsets


def greedy_attack(cov,weights,digest,name,restart):
    assert name in ATTACKERS and 0<=restart<RESTARTS
    m=len(cov["uniform"])
    indices=list(range(m))
    random.Random(seed(STREAM_ATTACK,digest,name,restart)).shuffle(indices)
    chosen=[]
    unions={arm:0 for arm in LEGACY_ATTACKS}
    for _ in range(R):
        best=None
        score=None
        for i in indices:
            if i in chosen:continue
            if name=="mixed_best_response":
                cost=sum(weights[j]*(unions[a]|cov[a][i]).bit_count()
                         for j,a in enumerate(ARMS))
            else:
                cost=(unions[name]|cov[name][i]).bit_count()
            if score is None or cost<score:
                best,score=i,cost
        assert best is not None
        chosen.append(best)
        for arm in LEGACY_ATTACKS:
            if name=="mixed_best_response" or arm==name:
                unions[arm]|=cov[arm][best]
    assert len(set(chosen))==R
    return tuple(sorted(chosen))


def fresh_attack(digest,m,restart):
    return tuple(sorted(random.Random(seed(STREAM_ATTACK,digest,"fresh",restart))
                        .sample(range(m),R)))


def template_attack(digest,kind,regions,restart):
    return tuple(sorted(random.Random(seed(STREAM_TEMPLATE,digest,kind,restart))
                        .sample(sorted(regions[kind]),R)))


def choose_mixed(weights,digest,trial):
    pick=random.Random(seed(STREAM_MIX,digest,trial)).randrange(QUANTA)
    cum=0
    for arm,w in zip(ARMS,weights):
        cum+=w
        if pick<cum:return arm
    raise AssertionError("invalid weights")


def ideal_floor(m,arm):
    if arm=="uniform":
        return Fraction(1)-Fraction(comb(m-R,K),comb(m,K))
    if arm.startswith("hybrid_"):
        u={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[arm]
        t=K-u
        return Fraction(1)-Fraction(comb(m-t-R,u),comb(m-t,u))
    return Fraction(0)


def detect(pred,truth,checked,changes):
    # Honest external integer witness under BOTH registered magnitudes.
    a=not nor24.exact_witness_accepts(pred,truth,checked,changes,SHIFTS[0])
    b=not nor24.exact_witness_accepts(pred,truth,checked,changes,SHIFTS[1])
    assert a==b==bool(set(checked)&set(changes))
    return a


def evaluate(training_path,fixture_path):
    training,train_sha,weights=load_training(training_path)
    public,fixture_sha=load_fixture(fixture_path,training)
    visible={r["full_public_input_sha256"]:public_proposal(r)
             for r in public["cases"]}
    assert len(visible)==12
    records=[]
    for row in public["cases"]:
        digest=row["full_public_input_sha256"]
        proposed=visible[digest]
        pred=proposed["pred"]
        regions=nor24.supports(pred,proposed["header"])
        m=len(pred)
        cal=schedules(regions,digest,"calibration")
        cov=coverage(cal,m)
        attacks={name:[greedy_attack(cov,weights,digest,name,rep)
                       for rep in range(RESTARTS)] for name in ATTACKERS}
        fresh=[fresh_attack(digest,m,rep) for rep in range(RESTARTS)]
        templates={kind:[template_attack(digest,kind,regions,rep)
                         for rep in range(RESTARTS)] for kind in KINDS}
        allattacks={**attacks,"fresh_uniform":fresh,
                    **{"template_"+k:v for k,v in templates.items()}}
        # Freeze all source-visible decisions AND schedule randomness first.
        evalplans=schedules(regions,digest,"evaluation")
        chosen=[choose_mixed(weights,digest,trial) for trial in range(EVAL)]
        true=private_future_after_freeze(row,pred)
        assert nor24.sha_target(true)==proposed["future_sha256"]
        counts={(name,arm):0 for name in allattacks for arm in ARMS}
        realized=Counter({name:0 for name in allattacks})
        valid_bad=0
        informed_bad=0
        for arm in ARMS:
            for trial,checked in enumerate(evalplans[arm]):
                if not nor24.exact_witness_accepts(pred,true,checked,frozenset(),1):
                    valid_bad+=1
                adaptive=nor24.informed_attack(checked,regions["uniform"],R)
                informed_bad+=int(bool(set(checked)&set(adaptive)))
                for name,variants in allattacks.items():
                    for ch in variants:
                        found=detect(pred,true,checked,ch)
                        counts[(name,arm)]+=found
                        if chosen[trial]==arm:realized[name]+=found
        assert valid_bad==informed_bad==0
        denom=QUANTA*RESTARTS*EVAL
        expected={name:sum(w*counts[(name,arm)]
                           for w,arm in zip(weights,ARMS))
                  for name in allattacks}
        floor=sum(Fraction(w,QUANTA)*ideal_floor(m,arm)
                  for w,arm in zip(weights,ARMS))
        assert floor>0
        records.append({
            "public_sha256":digest,"replicate":row["replicate"],
            "degree":row["degree"],"family":row["family"],
            "m":m,"public_mask":proposed["mask"],
            "chosen_class":proposed["class"],
            "public_prediction_sha256":proposed["future_sha256"],
            "sealed_true_future_sha256_postchoice":nor24.sha_target(true),
            "support_sizes":{k:len(v) for k,v in regions.items()},
            "calibration_sha256":sha256(canonical(cal)).hexdigest(),
            "evaluation_sha256":sha256(canonical(evalplans)).hexdigest(),
            "attacks":{name:[list(v) for v in sets]
                       for name,sets in allattacks.items()},
            "unique_attacks_per_class":{name:len(set(sets)) for name,sets in attacks.items()},
            "observations":[{"attacker":name,"auditor":arm,
                             "detected":counts[(name,arm)],"total":RESTARTS*EVAL}
                            for name in allattacks for arm in ARMS],
            "mixed_expected_numerator":expected,
            "mixed_expected_denominator":denom,
            "mixed_realized_detected":dict(realized),
            "mixed_arm_draw_counts":dict(Counter(chosen)),
            "ideal_source_mix_floor_fraction":[floor.numerator,floor.denominator],
            "valid_target_false_rejections":valid_bad,
            "fully_informed_attacker_detected":informed_bad,
            "two_nonzero_shift_disagreements":0,
        })
    paired=[]
    for rec in records:
        denom=rec["mixed_expected_denominator"]
        counts={(o["attacker"],o["auditor"]):o["detected"]
                for o in rec["observations"]}
        self_mix=rec["mixed_expected_numerator"]["mixed_best_response"]/denom
        self_baseline=counts[("hybrid_50","hybrid_50")]/(RESTARTS*EVAL)
        worst_mix=min(rec["mixed_expected_numerator"][a]/denom for a in ATTACKERS)
        worst_baseline=min(counts[(a,"hybrid_50")]/(RESTARTS*EVAL)
                           for a in ATTACKERS)
        template_mix=sum(w*rec["mixed_expected_numerator"]["template_"+a]/denom
                         for w,a in zip(TEMPLATE_WEIGHTS,KINDS[:4]))
        template_baseline=sum(w*counts[("template_"+a,"hybrid_50")]/(RESTARTS*EVAL)
                              for w,a in zip(TEMPLATE_WEIGHTS,KINDS[:4]))
        paired.append({
            "public_sha256":rec["public_sha256"],
            "degree":rec["degree"],"family":rec["family"],
            "mixed_self_response":self_mix,
            "hybrid50_self_response":self_baseline,
            "mixed_minus_hybrid50_self":self_mix-self_baseline,
            "mixed_worst_response":worst_mix,
            "hybrid50_worst_response":worst_baseline,
            "mixed_minus_hybrid50_worst":worst_mix-worst_baseline,
            "mixed_template_weighted":template_mix,
            "hybrid50_template_weighted":template_baseline,
            "mixed_template_fraction_of_hybrid50":(
                template_mix/template_baseline if template_baseline>0 else None),
            "mixed_off_template":rec["mixed_expected_numerator"]["template_off_template"]/denom,
            "hybrid50_off_template":counts[("template_off_template","hybrid_50")]/(RESTARTS*EVAL)
        })
    wins_self=sum(x["mixed_minus_hybrid50_self"]>0 for x in paired)
    wins_worst=sum(x["mixed_minus_hybrid50_worst"]>0 for x in paired)
    gains=statistics.mean(x["mixed_minus_hybrid50_self"] for x in paired)
    template_mean=statistics.mean(x["mixed_template_weighted"] for x in paired)
    template_wins=sum(x["mixed_template_weighted"] >= .75*x["hybrid50_template_weighted"]
                      for x in paired)
    gates={
        "self_response_source_wins":wins_self>=9,
        "mean_self_response_gain":gains>=.005,
        "worst_case_source_wins":wins_worst>=9,
        "template_mean_and_paired_retention":template_mean>=.17 and template_wins>=9,
        "native_truth_witness_floor_controls":all(
            x["public_mask"]>0 and
            x["public_prediction_sha256"]==x["sealed_true_future_sha256_postchoice"] and
            x["valid_target_false_rejections"]==x["fully_informed_attacker_detected"]==
            x["two_nonzero_shift_disagreements"]==0 and
            Fraction(*x["ideal_source_mix_floor_fraction"])>0
            for x in records)
    }
    return {
        "schema":SCHEMA,"train_receipt_sha256":train_sha,
        "test_public_fixture_sha256":fixture_sha,
        "historical_training_public_sources":24,
        "new_test_public_sources":12,
        "registered_auditor_classes":list(ARMS),
        "registered_attacker_classes":list(ATTACKERS),
        "frozen_sixclass_quanta_weights":list(weights),
        "calibration_trials_per_source":CAL,"eval_trials_per_source":EVAL,
        "attack_restarts_per_source_class":RESTARTS,
        "radius":R,"budget":K,"nonzero_deltas":list(SHIFTS),
        "public_proposals_frozen_before_private_truth":True,
        "calibrated_attack_sets_frozen_before_eval_schedules":True,
        "train_weights_frozen_before_new_test_sources":True,
        "honest_external_exact_coefficient_witness_assumed":True,
        "source_records":records,"primary_source_pairs":paired,
        "primary_self_source_wins":wins_self,
        "primary_worst_source_wins":wins_worst,
        "primary_mean_self_gain":gains,
        "primary_template_mean":template_mean,
        "primary_template_75pct_source_count":template_wins,
        "registered_five_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":("Synthetic finite RRC25, six distinct auditor classes "
                 "with frozen historical class-collapsed minimax and "
                 "historical template coverage constraint. External witness "
                 "authenticity and independent secret audit coins are assumptions. "
                 "Attacks/challenges are correlated within 12 test sources.")
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--make-fixture",type=Path)
    parser.add_argument("--fixture",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:parser.error("source generation and evaluation separated")
        tr,_,_=load_training(args.train)
        raw=canonical(make_fixture(tr))
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR31 NEW PUBLIC SOURCE FIXTURE",len(raw),sha256(raw).hexdigest())
        return
    if not args.fixture:parser.error("--fixture required")
    r=evaluate(args.train,args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(r))
    print("NOR31 FROZEN SIXCLASS WEIGHTS",r["frozen_sixclass_quanta_weights"])
    print("NOR31 SOURCE PAIRED",[(x["degree"],x["family"],
          round(x["mixed_minus_hybrid50_self"],6),
          round(x["mixed_minus_hybrid50_worst"],6),
          round(x["mixed_template_weighted"],6))
          for x in r["primary_source_pairs"]])
    print("NOR31 FIVE GATES",r["registered_five_gates"])
    print("NOR31 SCIENTIFIC PRIMARY PASS",r["registered_primary_pass"],
          "selfwins",r["primary_self_source_wins"],
          "worstwins",r["primary_worst_source_wins"],
          "template_source_retention",r["primary_template_75pct_source_count"])


if __name__=="__main__":
    main()
