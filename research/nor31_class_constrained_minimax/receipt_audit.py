"""Independent NOR-31 source, train, approximate attack and result replay.

Never imports NOR31's applied evaluator or historical train solver. Rebuilds
original RRC25 generating programs, exact 16-template public masks,
historical 7x6 payoffs, 6-class maximin optimum, independently sampled
audit challenges, trial-ID-set approximate best responses and every
source-level exact detection count.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations_with_replacement
from math import comb,isclose
from pathlib import Path
from zipfile import ZipFile
import json,random,statistics,sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import receipt_audit as original
from research.nor31_class_constrained_minimax import replay_plans as replay

ARMS=replay.ARMS
ATTACKERS=replay.ATTACKERS
LEGACY=replay.LEGACY_ATTACKS
KINDS=replay.KINDS
FAMILY_WEIGHTS=(.4,.3,.2,.1)
OLD26_SOURCE="e3f5a3df07d062d8681a39a7ddcba9a981b58af6f149f13bd372daf08ac6606f"
OLD26_FULL="a451897b780b024aac49f543833bc5e3ec32a734489d3f7a9b14a929bd11669a"
OLD30_SOURCE="9961afd81cac78b32fc5280fe54291f7db4dca0cf85cb6624560ddac420e1878"
OLD30_FULL="4189235f7cb5c14f3c7d1028abda264d893c7b5e5d2eda0c4c1de091f052b9c8"


def hash_future(y):
    return sha256(json.dumps(y,separators=(",",":")).encode()).hexdigest()


def original_archive(name,filenames):
    path=ROOT/"research"/name/"results"/(name.split("_")[0]+"_original_hosted.zip")
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        return {p:archive.read(p) for p in filenames}


def independently_fit_training(receipt):
    a=original_archive("nor26_policy_aware_audits",
                       ("nor26_fixture.json","nor26_full.json"))
    b=original_archive("nor30_minimax_audits",
                       ("nor30_fixture.json","nor30_full.json"))
    for raw,expected,length in (
        (a["nor26_fixture.json"],OLD26_SOURCE,2838554),
        (a["nor26_full.json"],OLD26_FULL,217954),
        (b["nor30_fixture.json"],OLD30_SOURCE,2284577),
        (b["nor30_full.json"],OLD30_FULL,337563)):
        assert len(raw)==length and sha256(raw).hexdigest()==expected
    s26=json.loads(a["nor26_fixture.json"])
    r26=json.loads(a["nor26_full.json"])
    s30=json.loads(b["nor30_fixture.json"])
    r30=json.loads(b["nor30_full.json"])
    assert len(s26["cases"])==len(r26["source_records"])==12
    assert len(s30["cases"])==len(r30["source_records"])==12
    x26=sorted(z["full_public_input_sha256"] for z in s26["cases"])
    x30=sorted(z["full_public_input_sha256"] for z in s30["cases"])
    assert len(set(x26+x30))==24
    assert receipt["historical_nor26_public_hashes"]==x26
    assert receipt["historical_nor30_public_hashes"]==x30
    assert receipt["historical_nor26_outcome_sha256"]==OLD26_FULL
    assert receipt["historical_nor30_outcome_sha256"]==OLD30_FULL
    count={(a,p):0 for a in LEGACY for p in ARMS}
    for record in r26["source_records"]:
        assert len(record["crossplay"])==49
        for cell in record["crossplay"]:
            if cell["auditor_policy"] in ARMS:
                assert cell["total"]==4096
                count[(cell["attacker_policy"],cell["auditor_policy"])]+=cell["detected"]
    matrix=[[count[(a,p)] for p in ARMS] for a in LEGACY]
    tpl=[0]*6
    for record in r30["source_records"]:
        cells={(c["attacker"],c["auditor"]):c for c in record["ordinary_arm_hit_counts"]}
        assert len(cells)==14*7
        for i,p in enumerate(ARMS):
            for f,w in zip(KINDS[:4],(4,3,2,1)):
                cell=cells[("template_"+f,p)]
                assert cell["total"]==4096
                tpl[i]+=w*cell["detected"]
    assert receipt["auditors"]==list(ARMS)
    assert receipt["attacker_classes"]==list(LEGACY)
    assert receipt["historical_nor26_matrix_integer_hits"]==matrix
    assert receipt["historical_nor30_template_weighted_numerators"]==tpl
    denominator=12*4096*10
    assert receipt["historical_template_denominator"]==denominator
    mh=sha256(replay.canonical({
        "attackers":LEGACY,"auditors":ARMS,"integer_hits":matrix,
        "denominator_per_cell":49152})).hexdigest()
    th=sha256(replay.canonical({
        "auditors":ARMS,"weighted_hits":tpl,
        "denominator_per_auditor":denominator})).hexdigest()
    assert receipt["historical_matrix_sha256"]==mh
    assert receipt["historical_template_sha256"]==th
    best=None
    best_key=None
    enumerated=eligible=0
    for draws in combinations_with_replacement(range(6),16):
        enumerated+=1
        w=tuple(draws.count(i) for i in range(6))
        if not 2<=w[0]<=12:continue
        if 100*sum(x*y for x,y in zip(w,tpl)) < 18*16*denominator:
            continue
        eligible+=1
        row_counts=[sum(c*x for c,x in zip(row,w)) for row in matrix]
        key=(min(row_counts),sum(row_counts),tuple(-x for x in w))
        if best is None or key>best_key:
            best,best_key=w,key
    solution=receipt["fitted_constrained"]
    assert solution["candidate_grid_enumerated"]==enumerated
    assert solution["candidate_grid_eligible"]==eligible
    assert solution["weights"]==list(best)
    assert solution["min_weighted_integer_detection"]==best_key[0]
    assert solution["sum_weighted_integer_detection"]==best_key[1]
    assert receipt["preregistration_commit"]=="7e37e8c57a68e7224a67626d2495e9bcf922a424"
    return best


def independent_source(row):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    assert rep in (24,25) and d in (512,1024)
    families=("wide_dense_small","wide_sparse_small","wide_rational_short")
    expected=(corpus.SEED+130031000+1000003*rep+
              100009*(0 if d==512 else 1)+10007*families.index(fam))
    assert row["source_seed"]==expected
    assert row["query_degree"]==3*d//2
    assert 0<=row["selected_attempt"]<100
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
    assert row["demos"]==[[list(src[:d+1]),list(y[:d+1])]
                         for src,y in demos]
    assert row["public_query_input"]==list(query)
    hrow=row["public_header"]
    header=corpus.Header(hrow["base"],hrow["delay"],
                         hrow["spacing"],hrow["amplitude"])
    assert planted[:4]==(header.base,header.delay,header.spacing,header.amplitude)
    pub=corpus.Case("nor31-independent",fam,
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),
        tuple(query),(),None)
    assert pub.program is None and pub.future==()
    assert original.public_source_hash(pub,d)==row["full_public_input_sha256"]
    mask=original.independent_native_mask(pub,header,d)
    assert mask==row["exact_native_public_mask"] and mask>0
    selected=next(j for j in range(16) if mask&(1<<j))
    pred=corpus.execute(query,original.candidate_program(header,selected))
    for j in range(16):
        if mask&(1<<j):
            assert corpus.execute(query,original.candidate_program(header,j))==pred
    assert corpus.execute(query,planted)==pred
    return mask,selected,pred,header


def supports(pred,header):
    m=len(pred)
    assert m in (769,1537)
    tail=set(range(m-96,m))
    magnitude=set(sorted(range(m),key=lambda i:(-abs(pred[i]),i))[:96])
    lag=set(range(header.delay+header.spacing,header.delay+header.spacing+96))
    r=dict(tail=tail,magnitude=magnitude,lag=lag,uniform=set(range(m)))
    r["off_template"]=set(range(m))-(tail|magnitude|lag)
    assert len(r["off_template"])>=16
    return r


def ideal_floor(m,arm):
    if arm=="uniform":
        return Fraction(1)-Fraction(comb(m-4,16),comb(m,16))
    if arm.startswith("hybrid_"):
        u={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[arm]
        t=16-u
        return Fraction(1)-Fraction(comb(m-t-4,u),comb(m-t,u))
    return Fraction(0)


def replay_all(train_path,fixture_path,result_path):
    tr_raw=Path(train_path).read_bytes()
    sraw=Path(fixture_path).read_bytes()
    rraw=Path(result_path).read_bytes()
    train=json.loads(tr_raw)
    fixture=json.loads(sraw)
    got=json.loads(rraw)
    assert train["schema"]=="nor31.historical_two_cohort_alias_collapsed_training.v1"
    weights=independently_fit_training(train)
    assert fixture["schema"]=="nor31.source_only_new_12_reps24_25.v1"
    assert got["schema"]=="nor31.fresh_source_sixclass_template_constrained_minimax.v1"
    assert got["train_receipt_sha256"]==sha256(tr_raw).hexdigest()
    assert got["test_public_fixture_sha256"]==sha256(sraw).hexdigest()
    assert got["frozen_sixclass_quanta_weights"]==list(weights)
    assert got["registered_auditor_classes"]==list(ARMS)
    assert got["registered_attacker_classes"]==list(ATTACKERS)
    assert got["historical_training_public_sources"]==24
    assert got["new_test_public_sources"]==12
    assert got["public_proposals_frozen_before_private_truth"]
    assert got["calibrated_attack_sets_frozen_before_eval_schedules"]
    assert got["train_weights_frozen_before_new_test_sources"]
    assert got["honest_external_exact_coefficient_witness_assumed"]
    rows=fixture["cases"]
    assert len(rows)==len(got["source_records"])==len(got["primary_source_pairs"])==12
    original_hashes=set(train["historical_nor26_public_hashes"]+
                       train["historical_nor30_public_hashes"])
    assert len({x["full_public_input_sha256"] for x in rows})==12
    assert all(row["full_public_input_sha256"] not in original_hashes for row in rows)
    paired=[]
    count_decisions=0
    for row,rec in zip(rows,got["source_records"],strict=True):
        pub=row["full_public_input_sha256"]
        mask,chosen,pred,head=independent_source(row)
        assert rec["public_sha256"]==pub
        assert rec["replicate"]==row["replicate"]
        assert rec["degree"]==row["degree"]
        assert rec["family"]==row["family"]
        assert rec["m"]==len(pred)
        assert rec["public_mask"]==mask
        assert rec["chosen_class"]==chosen
        assert rec["public_prediction_sha256"]==hash_future(pred)
        assert rec["sealed_true_future_sha256_postchoice"]==hash_future(pred)
        reg=supports(pred,head)
        assert rec["support_sizes"]=={k:len(v) for k,v in reg.items()}
        cal=replay.plans(reg,pub,"calibration")
        cover=replay.trial_cover(cal,len(pred))
        attacks={name:[replay.approximate_attack(cover,weights,pub,name,rep)
                       for rep in range(8)] for name in ATTACKERS}
        fresh=[replay.fresh_attack(pub,len(pred),rep) for rep in range(8)]
        templates={kind:[replay.support_attack(pub,kind,reg,rep)
                         for rep in range(8)] for kind in KINDS}
        groups={**attacks,"fresh_uniform":fresh,
                **{"template_"+k:v for k,v in templates.items()}}
        assert rec["attacks"]=={name:[list(a) for a in vals]
                               for name,vals in groups.items()}
        assert rec["unique_attacks_per_class"]=={
            name:len(set(vals)) for name,vals in attacks.items()}
        assert rec["calibration_sha256"]==sha256(replay.canonical(cal)).hexdigest()
        evaluate=replay.plans(reg,pub,"evaluation")
        assert rec["evaluation_sha256"]==sha256(replay.canonical(evaluate)).hexdigest()
        picks=[replay.realize_mixed(weights,pub,t) for t in range(512)]
        assert rec["mixed_arm_draw_counts"]==dict(Counter(picks))
        counts=Counter()
        realized=Counter({n:0 for n in groups})
        for arm in ARMS:
            for t,checked in enumerate(evaluate[arm]):
                inspected=set(checked)
                assert len(inspected)==16
                for name,variants in groups.items():
                    for a in variants:
                        detected=bool(inspected&set(a))
                        counts[(name,arm)]+=int(detected)
                        if picks[t]==arm:realized[name]+=int(detected)
                assert len(reg["uniform"]-inspected)>=4
        cells={(o["attacker"],o["auditor"]):o for o in rec["observations"]}
        assert len(cells)==len(groups)*6
        for pair,cell in cells.items():
            assert cell["detected"]==counts[pair],(pub,pair)
            assert cell["total"]==4096
        expected={name:sum(w*counts[(name,arm)]
                           for w,arm in zip(weights,ARMS)) for name in groups}
        assert rec["mixed_expected_numerator"]==expected
        assert rec["mixed_expected_denominator"]==16*4096
        assert rec["mixed_realized_detected"]==dict(realized)
        floor=sum(Fraction(w,16)*ideal_floor(len(pred),arm)
                  for w,arm in zip(weights,ARMS))
        assert floor>0
        assert rec["ideal_source_mix_floor_fraction"]==[
            floor.numerator,floor.denominator]
        assert rec["valid_target_false_rejections"]==0
        assert rec["fully_informed_attacker_detected"]==0
        assert rec["two_nonzero_shift_disagreements"]==0
        count_decisions+=len(groups)*6*4096
        den=16*4096
        selfmix=expected["mixed_best_response"]/den
        selfh=counts[("hybrid_50","hybrid_50")]/4096
        worstmix=min(expected[a]/den for a in ATTACKERS)
        worsth=min(counts[(a,"hybrid_50")]/4096 for a in ATTACKERS)
        tmplmix=sum(w*expected["template_"+a]/den
                    for a,w in zip(KINDS[:4],FAMILY_WEIGHTS))
        tmplh=sum(w*counts[("template_"+a,"hybrid_50")]/4096
                  for a,w in zip(KINDS[:4],FAMILY_WEIGHTS))
        paired.append({
            "public_sha256":pub,"degree":row["degree"],"family":row["family"],
            "mixed_self_response":selfmix,"hybrid50_self_response":selfh,
            "mixed_minus_hybrid50_self":selfmix-selfh,
            "mixed_worst_response":worstmix,"hybrid50_worst_response":worsth,
            "mixed_minus_hybrid50_worst":worstmix-worsth,
            "mixed_template_weighted":tmplmix,
            "hybrid50_template_weighted":tmplh,
            "mixed_template_fraction_of_hybrid50":tmplmix/tmplh if tmplh>0 else None,
            "mixed_off_template":expected["template_off_template"]/den,
            "hybrid50_off_template":counts[("template_off_template","hybrid_50")]/4096,
        })
    for original_row,expected in zip(got["primary_source_pairs"],paired,strict=True):
        assert original_row.keys()==expected.keys()
        for key,value in expected.items():
            if isinstance(value,float):
                assert isclose(original_row[key],value,rel_tol=0,abs_tol=1e-13),(key,original_row[key],value)
            else:
                assert original_row[key]==value
    selfwins=sum(v["mixed_minus_hybrid50_self"]>0 for v in paired)
    worstwins=sum(v["mixed_minus_hybrid50_worst"]>0 for v in paired)
    avg=statistics.mean(v["mixed_minus_hybrid50_self"] for v in paired)
    template=statistics.mean(v["mixed_template_weighted"] for v in paired)
    retention=sum(v["mixed_template_weighted"]>=.75*v["hybrid50_template_weighted"]
                  for v in paired)
    gates={
        "self_response_source_wins":selfwins>=9,
        "mean_self_response_gain":avg>=.005,
        "worst_case_source_wins":worstwins>=9,
        "template_mean_and_paired_retention":template>=.17 and retention>=9,
        "native_truth_witness_floor_controls":True,
    }
    assert got["registered_five_gates"]==gates
    assert got["primary_self_source_wins"]==selfwins
    assert got["primary_worst_source_wins"]==worstwins
    assert isclose(got["primary_mean_self_gain"],avg,rel_tol=0,abs_tol=1e-14)
    assert isclose(got["primary_template_mean"],template,rel_tol=0,abs_tol=1e-14)
    assert got["primary_template_75pct_source_count"]==retention
    assert got["registered_primary_pass"]==all(gates.values())
    return {
        "historical_training_source_units":24,
        "fresh_test_source_units":12,
        "reconstructed_16quantum_sixarm_grid":20349,
        "replayed_heldout_policy_world_decisions":count_decisions,
        "self_response_wins":selfwins,
        "worst_case_wins":worstwins,
        "template_75pct_retentions":retention,
        "source_mean_self_gain":avg,
        "source_mean_template_detection":template,
        "registered_primary_pass":all(gates.values()),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=replay_all(args.train,args.fixture,args.full)
    print("NOR31 INDEPENDENT CLASS-COLLAPSED SOURCE-LOCKED REPLAY",result)
    if args.check:
        assert result["fresh_test_source_units"]==12
        assert result["replayed_heldout_policy_world_decisions"]==(
            12*14*6*4096)


if __name__=="__main__":
    main()
