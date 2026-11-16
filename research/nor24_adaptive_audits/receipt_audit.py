"""NOR-24 independent original-interpreter and full trial replay.

The grader imports original RRC25 semantics and a separately written
challenge planner, never the NOR-24 experiment/heldout policy implementation.
"""
from __future__ import annotations
import argparse,json,random,statistics,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as base
from research.nor21_degree_scaling import receipt_audit as original
from research.nor24_adaptive_audits import replay_schedules as independent

KINDS=independent.F
ALL=independent.K
ARMS=independent.A
WEIGHTS=(.40,.30,.20,.10)
RADII=(1,4,16)
BUDGETS=(4,16,64)
N=1024

def hash_future(y):
    return sha256(json.dumps(y,separators=(",",":")).encode()).hexdigest()

def regenerate(row):
    rep,d,family=row["replicate"],row["observation_degree"],row["family"]
    assert rep in range(4) and d in (512,1024)
    seed=(base.SEED+94324000+1000003*rep+
          100009*(0 if d==512 else 1)+10007*(
          ("wide_dense_small","wide_sparse_small","wide_rational_short").index(family)))
    assert seed==row["source_seed"]
    rng=random.Random(seed)
    planted=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        planted=(rng.choice(base.OPS),rng.choice(base.DELAYS),
                 rng.choice(base.SPACINGS),rng.choice(base.AMPLITUDES),
                 rng.choice(base.OPS),rng.choice(base.OPS))
        demos=tuple((src:=original.independent_source(rng,family,3*d//2),
                    base.execute(src,planted)) for _ in range(2))
        query=original.independent_source(rng,family,3*d//2)
    assert planted is not None
    assert row["demos"]==[[list(s[:d+1]),list(y[:d+1])] for s,y in demos]
    assert row["public_query_input"]==list(query)
    h=row["public_header"]
    header=base.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    assert planted[:4]==(header.base,header.delay,header.spacing,header.amplitude)
    public=base.Case("nor24-independent",family,
        tuple((tuple(s),tuple(y)) for s,y in row["demos"]),tuple(query),(),None)
    assert public.program is None and public.future==()
    assert original.public_source_hash(public,d)==row["full_public_input_sha256"]
    mask=original.independent_native_mask(public,header,d)
    assert mask==row["original_public_accept_mask"] and mask>0
    chosen=next(j for j in range(16) if mask&(1<<j))
    predicted=base.execute(query,original.candidate_program(header,chosen))
    assert all(base.execute(query,original.candidate_program(header,j))==predicted
               for j in range(16) if mask&(1<<j))
    assert base.execute(query,planted)==predicted
    return (header,mask,chosen,predicted)

def fit(rows,sources):
    counts=Counter({f:0 for f in KINDS})
    registered=[]
    for row in rows:
        assert row["split"]=="TRAIN"
        key=row["full_public_input_sha256"]
        header,_,_,pred=sources[key]
        supports=independent.regions(pred,header)
        for episode in range(64):
            rng=random.Random(independent.seed(202611240071,key,episode))
            v=rng.random()
            acc=0.
            kind=KINDS[-1]
            for fam,w in zip(KINDS,WEIGHTS):
                acc+=w
                if v<acc:
                    kind=fam;break
            coords=tuple(sorted(rng.sample(sorted(supports[kind]),4)))
            registered.append([key,episode,kind,coords])
            counts[kind]+=1
    assert sum(counts.values())==768
    pi={f:(counts[f]+1)/772 for f in KINDS}
    digest=sha256(json.dumps(registered,separators=(",",":")).encode()).hexdigest()
    return counts,pi,digest

def audit(fixture_bytes,result_bytes):
    fixture=json.loads(fixture_bytes)
    result=json.loads(result_bytes)
    assert fixture["schema"]=="nor24.public_source_only_24_disjoint_train_test.v1"
    assert fixture["new_source_offset"]==94324000
    assert fixture["public_only"] and fixture["private_program_and_target_absent"]
    assert fixture["heldout_attack_locations_absent"]
    assert result["schema"]=="nor24.train_only_sequential_structured_coordinate_audit.v1"
    assert result["fixture_sha256"]==sha256(fixture_bytes).hexdigest()
    assert result["learned_prior_from_train_only"]
    assert result["all_public_proposals_frozen_before_private_target_access"]
    assert result["trusted_witness_values_are_new_external_information"]
    rows=fixture["cases"]
    assert len(rows)==result["public_source_count"]==24
    assert len({r["full_public_input_sha256"] for r in rows})==24
    assert Counter(r["split"] for r in rows)=={"TRAIN":12,"TEST":12}
    generated={r["full_public_input_sha256"]:regenerate(r) for r in rows}
    counts,prior,train_sha=fit([r for r in rows if r["split"]=="TRAIN"],generated)
    assert result["train_family_counts"]==dict(counts)
    assert result["fitted_prior"]==prior
    assert result["train_attack_position_receipt_sha256"]==train_sha
    assert result["weighted_evaluation_mixture"]==dict(zip(KINDS,WEIGHTS))
    test=[r for r in rows if r["split"]=="TEST"]
    assert len(test)==len(result["source_records"])==12
    primary=[]
    number=0
    for row,record in zip(test,result["source_records"],strict=True):
        key=row["full_public_input_sha256"]
        header,mask,chosen,pred=generated[key]
        assert record["split"]=="TEST"
        assert record["public_sha256"]==key
        assert record["public_mask"]==mask
        assert record["chosen_class"]==chosen
        assert record["replicate"]==row["replicate"]
        assert record["observation_degree"]==row["observation_degree"]
        assert record["family"]==row["family"]
        assert record["public_prediction_sha256"]==hash_future(pred)
        assert record["sealed_true_future_sha256_postchoice"]==hash_future(pred)
        support=independent.regions(pred,header)
        assert record["support_sizes"]=={f:len(support[f]) for f in ALL}
        attacks={(f,r):independent.attack(key,f,r,support)
                 for f in ALL for r in RADII}
        assert record["attack_positions"]=={
            f"{f}|{r}":sorted(attacks[(f,r)]) for f in ALL for r in RADII}
        count=Counter()
        for trial in range(N):
            schedule=independent.plan(support,prior,key,trial)
            number+=1
            for kind in ALL:
                for arm in ARMS:
                    path=schedule[arm] if arm!="oracle_support" else schedule[(arm,kind)]
                    assert len(path)==len(set(path))==64
                    for k in BUDGETS:
                        checked=set(path[:k])
                        assert len(checked)==k
                        for radius in RADII:
                            count[(kind,radius,k,arm)]+=int(bool(
                                checked&attacks[(kind,radius)]))
                            assert len(support[kind]-checked)>=radius
        assert record["schedule_generation_ns_total"]>0
        entries={(z["kind"],z["radius"],z["budget"],z["policy"]):z
                 for z in record["observations"]}
        assert len(entries)==5*3*3*5
        for index,z in entries.items():
            n=count[index]
            assert z["detected"]==n,(key,index)
            assert z["missed"]==N-n
            assert z["informed_detected"]==0
            assert z["informed_false_accept"]==N
        rates={arm:sum(WEIGHTS[j]*count[(f,4,16,arm)]/N
                       for j,f in enumerate(KINDS)) for arm in ARMS}
        primary.append({
            "public_sha256":key,"degree":row["observation_degree"],
            "family":row["family"],"rates":rates,
            "posterior_minus_uniform":rates["posterior"]-rates["uniform"],
            "posterior_minus_shuffled":rates["posterior"]-rates["shuffled_prior"],
            "posterior_minus_static":rates["posterior"]-rates["static_prior"],
        })
    assert result["primary_by_source"]==primary
    wins_u=sum(x["posterior_minus_uniform"]>0 for x in primary)
    wins_s=sum(x["posterior_minus_shuffled"]>0 for x in primary)
    mean=statistics.mean(x["posterior_minus_uniform"] for x in primary)
    assert result["primary_wins_vs_uniform"]==wins_u
    assert result["primary_wins_vs_shuffle"]==wins_s
    assert abs(result["primary_weighted_mean_gain_vs_uniform"]-mean)<1e-15
    assert result["registered_primary_pass"]==(
        wins_u>=9 and wins_s>=9 and mean>=.10)
    return {"original_sources_replayed":24,"train_episodes":768,
            "test_clean_schedules_replayed":number,
            "independent_policy_world_budget_checks":number*5*5*3*3,
            "registered_primary_pass":result["registered_primary_pass"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path)
    p.add_argument("--full",type=Path)
    p.add_argument("--check",action="store_true")
    a=p.parse_args()
    if not a.fixture or not a.full:p.error("both source and grade paths required")
    receipt=audit(a.fixture.read_bytes(),a.full.read_bytes())
    print("NOR24 INDEPENDENT ORIGINAL INTERPRETER REPLAY",receipt)
    if a.check:assert receipt["test_clean_schedules_replayed"]==12*1024

if __name__=="__main__":main()
