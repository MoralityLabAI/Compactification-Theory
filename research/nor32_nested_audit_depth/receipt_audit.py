"""NOR-32 separate-process independent 23,592,960 decision replay.

Does not import NOR32 implementation. Reconstructs all source-native targets,
both 2048-schedule HMAC batches using a sparse Fisher-Yates variant, all
18,432 depth-specific greedy attacks using trial-ID SETS, source SHA binding,
every 512..2048-bit detection mask, and preregistered source-level gates.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from fractions import Fraction
from math import comb,isclose
from pathlib import Path
import hashlib,hmac,json,random,statistics,sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor24_adaptive_audits import experiment as old
from research.nor25_robust_audits import experiment as prior
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

REPS=(26,27,28,29)
DEGREES=(512,1024)
DEPTHS=(512,1024,1536,2048)
ARMS=("hybrid_50","uniform16","static_prior")
PHASES=("calibration","evaluation")
REGIMES=("independent_calibration","leaked_evaluation_root")
N=24
R=4
RESTARTS=32
TRIALS=2048
TOTAL=N*len(ARMS)*len(REGIMES)*RESTARTS*sum(DEPTHS)
assert TOTAL==23592960
MAX=1<<256


def source_seed(rep,d,f):
    return (corpus.SEED+160032000+1000003*rep+
            100009*DEGREES.index(d)+10007*nor21.FAMILIES.index(f))


def true_future(row,pred):
    rep,d,f=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(source_seed(rep,d,f))
    for _ in range(row["selected_attempt"]+1):
        prog=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,f,3*d//2),
                     corpus.execute(x,prog)) for _ in range(2))
        query=nor21.public_source(rng,f,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,f)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    actual=corpus.execute(query,prog)
    assert actual==pred
    return actual


def commit(pre):
    return sha256(b"NOR32-attack-commit-v1\x00"+old.canonical(pre)).hexdigest()


def root_commit(root):
    return sha256(b"NOR32-root-v1\x00"+root).hexdigest()


def schedule(root,phase,digest,arm,trial,m,order):
    if arm=="static_prior":
        return tuple(order[:16])
    key=hmac.new(root,("NOR32-v1-"+phase+"-subkey").encode("ascii"),sha256).digest()
    if arm=="hybrid_50":
        top=tuple(order[:8])
        choices=[i for i in range(m) if i not in set(top)]
        draws=8
    else:
        top=()
        choices=list(range(m))
        draws=16
    swap={}
    out=[]
    for draw in range(draws):
        n=len(choices)-draw
        limit=MAX-(MAX%n)
        retry=0
        while True:
            msg=(f"NOR28-sampling-v1|{digest}|{arm}|{trial}|"
                 f"{draw}|{retry}").encode("ascii")
            val=int.from_bytes(hmac.new(key,msg,hashlib.sha256).digest(),"big")
            if val<limit:
                index=val%n
                break
            retry+=1
            assert retry<100
        out.append(swap.get(index,choices[index]))
        swap[index]=swap.get(n-1,choices[n-1])
    return top+tuple(out)


def trial_id_coverage(plans,m):
    sets=[set() for _ in range(m)]
    for trial,plan in enumerate(plans):
        for i in plan:
            sets[i].add(trial)
    return sets


def greedy(coverage,digest,arm,rep):
    tie=list(range(len(coverage)))
    random.Random(old.stream_seed(202611310111,digest,arm,rep)).shuffle(tie)
    used=set()
    seen=set()
    for _ in range(R):
        j=min((i for i in tie if i not in used),key=lambda i:len(seen|coverage[i]))
        used.add(j)
        seen|=coverage[j]
    return tuple(sorted(used)),len(seen)


def read(paths):
    names=("fixture","calibration","pre_sealed","pre_manifest",
           "evaluation","late_sealed","late_manifest","outcome")
    raw={k:Path(paths[k]).read_bytes() for k in names}
    hashes={k:sha256(raw[k]).hexdigest() for k in names}
    data={k:json.loads(raw[k]) for k in names}
    f=data["fixture"]
    assert f["schema"]=="nor32.public_only_24_fresh_sources.v1"
    assert f["source_offset"]==160032000 and f["replicates"]==list(REPS)
    assert f["source_only"] and f["hidden_truth_absent"]
    assert f["attack_sets_absent"] and f["challenge_entropy_absent"]
    assert len(f["cases"])==N
    assert len({x["full_public_input_sha256"] for x in f["cases"]})==N
    assert {(x["replicate"],x["observation_degree"],x["family"]) for x in f["cases"]}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in nor21.FAMILIES}
    roots={}
    for phase in PHASES:
        obj=data[phase]
        assert obj["schema"]=="nor32.nested_independent_os_roots_24x3x2048.v1"
        assert obj["phase"]==phase
        assert obj["fixture_sha256"]==hashes["fixture"]
        assert obj["source_count"]==N and obj["trial_count"]==TRIALS
        assert obj["root_origin"]=="os_secrets_token_bytes_32"
        assert obj["sampler"]=="HMAC-SHA256(NOR32 phase subkey) then NOR28 unbiased Fisher-Yates"
        assert len(obj["records"])==N
        root=bytes.fromhex(obj["root_hex_disclosed_after_freeze"])
        assert len(root)==32
        assert root_commit(root)==obj["root_commitment_sha256"]
        roots[phase]=root
    assert roots["calibration"]!=roots["evaluation"]
    assert data["calibration"]["precommit_manifest_sha256"] is None
    assert data["evaluation"]["precommit_manifest_sha256"]==hashes["pre_manifest"]
    assert data["evaluation"]["calibration_sha256"]==hashes["calibration"]
    pre={}
    late={}
    for prefix,regime in (("pre",REGIMES[0]),("late",REGIMES[1])):
        phase="calibration" if prefix=="pre" else "evaluation"
        manifest=data[prefix+"_manifest"]
        sealed=data[prefix+"_sealed"]
        tag="pre" if prefix=="pre" else "leaked"
        assert manifest["schema"]==f"nor32.{tag}_root_9216_commitments.v1"
        assert sealed["schema"]==f"nor32.{tag}_root_9216_salted_openings.v1"
        assert manifest["fixture_sha256"]==sealed["fixture_sha256"]==hashes["fixture"]
        assert manifest["phase_sha256"]==sealed["phase_sha256"]==hashes[phase]
        assert manifest["sealed_sha256"]==hashes[prefix+"_sealed"]
        assert manifest["calibration_sha256"]==sealed["calibration_sha256"]==hashes["calibration"]
        assert manifest["precommit_manifest_sha256"]==sealed["precommit_manifest_sha256"]==(
            None if prefix=="pre" else hashes["pre_manifest"])
        assert manifest["attacker_class"]==regime
        assert manifest["root_of_phase_known_before_committing"]==(prefix=="late")
        assert manifest["depths"]==list(DEPTHS)
        assert manifest["attack_count"]==len(manifest["commitments"])==len(sealed["openings"])==9216
        public={(x["source_sha256"],x["auditor"],x["depth"],x["replica"]):
                x["commitment_sha256"] for x in manifest["commitments"]}
        indexed={(x["source_sha256"],x["auditor"],x["depth"],x["replica"]):x
                 for x in sealed["openings"]}
        assert len(public)==len(indexed)==9216
        for key,op in indexed.items():
            assert op["attack_class"]==regime
            assert op["changed_indices"]==sorted(op["changed_indices"])
            assert len(set(op["changed_indices"]))==len(op["changed_indices"])==R
            assert len(op["nonce_hex"])==64
            assert commit(op)==public[key]
            tampered=dict(op,nonce_hex=("0" if op["nonce_hex"][0]!="0" else "1")
                          +op["nonce_hex"][1:])
            assert commit(tampered)!=public[key]
        if prefix=="pre":
            pre=indexed
            pre_commit=public
        else:
            late=indexed
            for key,op in late.items():
                prev=pre[key]
                changed=dict(prev,changed_indices=op["changed_indices"],nonce_hex=op["nonce_hex"])
                assert commit(changed)!=pre_commit[key]
    out=data["outcome"]
    assert out["schema"]=="nor32.nested_depth_attack_coverage_and_detection.v1"
    for name,field in (("fixture","fixture_sha256"),("calibration","calibration_sha256"),
                       ("pre_sealed","pre_sealed_sha256"),("pre_manifest","pre_manifest_sha256"),
                       ("evaluation","evaluation_sha256"),("late_sealed","late_sealed_sha256"),
                       ("late_manifest","late_manifest_sha256")):
        assert hashes[name]==out[field],name
    assert out["source_count"]==N
    assert out["calibration_root_commitment"]==data["calibration"]["root_commitment_sha256"]
    assert out["evaluation_root_commitment"]==data["evaluation"]["root_commitment_sha256"]
    assert out["calibration_root_os_origin"]==out["evaluation_root_os_origin"]=="os_secrets_token_bytes_32"
    assert out["depths"]==list(DEPTHS)
    assert out["restarts"]==RESTARTS
    assert out["early_commitment_count"]==out["late_commitment_count"]==9216
    assert out["fixed_batch_decisions"]==TOTAL
    assert out["informed_per_trial_controls"]==N*len(ARMS)*TRIALS
    assert out["informed_per_trial_detected"]==0
    assert out["source_native_mask_mismatches"]==0
    assert out["honest_target_rejections"]==0
    assert out["delta_disagreements"]==0
    assert out["forged_prior_commitments_accepted"]==0
    assert out["postcommit_root_disclosure_changes"]==0
    assert out["monotone_nested_coverage"]
    assert len(out["source_records"])==N
    paired=[]
    coverage_count={arm:{str(depth):{str(m):0 for m in (769,1537)}
                         for depth in DEPTHS} for arm in ARMS}
    policy_means={arm:{str(depth):{reg:[] for reg in REGIMES}
                       for depth in DEPTHS} for arm in ARMS}
    counted=0
    support_changed=0
    for src,cal,e,rec in zip(f["cases"],data["calibration"]["records"],
                             data["evaluation"]["records"],out["source_records"]):
        digest=src["full_public_input_sha256"]
        assert digest==cal["source_sha256"]==e["source_sha256"]==rec["source_sha256"]
        native,h=old.public_only_case(src)
        assert native.program is None and native.future==()
        mask=nor21.native_mask(native,h,src["observation_degree"])
        proposed=old.public_proposal(src)
        assert mask==src["original_public_accept_mask"]==proposed["mask"]
        assert rec["public_mask"]==cal["public_mask"]==e["public_mask"]==mask
        assert rec["prediction_sha256"]==cal["public_prediction_sha256"]==e["public_prediction_sha256"]==proposed["sha256"]
        target=true_future(src,proposed["predicted"])
        assert rec["true_target_sha256_after_freeze"]==old.sha_target(target)
        m=len(target)
        assert m==rec["m"]==cal["m"]==e["m"]
        order=old.static_schedule(old.supports(proposed["predicted"],h),
                                  prior.PRIOR,digest,0)
        assert rec["risk_prefix"]==cal["risk_prefix"]==e["risk_prefix"]==list(order[:8])
        for phase,phase_row in (("calibration",cal),("evaluation",e)):
            for arm in ARMS:
                plans=phase_row["schedules"][arm]
                assert len(plans)==TRIALS
                for trial,plan in enumerate(plans):
                    assert tuple(plan)==schedule(roots[phase],phase,digest,arm,trial,m,order)
        obs={(x["auditor"],x["depth"],x["attack_class"],x["replica"]):x
             for x in rec["observations"]}
        assert len(obs)==len(ARMS)*len(DEPTHS)*len(REGIMES)*RESTARTS
        for arm in ARMS:
            cal_plans=cal["schedules"][arm]
            eval_plans=e["schedules"][arm]
            for depth in DEPTHS:
                dstr=str(depth)
                never=m-len({i for plan in eval_plans[:depth] for i in plan})
                assert never==rec["never_inspected"][arm][dstr]
                if never>=R:
                    coverage_count[arm][dstr][str(m)]+=1
                for regime,train,opened in (
                    (REGIMES[0],cal_plans,pre),(REGIMES[1],eval_plans,late)):
                    exposures=trial_id_coverage(train[:depth],m)
                    hit_count=0
                    uniques=set()
                    for rep in range(RESTARTS):
                        key=(digest,arm,depth,rep)
                        A,training_hits=greedy(exposures,digest,arm,rep)
                        op=opened[key]
                        assert tuple(op["changed_indices"])==A
                        assert op["training_covered_schedules"]==training_hits
                        uniques.add(A)
                        if regime==REGIMES[1] and A!=tuple(pre[key]["changed_indices"]):
                            support_changed+=1
                        hits=0
                        bitmask=0
                        attacked=set(A)
                        for trial,plan in enumerate(eval_plans[:depth]):
                            if attacked.intersection(plan):
                                hits+=1
                                bitmask|=1<<trial
                        o=obs[(arm,depth,regime,rep)]
                        assert o["total"]==depth and o["hits"]==hits
                        assert o["detected_bits_hex"]==format(bitmask,f"0{depth//4}x")
                        hit_count+=hits
                        counted+=depth
                    assert rec["unique_attacks"][arm][dstr][regime]==len(uniques)
                    rate=hit_count/(RESTARTS*depth)
                    assert isclose(rec["detection_rates"][arm][dstr][regime],
                                   rate,abs_tol=1e-15)
                    policy_means[arm][dstr][regime].append(rate)
            for a,b in zip(DEPTHS,DEPTHS[1:]):
                assert rec["never_inspected"][arm][str(a)]>=rec["never_inspected"][arm][str(b)]
        floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        assert rec["fixed_blind_one_trial_floor"]==[floor.numerator,floor.denominator]
        delta=(rec["detection_rates"]["hybrid_50"]["2048"][REGIMES[0]]-
               rec["detection_rates"]["hybrid_50"]["2048"][REGIMES[1]])
        assert isclose(delta,rec["hybrid2048_independent_minus_leaked"],abs_tol=1e-15)
        paired.append(delta)
    assert counted==TOTAL
    assert out["late_changed_attack_support_count"]==support_changed
    assert out["batch_zero_hit_feasibility"]==coverage_count
    for arm in ARMS:
        for d in DEPTHS:
            for reg in REGIMES:
                mean=statistics.mean(policy_means[arm][str(d)][reg])
                assert isclose(out["equal_source_detection"][arm][str(d)][reg],
                               mean,abs_tol=1e-15)
    long_512=coverage_count["hybrid_50"]["512"]["1537"]
    long_2048=coverage_count["hybrid_50"]["2048"]["1537"]
    assert out["long_hybrid_sources_with_four_unseen_at_512"]==long_512
    assert out["long_hybrid_sources_with_four_unseen_at_2048"]==long_2048
    wins=sum(x>0 for x in paired)
    gain=statistics.mean(paired)
    assert out["hybrid2048_positive_source_pairs"]==wins
    assert isclose(out["hybrid2048_source_mean_detection_gain"],gain,abs_tol=1e-15)
    gates={
        "long_output_hybrid_four_unseen_512_vs_2048":
            long_512>=10 and long_2048<=2 and long_512-long_2048>=8,
        "2048_hybrid_calibrated_detection_beats_known_root":
            wins>=20 and gain>=.005,
        "coverage_monotone_and_commitments_precede_eval_root":True,
        "exact_two_root_replay_23592960_witness_decisions":True,
        "fully_informed_evasion_and_commitment_binding":True,
    }
    assert out["registered_gates"]==gates
    assert out["registered_primary_pass"]==all(gates.values())
    print("NOR32 ORIGINAL INDEPENDENT REPLAY PASS",N,"SOURCES",counted,
          "AUDIT DECISIONS","LONG-HYBRID 512/2048",long_512,long_2048,
          "2048 HYBRID PAIRED",wins,gain,
          "SCIENTIFIC PASS",out["registered_primary_pass"])
    for k in names:
        print("NOR32 ORIGINAL",k,len(raw[k]),hashes[k])


def main():
    p=argparse.ArgumentParser()
    for key in ("fixture","calibration","pre-sealed","pre-manifest",
                "evaluation","late-sealed","late-manifest","outcome"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    read({"fixture":a.fixture,"calibration":a.calibration,
          "pre_sealed":a.pre_sealed,"pre_manifest":a.pre_manifest,
          "evaluation":a.evaluation,"late_sealed":a.late_sealed,
          "late_manifest":a.late_manifest,"outcome":a.outcome})


if __name__=="__main__":main()
