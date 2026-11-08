"""QO-18: two-round DAgger + validation-calibrated selective gate.

Strict sequence: read frozen QO-17 weights; aggregate student-visited
TRAIN and VALIDATION histories, with QO-12 labels; fit two new recurrent
models, selecting epochs on validation only; choose gate using fresh
VALIDATION-only source trajectories; ONLY THEN import the independent
QO-18 frozen test distribution for supervised and equal-budget policy
evaluation. QO-07 remains the exclusive certificate authority.
"""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
from random import Random
import json

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate, choose_probe, frozen_error,
)
from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TOLERANCE, evidence_features,
)
from research.ingression_dynamics_v1.gap_progress_teacher import (
    greedy_gain_cell,
)
from research.ingression_dynamics_v1.gap_student_data import (
    example_from_ledger,source_stats,
)
from research.ingression_dynamics_v1.gap_student_pilot import (
    metric_for_student, run_gap_policy, _cost, _mean,
    _paired_comparisons,
)
from research.ingression_dynamics_v1.qo18_dagger_data import (
    train_dagger_rounds,DAGGER_SEEDS,CHECKPOINTS,
)
from research.ingression_dynamics_v1.qo18_gate_calibration import (
    validation_calibration_states,select_gate,confusion,
    CAL_SEEDS,THRESHOLDS,POLICIES as CAL_POLICIES,
    CHECKPOINTS as CAL_CHECKPOINTS,
)
from research.ingression_dynamics_v1.qo17_occupancy_data import is_eligible
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    from_receipt, forward, trainable_count, inference_multiplies,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as run_bottleneck,stream_seed,verify_common_stream_prefix,
)

ROOT=Path(__file__).resolve().parents[2]
QO17_RECEIPT=ROOT/"results/qo17_occupancy/receipt.json"
STUDENT_ARMS=("qo17_frozen_rnn","dagger_round1_fixed_005",
              "dagger_round2_fixed_005","dagger_round2_calibrated")
POLICIES=("bottleneck","exact_gap_h2")+STUDENT_ARMS


def read_qo17_reference():
    raw=QO17_RECEIPT.read_bytes()
    original=json.loads(raw.decode("ascii"))
    model=from_receipt(
        original["new_model_weights"]["recurrent_occupancy_margin_gate"])
    if trainable_count(model)!=138:
        raise AssertionError("historical QO-17 recurrent architecture changed")
    return model,sha256(raw).hexdigest()


def fit_and_calibrate():
    """Complete all training and threshold choice before frozen import."""
    old,old_hash=read_qo17_reference()
    trained=train_dagger_rounds(old)
    first=trained["rounds"][0]["model"]
    final=trained["final_model"]
    calibration_rows,calibration_meta=validation_calibration_states(final)
    selection=select_gate(calibration_rows)
    return {
        "models":{
            "qo17_frozen_rnn":old,
            "dagger_round1_fixed_005":first,
            "dagger_round2_fixed_005":final,
            "dagger_round2_calibrated":final,
        },
        "reference_qo17_sha256":old_hash,
        "training":trained,
        "calibration_samples":calibration_rows,
        "calibration_provenance":calibration_meta,
        "calibration_choice":selection,
    }


def run_student_policy(case_index,case,seed,arm,model,threshold,checkpoints):
    """Predictable pre-draw selection with QO-07's unchanged certificate."""
    if arm not in STUDENT_ARMS:
        raise ValueError("unsupported QO-18 student policy")
    if len(case["rates"])!=2 or any(len(r)!=1 for r in case["rates"]):
        raise ValueError("two addressable Bernoulli cells required")
    ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
    streams=[Random(stream_seed(case_index,seed,i,0)) for i in (0,1)]
    rates=case["rates"]
    truth=abs(rates[0][0]-rates[1][0])/2<=case["tolerance"]
    cost=_cost()
    cost["disabled_gate_fallbacks"]=0
    cost["engagement_threshold"]=None if threshold is None else str(threshold)
    rows=[]
    for t in range(checkpoints[-1]+1):
        if t in checkpoints:
            cert=certificate(ledger.snapshot(),1,case["tolerance"])
            covered=all(lo<=rates[i][0]<=hi for i,row in
                        enumerate(ledger.bounds) for lo,hi in row)
            decision=cert["decision"]
            wrong=((decision=="certified-feasible" and not truth)
                   or (decision=="certified-infeasible" and truth))
            if wrong and covered:
                raise AssertionError("wrong certificate on covered truth")
            error=frozen_error(rates,cert["transient_labels"],
                               cert["frozen_midpoints"])
            if covered and error>cert["robust_macro_error_upper"]:
                raise AssertionError("QO-07 robust certificate violated")
            rows.append({
                "scenario":case["id"],"seed":seed,"policy":arm,
                "queries":t,"decision":decision,
                "false_cert":wrong,"true_rates_covered":covered,
                "true_feasible":truth,
                "certification_gap":str(
                    cert["robust_macro_error_upper"]-
                    cert["true_optimum_lower"]),
                "robust_upper":str(cert["robust_macro_error_upper"]),
                "true_optimum_lower":str(cert["true_optimum_lower"]),
                "true_frozen_error":str(error),
                "cell_query_counts":[ledger.counts[i][0] for i in (0,1)],
                "policy_cost":dict(cost),
            })
        if t==checkpoints[-1]:
            break
        fallback=choose_probe(ledger,1,"bottleneck")
        if min(ledger.counts[i][0] for i in (0,1))<8:
            cost["readiness_fallbacks"]+=1
            chosen=fallback
        elif cert_decision(ledger)!="undetermined":
            cost["already_certified_fallbacks"]+=1
            chosen=fallback
        else:
            cost["eligible_states"]+=1
            if threshold is None:
                cost["low_gain_fallbacks"]+=1
                cost["disabled_gate_fallbacks"]+=1
                chosen=fallback
            else:
                prediction=forward(model,evidence_features(ledger))
                cost["student_inferences"]+=1
                cost["student_multiply_accumulates"]+=inference_multiplies(model)
                if max(prediction)<float(threshold):
                    cost["low_gain_fallbacks"]+=1
                    chosen=fallback
                else:
                    chosen=(greedy_gain_cell(ledger,prediction),0)
                    cost["model_chosen_actions"]+=1
                    cost["selected_cell_"+str(chosen[0])]+=1
        i,a=chosen
        ledger.observe(i,a,int(streams[i].random()<float(rates[i][0])))
    if cost["eligible_states"]!=cost["low_gain_fallbacks"]+cost["model_chosen_actions"]:
        raise AssertionError("incomplete model/abstention accounting")
    if cost["readiness_fallbacks"]+cost["already_certified_fallbacks"]+(
            cost["eligible_states"])!=checkpoints[-1]:
        raise AssertionError("one or more sampling actions unaccounted")
    return tuple(rows),tuple(ledger.receipts),cost


def cert_decision(ledger):
    return certificate(ledger.snapshot(),1,TOLERANCE)["decision"]


def visited_audit(receipts,model,arm,case,seed,threshold,checkpoints):
    """Post-hoc teacher labels, never accessed by online learned actions."""
    ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
    rows=[]
    for t in range(checkpoints[-1]+1):
        if t in checkpoints and is_eligible(ledger):
            ex=example_from_ledger(
                ledger,"qo18_frozen_visited",int(case["id"].split("_")[-1]),
                seed,arm,t)
            pred=forward(model,ex["tokens"])
            chosen=0 if pred[0]>=pred[1] else 1
            rows.append({
                "scenario":case["id"],"seed":seed,
                "arm":arm,"query":t,
                "cluster":case["id"]+":"+str(seed),
                "counts":[ledger.counts[i][0] for i in (0,1)],
                "true_q":[str(x) for x in ex["raw_q"]],
                "teacher_positive":max(ex["q"])>=F(1,20),
                "model_score":max(pred),
                "model_q":list(pred),
                "predicted_engages":(
                    threshold is not None and max(pred)>=float(threshold)),
                "teacher_action_informative":ex["raw_q"][0]!=ex["raw_q"][1],
                "true_action_regret":str(max(ex["raw_q"])-ex["raw_q"][chosen]),
                "posthoc_teacher_gap_evaluations":ex["teacher_gap_evaluations"],
            })
        if t==checkpoints[-1]:
            break
        i,a,y=receipts[t]
        ledger.observe(i,a,y)
    if tuple(ledger.receipts)!=tuple(receipts):
        raise AssertionError("visited posthoc audit replay changed observations")
    return tuple(rows)


def full_report():
    fitted=fit_and_calibrate()
    models=fitted["models"]
    choice=fitted["calibration_choice"]
    # Test module imported only AFTER all fitting and validation gate selection.
    from research.ingression_dynamics_v1.qo18_frozen_contract import (
        FROZEN_SCENARIOS,FROZEN_SUPERVISED_SEEDS,FROZEN_POLICY_SEEDS,
        SUPERVISED_CHECKPOINTS,POLICY_CHECKPOINTS,VISITED_CHECKPOINTS,
        TRAIN_DAGGER_SEEDS,VALIDATION_DAGGER_SEEDS,
        DAGGER_SOURCE_CHECKPOINTS,CALIBRATION_SEEDS,
        CALIBRATION_SOURCE_POLICIES,CALIBRATION_CHECKPOINTS,
        GATE_CANDIDATES,ARMS,PRIMARY_ARM,frozen_supervised_ledgers,
        contract,contract_sha256,
    )
    if POLICIES!=ARMS or DAGGER_SEEDS!={
        "train":TRAIN_DAGGER_SEEDS,"validation":VALIDATION_DAGGER_SEEDS}:
        raise AssertionError("registered arm/data split drift")
    if (CHECKPOINTS!=DAGGER_SOURCE_CHECKPOINTS or CAL_SEEDS!=CALIBRATION_SEEDS
            or CAL_POLICIES!=CALIBRATION_SOURCE_POLICIES or
            CAL_CHECKPOINTS!=CALIBRATION_CHECKPOINTS or
            THRESHOLDS!=GATE_CANDIDATES):
        raise AssertionError("validation threshold or sampling protocol drift")
    if PRIMARY_ARM!="dagger_round2_calibrated":
        raise AssertionError("primary arm changed")
    labeled,failures=frozen_supervised_ledgers()
    supervised=[]
    for idx,seed,sched,t,ledger in labeled:
        if cert_decision(ledger)=="undetermined":
            supervised.append(example_from_ledger(
                ledger,"qo18_frozen_test",idx,seed,sched,t))
    if not supervised:
        raise AssertionError("no independent undecided evidence states")
    test_source_stats=source_stats(supervised,failures)
    supervised_metrics={}
    fixed_schedule_calibration={}
    for arm,model in models.items():
        supervised_metrics[arm]=metric_for_student(model,supervised)
        pred_labels=[]
        for ex in supervised:
            pred=forward(model,ex["tokens"])
            pred_labels.append({
                "cluster":str(ex["rate_index"])+":"+str(ex["seed"])+":"+
                    str(ex["schedule"]),
                "teacher_positive":max(ex["q"])>=F(1,20),
                "model_score":max(pred),
            })
        gate=(F(choice["threshold"]) if arm==PRIMARY_ARM and
              choice["threshold"] is not None else
              None if arm==PRIMARY_ARM else F(1,20))
        fixed_schedule_calibration[arm]=confusion(pred_labels,gate)
    rows,traces=[],[]
    all_costs={arm:_cost() for arm in ARMS}
    visited={arm:[] for arm in STUDENT_ARMS}
    source_hashes={}
    for ci,case in enumerate(FROZEN_SCENARIOS):
        for seed in FROZEN_POLICY_SEEDS:
            source_hashes[case["id"]+":"+str(seed)]={}
            for arm in ARMS:
                if arm=="bottleneck":
                    observations,history=run_bottleneck(
                        ci,case,seed,"bottleneck",POLICY_CHECKPOINTS)
                    cost=_cost()
                    observations=[dict(
                        r,policy=arm,certification_gap=str(
                            F(r["robust_upper"])-F(r["true_optimum_lower"])),
                        policy_cost=dict(cost)) for r in observations]
                elif arm=="exact_gap_h2":
                    observations,history,cost=run_gap_policy(
                        ci,case,seed,"exact_qo12_gap_h2",{},
                        POLICY_CHECKPOINTS)
                    observations=[dict(x,policy=arm) for x in observations]
                else:
                    gate=(F(choice["threshold"]) if arm==PRIMARY_ARM
                          and choice["threshold"] is not None else
                          None if arm==PRIMARY_ARM else F(1,20))
                    observations,history,cost=run_student_policy(
                        ci,case,seed,arm,models[arm],gate,POLICY_CHECKPOINTS)
                    visited[arm].extend(visited_audit(
                        history,models[arm],arm,case,seed,gate,VISITED_CHECKPOINTS))
                if len(history)!=POLICY_CHECKPOINTS[-1] or not (
                        verify_common_stream_prefix(ci,case,seed,history)):
                    raise AssertionError("nonreplayable held-out paired stream")
                rows.extend(observations)
                h=sha256(json.dumps(history,separators=(",",":")).encode("ascii")).hexdigest()
                source_hashes[case["id"]+":"+str(seed)][arm]=h
                traces.append({
                    "scenario":case["id"],"seed":seed,"policy":arm,
                    "queries":len(history),"sha256":h,
                })
                for k in _cost():
                    all_costs[arm][k]+=cost[k]
    grouped=[]
    for at in POLICY_CHECKPOINTS:
        for arm in ARMS:
            batch=[r for r in rows if r["queries"]==at and r["policy"]==arm]
            if len(batch)!=len(FROZEN_SCENARIOS)*len(FROZEN_POLICY_SEEDS):
                raise AssertionError("missing frozen policy strata")
            grouped.append({
                "queries":at,"policy":arm,"n":len(batch),
                "correct_certificates":sum(r["decision"]!="undetermined"
                    and not r["false_cert"] for r in batch),
                "false_certificates":sum(r["false_cert"] for r in batch),
                "coverage_misses":sum(not r["true_rates_covered"] for r in batch),
                "mean_gap":str(_mean(r["certification_gap"] for r in batch)),
                "mean_true_tv_error":str(_mean(r["true_frozen_error"] for r in batch)),
            })
    paired=_paired_comparisons(
        rows,ARMS,POLICY_CHECKPOINTS,FROZEN_SCENARIOS,FROZEN_POLICY_SEEDS)
    visit_meta={}
    for arm,samples in visited.items():
        metrics=confusion(samples,F(choice["threshold"]) if arm==PRIMARY_ARM
            and choice["threshold"] is not None else
            None if arm==PRIMARY_ARM else F(1,20))
        metrics["posthoc_teacher_evaluations"]=sum(
            row["posthoc_teacher_gap_evaluations"] for row in samples)
        metrics["informative_action_decisions"]=sum(
            row["teacher_action_informative"] for row in samples)
        metrics["informative_action_mistakes"]=sum(
            row["teacher_action_informative"]
            and F(row["true_action_regret"])>0 for row in samples)
        metrics["independent_case_seed_clusters"]=len(set(
            row["cluster"] for row in samples))
        visit_meta[arm]={"metrics":metrics,"samples":samples}
    primary=next(x for x in paired if x["budget"]==POLICY_CHECKPOINTS[-1]
                 and x["policy"]==PRIMARY_ARM)
    primary_vis=visit_meta[PRIMARY_ARM]["metrics"]
    gate_pass=(primary_vis["negatives"]>0 and primary_vis["positives"]>0
        and primary_vis["false_engagement_rate"]<=0.10
        and primary_vis["recall"]>=0.80)
    return {
        "contract":"QO-18 source-frozen two-round DAgger selective fallback experiment",
        "test_contract":contract(),"test_contract_sha256":contract_sha256(),
        "historical_qo17_receipt_sha256":fitted["reference_qo17_sha256"],
        "training":fitted["training"],
        "calibration_source":fitted["calibration_provenance"],
        "calibration_samples":fitted["calibration_samples"],
        "calibration_selection":choice,
        "student_models":models,
        "parameter_counts":{arm:trainable_count(model)
                             for arm,model in models.items()},
        "inference_macs_per_call":{arm:inference_multiplies(model)
                                    for arm,model in models.items()},
        "frozen_supervised_stats":test_source_stats,
        "frozen_supervised_examples":len(supervised),
        "frozen_supervised_failures":list(failures),
        "frozen_supervised_fingerprint":sha256(json.dumps(
            [(ex["rate_index"],ex["seed"],ex["schedule"],ex["time"],
              [str(x) for x in ex["raw_q"]]) for ex in supervised],
              separators=(",",":")).encode("ascii")).hexdigest(),
        "supervised_metrics":supervised_metrics,
        "fixed_schedule_engagement":fixed_schedule_calibration,
        "visited_engagement":visit_meta,
        "policy_arms":list(ARMS),"policy_groups":grouped,
        "policy_costs":all_costs,
        "paired":paired,
        "primary_policy":{
            "arm":PRIMARY_ARM,
            "budget":POLICY_CHECKPOINTS[-1],
            "paired_gap":primary,
            "test_gate_constraints_pass":gate_pass,
            "test_gate_confusion":{
                k:primary_vis[k] for k in ("tp","fp","tn","fn",
                "positives","negatives","precision","recall",
                "false_engagement_rate")
            },
            "validation_gate_feasible":choice["feasible"],
        },
        "policy_rows":rows,"trajectory_hashes":traces,
        "matched_source_hashes":source_hashes,
        "counts":{
            "trajectories":len(traces),
            "snapshots":len(rows),
            "queries":sum(x["queries"] for x in traces),
        },
        "limitations":"Tiny two-cell known-addressable Bernoulli dynamics. Posthoc gate labels do not alter observed policy. No Qwen, BitAgent, biological or hidden-state transfer.",
    }
