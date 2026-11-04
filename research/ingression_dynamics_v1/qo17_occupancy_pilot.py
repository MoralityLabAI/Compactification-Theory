"""QO-17: registered 2x2 architecture/loss occupancy experiment.

Train/validation source families only are accessed until every model
has fitted. The QO-17 frozen test generator is imported only inside
full_report after the four fit() calls. QO-16 fixed-schedule models
are read as fixed historical comparators, never reselected.
QO-07 confidence sequences alone authorize certificates.
"""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
from random import Random
import json

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate,
)
from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TOLERANCE, evidence_features,
)
from research.ingression_dynamics_v1.gap_progress_teacher import (
    NORMALIZED_GAIN_GATE, gain_to_target,
)
from research.ingression_dynamics_v1.gap_student_data import (
    example_from_ledger, source_stats,
)
from research.ingression_dynamics_v1.gap_student_pilot import (
    metric_for_student, run_gap_policy, _cost, _mean,
    _paired_comparisons,
)
from research.ingression_dynamics_v1.qo17_occupancy_data import (
    training_mixture, is_eligible, OCC_SEEDS, OCC_CHECKPOINTS,
)
from research.ingression_dynamics_v1.qo17_margin_gate_loss import (
    fit_augmented,
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    fit, create_linear, create_rnn, forward, from_receipt, trainable_count,
    inference_multiplies,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as run_bottleneck, verify_common_stream_prefix,
)

ROOT=Path(__file__).resolve().parents[2]
QO16_RECEIPT=ROOT/"results/gap_student/receipt.json"
LEARNED_ARMS=(
    "qo16_linear_fixed_mse","qo16_recurrent_fixed_mse",
    "linear_occupancy_mse","recurrent_occupancy_mse",
    "linear_occupancy_margin_gate","recurrent_occupancy_margin_gate",
)
TRAINED_ARMS=LEARNED_ARMS[2:]


def load_qo16_baselines():
    raw=QO16_RECEIPT.read_bytes()
    receipt=json.loads(raw.decode("ascii"))
    weights=receipt["trained_models"]
    return {
        "qo16_linear_fixed_mse":from_receipt(weights["linear_gap_student"]),
        "qo16_recurrent_fixed_mse":from_receipt(weights["recurrent_gap_student"]),
    },sha256(raw).hexdigest()


def train_models():
    """Train four ablations before touching any QO-17 frozen test module."""
    train,train_meta=training_mixture("train")
    validation,val_meta=training_mixture("validation")
    trained={}
    fits={}
    configurations=(
        ("linear_occupancy_mse",create_linear,fit),
        ("recurrent_occupancy_mse",create_rnn,fit),
        ("linear_occupancy_margin_gate",create_linear,fit_augmented),
        ("recurrent_occupancy_margin_gate",create_rnn,fit_augmented),
    )
    for name,constructor,trainer in configurations:
        result=trainer(constructor(),train,validation)
        trained[name]=result["weights"]
        fits[name]={k:v for k,v in result.items() if k!="weights"}
    return {
        "new_models":trained,
        "fits":fits,
        "train_metadata":train_meta,
        "validation_metadata":val_meta,
        "train_examples":len(train),
        "validation_examples":len(validation),
    }


def _mapped_policy(arm,model):
    if arm=="exact_gap_h2":
        return "exact_qo12_gap_h2",{}
    if arm.endswith("linear_fixed_mse") or arm.startswith("linear_"):
        return "linear_gap_student",{"linear_gap_student":model}
    return "recurrent_gap_student",{"recurrent_gap_student":model}


def visited_calibration(ledger_receipts,model,case,seed,arm,checkpoints):
    """Post-hoc exact reference, never consulted by the online student."""
    ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
    rows=[]
    for t in range(checkpoints[-1]+1):
        if t in checkpoints and is_eligible(ledger):
            ex=example_from_ledger(
                ledger,"heldout_visited",int(case["id"].split("_")[-1]),
                seed,arm,t)
            predicted=forward(model,ex["tokens"])
            truth=max(ex["q"])>=NORMALIZED_GAIN_GATE
            guessed=max(predicted)>=float(NORMALIZED_GAIN_GATE)
            chosen=0 if predicted[0]>=predicted[1] else 1
            q=ex["raw_q"]
            rows.append({
                "scenario":case["id"],"seed":seed,
                "arm":arm,"query":t,
                "counts":[ledger.counts[i][0] for i in (0,1)],
                "exact_q":[str(v) for v in q],
                "normalized_q":[str(v) for v in ex["q"]],
                "pred_q":list(predicted),
                "teacher_engages":truth,"student_engages":guessed,
                "teacher_action_informative":q[0]!=q[1],
                "exact_teacher_regret":str(max(q)-q[chosen]),
                "teacher_gap_evaluations":ex["teacher_gap_evaluations"],
            })
        if t==checkpoints[-1]:
            break
        i,a,bit=ledger_receipts[t]
        ledger.observe(i,a,bit)
    if tuple(ledger.receipts)!=tuple(ledger_receipts):
        raise AssertionError("posthoc visited-state audit changed observed streams")
    return tuple(rows)


def _calibration_metrics(rows):
    tp=fp=tn=fn=0
    informative=disagree=0
    raw_regret=F(0)
    for row in rows:
        truth=row["teacher_engages"]
        pred=row["student_engages"]
        if truth and pred:tp+=1
        elif pred:fp+=1
        elif truth:fn+=1
        else:tn+=1
        if row["teacher_action_informative"]:
            informative+=1
            disagree+=int(F(row["exact_teacher_regret"])>0)
            raw_regret+=F(row["exact_teacher_regret"])
    return {
        "eligible_visited_samples":len(rows),
        "true_positive":tp,"false_positive":fp,
        "true_negative":tn,"false_negative":fn,
        "precision":tp/(tp+fp) if tp+fp else None,
        "recall":tp/(tp+fn) if tp+fn else None,
        "false_engagement_rate":fp/(fp+tn) if fp+tn else None,
        "missed_engagement_rate":fn/(fn+tp) if fn+tp else None,
        "informative_decisions":informative,
        "informative_decision_mistakes":disagree,
        "mean_informative_true_regret":(
            float(raw_regret/informative) if informative else None),
        "posthoc_exact_gap_evaluations":sum(
            r["teacher_gap_evaluations"] for r in rows),
    }


def full_report():
    trained=train_models()
    historical,historical_hash=load_qo16_baselines()
    models={**historical,**trained["new_models"]}
    # Source-frozen evaluation import only AFTER all fit() calls.
    from research.ingression_dynamics_v1.qo17_frozen_contract import (
        FROZEN_SCENARIOS,FROZEN_SUPERVISED_SEEDS,FROZEN_POLICY_SEEDS,
        SUPERVISED_CHECKPOINTS,POLICY_CHECKPOINTS,
        VISITED_CAL_CHECKPOINTS,ARMS,MAIN_STUDENT,
        TRAIN_OCCUPANCY_SEEDS,VALIDATION_OCCUPANCY_SEEDS,
        generate_frozen_supervised_ledgers, frozen_contract,contract_sha256,
    )
    if OCC_SEEDS!={"train":TRAIN_OCCUPANCY_SEEDS,
                  "validation":VALIDATION_OCCUPANCY_SEEDS} or (
                      OCC_CHECKPOINTS!=(16,24,32,40,56,72,96,128)):
        raise AssertionError("occupancy provenance differs from first frozen commit")
    if tuple(models)!=LEARNED_ARMS or ARMS!=(
            ("bottleneck","exact_gap_h2")+LEARNED_ARMS):
        raise AssertionError("model or frozen arm ordering drifted")
    frozen_ledgers,failures=generate_frozen_supervised_ledgers()
    supervised=[]
    for idx,seed,schedule,t,ledger in frozen_ledgers:
        if certificate(ledger.snapshot(),1,TOLERANCE)["decision"]=="undetermined":
            supervised.append(example_from_ledger(
                ledger,"qo17_frozen_test",idx,seed,schedule,t))
    if not supervised:
        raise AssertionError("no unresolved frozen-test states")
    supervised_stats=source_stats(supervised,failures)
    supervised_metrics={arm:metric_for_student(model,supervised)
                        for arm,model in models.items()}
    rows,traces=[],[]
    costs={arm:_cost() for arm in ARMS}
    audits={arm:[] for arm in LEARNED_ARMS}
    for ci,case in enumerate(FROZEN_SCENARIOS):
        for seed in FROZEN_POLICY_SEEDS:
            for arm in ARMS:
                if arm=="bottleneck":
                    observed,receipts=run_bottleneck(
                        ci,case,seed,"bottleneck",POLICY_CHECKPOINTS)
                    cost=_cost()
                    observed=[
                        dict(row,policy=arm,
                             certification_gap=str(
                                 F(row["robust_upper"])-
                                 F(row["true_optimum_lower"])),
                             policy_cost=dict(cost))
                        for row in observed]
                else:
                    mapped,selected_models=_mapped_policy(
                        arm,models.get(arm))
                    observed,receipts,cost=run_gap_policy(
                        ci,case,seed,mapped,selected_models,POLICY_CHECKPOINTS)
                    observed=[dict(row,policy=arm) for row in observed]
                if len(receipts)!=POLICY_CHECKPOINTS[-1] or not (
                        verify_common_stream_prefix(ci,case,seed,receipts)):
                    raise AssertionError("paired Bernoulli histories not replayable")
                rows.extend(observed)
                traces.append({
                    "scenario":case["id"],"seed":seed,"policy":arm,
                    "queries":len(receipts),
                    "sha256":sha256(json.dumps(
                        receipts,separators=(",",":")).encode("ascii")).hexdigest(),
                })
                for key in costs[arm]:
                    costs[arm][key]+=cost[key]
                if arm in LEARNED_ARMS:
                    audits[arm].extend(visited_calibration(
                        receipts,models[arm],case,seed,arm,
                        VISITED_CAL_CHECKPOINTS))
    groups=[]
    for at in POLICY_CHECKPOINTS:
        for arm in ARMS:
            subset=[r for r in rows if r["queries"]==at and r["policy"]==arm]
            if len(subset)!=len(FROZEN_SCENARIOS)*len(FROZEN_POLICY_SEEDS):
                raise AssertionError("incomplete paired policy groups")
            groups.append({
                "queries":at,"policy":arm,"n":len(subset),
                "correct_certificates":sum(
                    r["decision"]!="undetermined" and not r["false_cert"]
                    for r in subset),
                "false_certificates":sum(r["false_cert"] for r in subset),
                "coverage_failures":sum(not r["true_rates_covered"]
                                        for r in subset),
                "mean_certification_gap":str(_mean(
                    r["certification_gap"] for r in subset)),
                "mean_true_frozen_error":str(_mean(
                    r["true_frozen_error"] for r in subset)),
            })
    # Bootstrap uses frozen 20261104 seed and scenario-cluster resampling.
    paired=_paired_comparisons(
        rows,ARMS,POLICY_CHECKPOINTS,FROZEN_SCENARIOS,
        FROZEN_POLICY_SEEDS)
    return {
        "contract":"QO-17 independent source-frozen occupancy and margin/gate ablation",
        "frozen_contract":frozen_contract(),
        "frozen_contract_sha256":contract_sha256(),
        "historical_qo16_receipt_sha256":historical_hash,
        "training":trained["train_metadata"],
        "validation":trained["validation_metadata"],
        "fits":trained["fits"],
        "new_model_weights":trained["new_models"],
        "fixed_qo16_model_weights":historical,
        "parameter_counts":{a:trainable_count(m) for a,m in models.items()},
        "inference_macs":{a:inference_multiplies(m) for a,m in models.items()},
        "frozen_test_examples":len(supervised),
        "frozen_test_stats":supervised_stats,
        "frozen_test_failures":list(failures),
        "frozen_test_labels_fingerprint":sha256(json.dumps(
            [(ex["rate_index"],ex["seed"],ex["schedule"],ex["time"],
              [str(x) for x in ex["raw_q"]])
             for ex in supervised],separators=(",",":")).encode("ascii")).hexdigest(),
        "supervised_metrics":supervised_metrics,
        "visited_calibration":{
            arm:{
                "metrics":_calibration_metrics(samples),
                "samples":samples,
            } for arm,samples in audits.items()
        },
        "policy_arms":list(ARMS),
        "policy_groups":groups,
        "policy_costs":costs,
        "paired_comparisons":paired,
        "primary":{
            "policy":MAIN_STUDENT,"budget":POLICY_CHECKPOINTS[-1],
            "paired":next(x for x in paired if x["budget"]==POLICY_CHECKPOINTS[-1]
                          and x["policy"]==MAIN_STUDENT),
        },
        "policy_rows":rows,
        "trajectory_hashes":traces,
        "counts":{
            "trajectories":len(traces),
            "snapshots":len(rows),
            "queries":sum(r["queries"] for r in traces),
        },
        "limits":"Digital two-state one-action QO-07 IID-addressable model; exact H2 progress planner is not long-horizon optimal; no LLM/biology transfer or model-level uniform error certificate.",
    }
