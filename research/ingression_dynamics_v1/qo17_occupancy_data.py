"""QO-17 occupancy data strictly from training/validation source families.

Never imports QO-17 frozen test generator. Two predictable source-policy
trajectories (QO-08 bottleneck, exact QO-12 gap-progress teacher) visit
different evidence states, using independent per-cell Bernoulli streams.
Each selected eligible public state is labeled by the unchanged QO-12
exact teacher, with all visited state receipts kept for auditing.
"""
from fractions import Fraction as F
from random import Random
from hashlib import sha256
import json

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, choose_probe, certificate,
)
from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA,BISECTION_BITS,TOLERANCE,TRAIN_RATES,VALIDATION_RATES,
)
from research.ingression_dynamics_v1.gap_student_data import (
    example_from_ledger,train_validation,source_stats,
)
from research.ingression_dynamics_v1.gap_progress_teacher import (
    READINESS_MIN_PER_CELL,NORMALIZED_GAIN_GATE,gap_action_values,
    gain_to_target,greedy_gain_cell,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    stream_seed,verify_common_stream_prefix,
)
from research.ingression_dynamics_v1.two_step_gap_probe import planning_state

OCC_SEEDS={"train":(3101,3109),"validation":(3301,3313)}
OCC_CHECKPOINTS=(16,24,32,40,56,72,96,128)
SOURCE_POLICIES=("bottleneck","exact_gap_h2")

def source_rates(split):
    if split=="train":return TRAIN_RATES
    if split=="validation":return VALIDATION_RATES
    raise ValueError("training occupancy only supports original train/validation families")

def is_eligible(ledger):
    return min(ledger.counts[i][0] for i in (0,1))>=READINESS_MIN_PER_CELL and (
        certificate(ledger.snapshot(),1,TOLERANCE)["decision"]=="undetermined")

def _predictable_source_choice(ledger,policy,stats):
    baseline=choose_probe(ledger,1,"bottleneck")
    if policy=="bottleneck" or not is_eligible(ledger):
        stats["bottleneck_choices"]+=1
        return baseline
    if policy!="exact_gap_h2":
        raise ValueError("unknown occupancy source")
    raw,work=gap_action_values(ledger)
    stats["teacher_policy_calls"]+=1
    stats["teacher_policy_gap_evaluations"]+=work["gap_evaluations"]
    scores=tuple(gain_to_target(x) for x in raw)
    if max(scores)<NORMALIZED_GAIN_GATE:
        stats["teacher_policy_low_gain_fallbacks"]+=1
        stats["bottleneck_choices"]+=1
        return baseline
    stats["teacher_policy_chosen_actions"]+=1
    return greedy_gain_cell(ledger,scores),0

def _public_state_receipt(ledger):
    return {
        "counts":[ledger.counts[i][0] for i in (0,1)],
        "hits":[ledger.hits[i][0] for i in (0,1)],
        "bounds":[[str(ledger.bounds[i][0][0]),str(ledger.bounds[i][0][1])]
                  for i in (0,1)],
        "history_sha256":sha256(json.dumps(
            ledger.receipts,separators=(",",":")).encode("ascii")).hexdigest(),
        "probes":ledger.query_count(),
    }

def generate_occupancy(split):
    rates=source_rates(split)
    examples=[]
    trajectory_receipts=[]
    costs={
        "teacher_policy_calls":0,"teacher_policy_gap_evaluations":0,
        "teacher_policy_chosen_actions":0,
        "teacher_policy_low_gain_fallbacks":0,"bottleneck_choices":0,
        "snapshot_label_calls":0,"snapshot_gap_evaluations":0,
        "snapshot_candidate_evaluations":0,
        "ineligible_snapshots":0,
        "invalid_source_streams":0,
    }
    for idx,rate in enumerate(rates):
        case={"id":"qo17_"+split+"_rate_"+str(idx),
              "rates":((rate[0],),(rate[1],)),
              "classes":1,"tolerance":TOLERANCE}
        for seed in OCC_SEEDS[split]:
            for policy in SOURCE_POLICIES:
                ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
                streams=[Random(stream_seed(idx,seed,i,0)) for i in (0,1)]
                for t in range(OCC_CHECKPOINTS[-1]+1):
                    if t in OCC_CHECKPOINTS:
                        if is_eligible(ledger):
                            ex=example_from_ledger(
                                ledger,split+"_occupancy",idx,seed,policy,t)
                            ex["source_policy"]=policy
                            ex["public_state_receipt"]=_public_state_receipt(ledger)
                            examples.append(ex)
                            costs["snapshot_label_calls"]+=1
                            costs["snapshot_gap_evaluations"]+=ex["teacher_gap_evaluations"]
                            costs["snapshot_candidate_evaluations"]+=ex[
                                "teacher_candidate_evaluations"]
                        else:
                            costs["ineligible_snapshots"]+=1
                    if t==OCC_CHECKPOINTS[-1]:
                        break
                    cell=_predictable_source_choice(ledger,policy,costs)
                    i,a=cell
                    outcome=int(streams[i].random()<float(rate[i]))
                    ledger.observe(i,a,outcome)
                if len(ledger.receipts)!=OCC_CHECKPOINTS[-1] or not (
                        verify_common_stream_prefix(idx,case,seed,ledger.receipts)):
                    costs["invalid_source_streams"]+=1
                    raise AssertionError("source occupancy has non-replayable streams")
                trajectory_receipts.append({
                    "split":split,"rate_index":idx,"seed":seed,
                    "source_policy":policy,"queries":ledger.query_count(),
                    "sha256":sha256(json.dumps(
                        ledger.receipts,separators=(",",":")).encode("ascii")).hexdigest(),
                })
    meta=source_stats(examples,()) if examples else {}
    meta.update({
        "split":split,
        "rate_count":len(rates),"source_policies":list(SOURCE_POLICIES),
        "seeds":list(OCC_SEEDS[split]),
        "snapshot_checkpoints":list(OCC_CHECKPOINTS),
        "occupancy_eligible_examples":len(examples),
        "occupancy_trajectory_count":len(trajectory_receipts),
        "cost":costs,
        "trajectory_hashes":trajectory_receipts,
    })
    return tuple(examples),meta

def training_mixture(split):
    """One source fixed-schedule set plus one set of visited occupancy states."""
    source,fixed_stats=train_validation(split) # QO-16 source + swaps
    occupancy,occ_stats=generate_occupancy(split)
    if not source or not occupancy:
        raise AssertionError("missing split-safe occupancy examples")
    full=source+occupancy
    return full,{
        "split":split,
        "fixed_source":fixed_stats,
        "occupancy":occ_stats,
        "total_training_rows":len(full),
        "mixture":"1x QO-16 source with swaps + 1x visited eligible snapshots",
        "exact_teacher_target": "QO-12 two-step expected certification gap gain",
        "occupancy_examples":[{
            "rate_index":ex["rate_index"],"seed":ex["seed"],
            "source_policy":ex["source_policy"],"time":ex["time"],
            "raw_q":[str(x) for x in ex["raw_q"]],
            "normalized_q":[str(x) for x in ex["q"]],
            "public_evidence":ex["public_state_receipt"],
        } for ex in occupancy],
    }
