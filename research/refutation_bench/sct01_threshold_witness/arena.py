"""SCT-01B: host-owned proper-clause stream and bounded witness monitors.

A finite RefutationBench control. Input is a uniformly drawn clause-type
stream from two disjoint complete 3-SAT gadgets, with replacement. Every
policy receives one clause ID at a time and only its declared live state.

Python in-process policies are trusted for finite testing; this class
does not sandbox untrusted Python introspection or external memory.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random

try:
    from .reference import (
        R, K, M, GADGET_CLAUSES, unsat, hit_cdf,
        encode_monitor, step_monitor, certifying_initial_state,
        certifying_transition, certifying_output, assert_certificate_sound,
    )
except ImportError:
    from reference import (
        R, K, M, GADGET_CLAUSES, unsat, hit_cdf,
        encode_monitor, step_monitor, certifying_initial_state,
        certifying_transition, certifying_output, assert_certificate_sound,
    )

SEED = 1752352026
HORIZONS = (8,16,24,31,32,40,48,64)
TRIALS = 1024
POLICIES = ("exact_binary_16bit","certifying_16bit",
            "first_gadget_only_8bit","arrival_counters_8bit")


def update(policy: str, state: int, clause: int) -> int:
    if policy not in POLICIES or type(state) is not int or type(clause) is not int or (
            state<0 or clause not in range(M)):
        raise ValueError("invalid monitor action")
    if policy=="exact_binary_16bit":
        return step_monitor(state,clause)
    if policy=="certifying_16bit":
        return certifying_transition(state,clause)
    if policy=="first_gadget_only_8bit":
        return state | (1<<clause) if clause<8 else state
    # Fixed eight bits: two saturated arrival counters, four bits each.
    g=clause//GADGET_CLAUSES
    count=(state>>(g*4))&15
    return state if count>=8 else state+(1<<(g*4))


def decision(policy: str, state: int):
    if policy=="exact_binary_16bit":
        return ("UNSAT",None) if state==(255**2) else ("SAT",None)
    if policy=="certifying_16bit":
        return certifying_output(state)
    if policy=="first_gadget_only_8bit":
        return ("UNSAT",0) if state==255 else ("SAT",None)
    if policy=="arrival_counters_8bit":
        return ("UNSAT",None) if (state&15)>=8 or (state>>4)>=8 else ("SAT",None)
    raise ValueError("unknown monitor")


def state_bits(policy: str) -> int:
    if policy not in POLICIES:
        raise ValueError("unknown monitor")
    return 16 if policy.endswith("16bit") else 8


def run_episode(sequence: tuple[int,...], policy:str) -> dict:
    if not sequence or any(type(c) is not int or c<0 or c>=M for c in sequence):
        raise ValueError("nonempty valid clause sequence required")
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    state=0
    mask=0
    hits=0
    for clause in sequence:
        state=update(policy,state,clause)
        mask |= 1<<clause
        if state>=1<<state_bits(policy):
            raise AssertionError("monitor escaped declared bit alphabet")
        hits+=1
    claim,proof=decision(policy,state)
    actual=unsat(mask)
    verified=assert_certificate_sound(mask,(claim,proof)) if (
        policy in ("certifying_16bit","first_gadget_only_8bit")) else None
    if policy=="certifying_16bit" and not verified:
        raise AssertionError("exact witness monitor produced invalid proof")
    return {
        "draw_count":hits,"distinct_clauses":mask.bit_count(),
        "ground_truth_unsat":actual,
        "claim":claim,"proof_gadget":proof,
        "correct":(claim=="UNSAT")==actual,
        "false_positive":claim=="UNSAT" and not actual,
        "false_negative":claim=="SAT" and actual,
        "positive_certificate_verified":verified if claim=="UNSAT" else None,
        "terminal_state":state,
        "mutable_state_bit_budget":state_bits(policy),
        "original_clause_stream_access_at_decode":False,
        "source_verifier_owns_clause_presence":True,
    }


def fixed_sequence(seed:int,length:int)->tuple[int,...]:
    if type(seed) is not int or type(length) is not int or length<1:
        raise ValueError("seed integer and positive length required")
    rng=random.Random(seed)
    return tuple(rng.randrange(M) for _ in range(length))


def dataset_hash(sequences):
    return hashlib.sha256(json.dumps(sequences,separators=(",",":")).encode()).hexdigest()


def experiment(horizons=HORIZONS,trials=TRIALS,seed=SEED):
    if not horizons or type(trials) is not int or trials<1:
        raise ValueError("nonempty horizons and positive trials required")
    cohorts=[]
    for T in horizons:
        if type(T) is not int or T<1:
            raise ValueError("positive time horizon required")
        sequences=[
            fixed_sequence(seed+1000003*T+7919*i,T)
            for i in range(trials)]
        # OR accumulation rather than arithmetic sum handles repeated clauses.
        true_rate=F(sum(
            unsat(_mask(sequence)) for sequence in sequences),trials)
        rows=[]
        for policy in POLICIES:
            results=[run_episode(seq,policy) for seq in sequences]
            fp=sum(x["false_positive"] for x in results)
            fn=sum(x["false_negative"] for x in results)
            correct=sum(x["correct"] for x in results)
            if policy.startswith("exact_") or policy=="certifying_16bit":
                if fp or fn:
                    raise AssertionError("exact streaming oracle failed")
            if policy=="first_gadget_only_8bit" and fp:
                raise AssertionError("first-gadget-only monitor fabricated a contradiction")
            rows.append({
                "policy":policy,"mutable_state_bit_budget":state_bits(policy),
                "correct":correct,"false_positive":fp,"false_negative":fn,
                "accuracy":str(F(correct,trials)),
                "verified_positive_witnesses":sum(
                    x["positive_certificate_verified"] is True for x in results),
                "witness_attempts":sum(x["claim"]=="UNSAT" and
                    policy in ("certifying_16bit","first_gadget_only_8bit") for x in results),
            })
        actual=hit_cdf(T)
        cohorts.append({
            "horizon":T,"draws_per_episode":T,"episodes":trials,
            "sequences_sha256":dataset_hash(sequences),
            "sample_UNSAT_rate":str(true_rate),
            "exact_UNSAT_probability":str(actual),
            "exact_UNSAT_probability_float":float(actual),
            "policies":rows})
    return {
        "schema":"sct01b.sat_stream_monitor.v1",
        "status":"FROZEN_SYNTHETIC_SAT_STREAM_CONTROL",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed":seed,"trials_per_horizon":trials,
        "horizons":list(horizons),"clauses_uniform_with_replacement":True,
        "initial_state_zero":True,
        "public_rule":"two disjoint complete eight-clause three-variable gadgets",
        "source_stream_visible_to_verifier_only_after_episode":True,
        "original_learned_model_weights_used":False,
        "trained_reasoner_tested":False,
        "in_process_python_is_not_a_security_sandbox":True,
        "cohorts":cohorts}


def _mask(seq):
    m=0
    for c in seq:
        m|=1<<c
    return m


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials",type=int,default=TRIALS)
    p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    report=experiment(trials=a.trials)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    compact=[{"t":r["horizon"],
              "actual_rate":float(F(r["sample_UNSAT_rate"])),
              "exact_rate":r["exact_UNSAT_probability_float"],
              "policies":[{"name":x["policy"],"accuracy":float(F(x["accuracy"])),
                           "fp":x["false_positive"],"fn":x["false_negative"]}
                          for x in r["policies"]]}
             for r in report["cohorts"]]
    print(json.dumps({"status":report["status"],"cohorts":compact},indent=2))


if __name__=="__main__":
    main()
