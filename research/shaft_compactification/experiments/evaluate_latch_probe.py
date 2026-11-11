"""SHAFT-COMP-21: evaluate compact probes only after 90% native task gate.

The model and its parameters remain intact throughout. Heldout evaluation
requires the previously frozen, SHA-256-verified calibration receipt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from research.shaft_compactification.experiments import architecture_vocab_transfer as base
from research.shaft_compactification.experiments import persistent_goal_control as comp20
from research.shaft_compactification.experiments.aux_latch_learning import (
    REGISTRATION,V,FAMILIES,ARMS,
)


def evaluate_checkpoint(family,seed,phase,arm,model_receipt,weights_path):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    prior=json.loads(Path(model_receipt).read_text())
    if (prior["family"],prior["seed"],prior["phase"],prior["arm"])!=(family,seed,phase,arm):
        raise ValueError("model receipt mismatch")
    result={"experiment":"SHAFT-COMP-21","phase":phase,"family":family,
            "seed":seed,"arm":arm,"base_gate_passed":prior["probe_gate_passed"],
            "registered_gate_accuracy_d12":prior["accuracy_d12"]["bit_accuracy"],
            "probe_state":"not_run_gate_fail","reg_commit":REGISTRATION}
    if not prior["probe_gate_passed"]:
        return result
    ck=torch.load(weights_path,map_location="cpu",weights_only=True)
    assert ck["config"]["seed"]==seed and ck["config"]["arm"]==arm
    model=base.RecurrentMemory("gru",V)
    model.load_state_dict(ck["base"])
    model.eval()
    assert base.stable_sha(model)==prior["model_sha256"]
    probes,ncontexts=comp20.train_probes(model,seed,family)
    rows,details=[],{}
    for depth in comp20.STATE_DEPTHS:
        batch,checks=comp20.make_pair_rows(model,probes,seed,depth,family,comp20.PER_GOAL)
        rows.extend(batch)
        details[str(depth)]=checks
    result.update({
        "probe_state":"completed","probe_training_contexts":ncontexts,
        "probe_hashes":{key:base.stable_sha(p) for key,(p,_) in probes.items()},
        "probe_loss":{key:loss for key,(_,loss) in probes.items()},
        "depth_checks":details,"summary":comp20.summarize(rows),
        "rows":rows,
    })
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--family",choices=tuple(FAMILIES),required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--phase",choices=("calibration","heldout"),required=True)
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--model-receipt",type=Path,required=True)
    parser.add_argument("--weights",type=Path)
    parser.add_argument("--frozen-gate",type=Path)
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    if a.seed not in FAMILIES[a.family][a.phase]:
        raise ValueError("unregistered seed")
    if a.phase=="heldout":
        if not a.frozen_gate or not a.frozen_sha256:
            raise ValueError("missing prior calibration gate")
        raw=a.frozen_gate.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=a.frozen_sha256:
            raise ValueError("calibration SHA-256 mismatch")
        if json.loads(raw).get("registration_commit")!=REGISTRATION:
            raise ValueError("invalid prior calibration")
    prior=json.loads(a.model_receipt.read_text())
    if prior["probe_gate_passed"] and not a.weights:
        raise ValueError("gated model needs frozen weights")
    result=evaluate_checkpoint(a.family,a.seed,a.phase,a.arm,a.model_receipt,a.weights)
    payload=json.dumps(result,indent=2,sort_keys=True)+"\n"
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(payload)
    print(json.dumps({
        "family":a.family,"seed":a.seed,"arm":a.arm,"phase":a.phase,
        "gate":result["base_gate_passed"],"probe":result["probe_state"],
        "eligible":result.get("summary",{}).get("eligible"),
        "positives":result.get("summary",{}).get("positives"),
    }),flush=True)


if __name__=="__main__":
    main()
