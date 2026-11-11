"""SHAFT-COMP-23: stdlib audit of the registered neural result.

Verifies aggregate bit counts and complete-response counts from the frozen
summary. The expensive six-model deterministic replay is opt-in; its six
per-model SHA-256 receipts are frozen in the same file.
"""
from __future__ import annotations
import json
from pathlib import Path

DEFAULT="research/shaft_compactification/results/comp23_summary.json"

def audit(path=DEFAULT):
    r=json.loads(Path(path).read_text())
    assert r["registration_commit"]=="fae6c5c6843ee8f2a8f85686012ee559dc37d281"
    assert r["frozen_calibration_commit"]=="5bb2e00777aeea7a88b5b2e374abfa21d3892647"
    assert r["seeds"]["heldout"]=={"repeat_toggle":[2302,2303],"successor_toggle":[2312,2313]}
    assert r["decoder_training"]["oracle_labels_used_in_decoder_fitting"] is False
    assert r["decoders"]["direct36"]["parameters"]==r["decoders"]["factorized900"]["parameters"]==900
    assert r["decoders"]["directmlp890"]["parameters"]==890
    assert r["heldout_depth40"]["states"]==4096
    assert r["heldout_depth40"]["total_binary_decisions"]==4096*36
    checked={}
    for depth,denominator in (("heldout_depth40",4096),("heldout_depth12",4096),("calibration_depth40",2048)):
        cell=r[depth]
        checked[depth]={}
        for name,metrics in cell["methods"].items():
            bits=metrics["bits_correct"]
            exact=metrics["exact_correct"]
            assert 0<=bits<=denominator*36
            assert 0<=exact<=denominator
            acc=bits/(denominator*36)
            exact_acc=exact/denominator
            assert abs(acc-metrics["bit_accuracy"])<1e-6,(depth,name,acc,metrics["bit_accuracy"])
            assert abs(exact_acc-metrics["exact_accuracy"])<1e-6
            checked[depth][name]=(round(acc,6),round(exact_acc,6))
    primary=checked["heldout_depth40"]["factorized900"][0]-checked["heldout_depth40"]["direct36"][0]
    assert abs(primary-r["primary"]["difference"])<1e-6
    assert r["primary"]["bootstrap95"][0]>0
    assert r["primary"]["positive_heldout_models"]==4
    assert r["primary"]["preregistered_support"]
    assert r["reproducibility"]["complete_six_model_replay_identical"]
    assert all(len(x)==64 for x in r["reproducibility"]["per_model_full_json_sha256"].values())
    assert all(m["base_acc12"]>=0.9 for m in r["heldout_models"])
    return {"status":"verified","primary_difference":round(primary,6),
            "heldout_depth40":checked["heldout_depth40"],
            "heldout_model_count":len(r["heldout_models"])}

if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
