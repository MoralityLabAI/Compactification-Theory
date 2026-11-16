"""Offline evidence audit: no CUDA, model loading, or inference."""
import csv
from collections import Counter, defaultdict
import json
import numpy as np
from preflight import ROOT, RESULTS, sha, save, CEILING
from core import fingerprint, LAYERS, RANKS


def read(name):
    with open(RESULTS/name,newline="") as f:
        return list(csv.DictReader(f))


def main():
    s=json.loads((RESULTS/"summary.json").read_text())
    assert s["source_hash"]==fingerprint(ROOT)[0]
    for name,digest in s["result_hashes"].items():
        assert sha(RESULTS/name)==digest,name
    data=json.loads((RESULTS/"dataset.json").read_text())
    assert Counter(x["split"] for x in data)==dict(train=128,calibration=64,heldout=96)
    assert len({x["prompt"] for x in data})==288
    assert all(32<=len(x["input_ids"])<=64 for x in data)
    assert all(2<=x["delay"]<=8 for x in data)
    a=np.load(RESULTS/"activations.npy",mmap_mode="r")
    assert a.shape==(288,3,1024) and a.dtype==np.float16 and np.isfinite(a).all()
    states=read("per_state.csv")
    assert len(states)==20160
    assert all(x["A"]==x["B"]==x["C"]=="" for x in states)
    assert all(int(x["null_count"])==128 for x in states)
    groups=defaultdict(set)
    with open(RESULTS/"null_distribution.csv",newline="") as f:
        count=0
        for x in csv.DictReader(f):
            groups[(x["state_id"],int(x["layer"]),int(x["rank"]))].add((int(x["null_index"]),x["subspace_fingerprint"]))
            count+=1
    assert count==288*3*7*128
    assert len(groups)==288*3*7
    assert all(len(v)==128 and len({p[1] for p in v})==128 for v in groups.values())
    interventions=read("causal_interventions.csv")
    assert len(interventions)==96*4*4
    primary=[x for x in interventions if x["arm"]!="exploratory_centered_residual_erasure"]
    assert len(primary)==96*4*3
    assert all(float(x["radius"])==0.5 for x in primary)
    assert all(x["effect_type"]=="conditional_residual_state_effect" for x in interventions)
    memory=json.loads((RESULTS/"memory_profile.json").read_text())
    assert memory["status"]=="complete"
    assert all(x["peak_allocated_bytes"]<CEILING for x in memory["events"])
    assert all("RTX 3050 Laptop" in x["device_name"] for x in memory["events"])
    smoke=json.loads((RESULTS/"smoke.json").read_text())
    assert smoke["status"]=="passed" and 32<=smoke["tokens"]<=64
    assert s["full_ROC_card"] is False and s["neural_same_state_jacobian"] is False
    replay=json.loads((RESULTS/"replay"/"activation_replay.json").read_text())
    assert replay["status"]=="passed"
    assert replay["source_hash"]==s["source_hash"]
    receipt=dict(status="passed",source_hash=s["source_hash"],null_rows=count,
        per_state_rows=len(states),intervention_rows=len(interventions),
        physical_route_valid_count=sum(int(x["physical_route_valid"]) for x in primary),
        peak_allocated_bytes=s["peak_allocated_bytes"],hashes_checked=len(s["result_hashes"]),
        activation_replay_status=replay["status"])
    save(RESULTS/"verification.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
