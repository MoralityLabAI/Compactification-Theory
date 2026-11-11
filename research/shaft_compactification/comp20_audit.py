"""Independent SHAFT-COMP-20 AUC and exact automaton audit (stdlib only)."""
from pathlib import Path
import json
from research.shaft_compactification.persistent_latch_finite import exact_reference

def auc(positive,negative,wins,ties):
    pair_count=positive*negative
    if pair_count<=0 or min(wins,ties)<0 or wins+ties>pair_count:
        raise ValueError("invalid Mann–Whitney count")
    return (wins+ties/2)/pair_count

def audit(root=Path("research/shaft_compactification/results")):
    root=Path(root)
    r=json.loads((root/"comp20_summary.json").read_text())
    exact=json.loads((root/"comp20_exact_finite.json").read_text())
    freeze=json.loads((root/"comp20_frozen_calibration.json").read_text())
    assert exact==exact_reference()
    assert freeze["registration_commit"]==r["registration_commit"]
    assert freeze["phase"]=="calibration_thresholds"
    assert freeze["pooled_thresholds"]==r["fixed_thresholds"]
    assert (r["heldout"]["eligible"],r["heldout"]["positives"])==(1016,200)
    computed={}
    for phase in ("calibration","heldout"):
        computed[phase]={}
        for feature,cell in r["eligible_primary_pair_counts"][phase].items():
            value=auc(cell["positive"],cell["negative"],cell["wins"],cell["ties"])
            assert abs(value-r[phase]["auc"][feature])<1e-6,(phase,feature,value)
            computed[phase][feature]=round(value,6)
    assert abs(computed["heldout"]["history"]-computed["heldout"]["goal"]-r["uncertainty"]["primary_delta_auc"])<1e-6
    assert r["uncertainty"]["bootstrap95"][0]<0<r["uncertainty"]["bootstrap95"][1]
    assert not r["supported_neural_transfer"] and not r["fpr_transfer_pass"]
    assert all(m["accuracy_d12"]<.90 for m in r["models"].values())
    return {"computed_aucs":computed,"negative_primary_confirmed":True,
            "exact_state_count":768,"trained_seed_count":len(r["models"])}

if __name__=="__main__":
    print(json.dumps(audit(),sort_keys=True,indent=2))
