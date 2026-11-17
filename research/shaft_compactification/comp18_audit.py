"""SHAFT-COMP-18: independent, standard-library receipt and ROC-AUC audit.

The numerical results are from registered synthetic GRU/LSTM runs. Audits
frozen strict positive-negative pair counts, the thresholds committed before
heldout training, and the registered cell-level decision criteria. Does not
require PyTorch or train any model.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path("research/shaft_compactification")
SUMMARY=ROOT/"results/comp18_summary.json"
THRESHOLDS=ROOT/"results/comp18_calibration_thresholds.json"


def auc_from_counts(r):
    pos=r["positives"]
    neg=r["negatives"]
    strict=r["strict_wins"]
    tied=r["ties"]
    if pos<1 or neg<1 or strict<0 or tied<0 or strict+tied>pos*neg:
        raise ValueError("invalid Mann-Whitney pair counts")
    return (strict+tied/2)/(pos*neg)


def audit(summary_path=SUMMARY, thresholds_path=THRESHOLDS):
    s=json.loads(Path(summary_path).read_text())
    t=json.loads(Path(thresholds_path).read_text())
    assert s["experiment"]=="SHAFT-COMP-18"
    assert t["registration_commit"]==s["protocol_commit"]
    assert t["calibration_full_sha256"]==s["calibration"]["full_merged_sha256"]
    assert s["calibration_threshold_freeze_commit"]=="652ee2a0f3bcadc56474267013fcdb3627ddaffd"
    assert s["computation"]["registered_holdout_seeds"]==[1812,1813,1822,1823,1832,1833,1842,1843]
    assert s["heldout"]["eligible"]==794 and s["heldout"]["positives"]==109
    expected_names=("instant","history","instant_norm","history_norm","distance")
    observed={}
    for name in expected_names:
        c=s["heldout"]["exact_pair_counts"][name]
        assert c["positives"]==s["heldout"]["positives"]
        assert c["positives"]+c["negatives"]==s["heldout"]["eligible"]
        observed[name]=auc_from_counts(c)
        assert abs(observed[name]-s["heldout"]["auc"][name])<1e-6
    assert abs(observed["history"]-observed["instant"]-s["inference"]["registered_auc_delta"])<1e-6
    assert s["inference"]["delta_bootstrap95"][0]>0
    assert s["inference"]["permutations_valid"]==1000
    assert s["inference"]["bootstraps_valid"]==1000
    assert s["preregistered_decision"]["auc_support_criteria_met"]
    assert s["preregistered_decision"]["fixed_global_fpr_transfer_criteria_met"]
    assert s["preregistered_decision"]["positive_models"]==8
    assert s["preregistered_decision"]["positive_cells"]==4
    assert s["fixed_threshold_source"]["history"]==t["global_thresholds"]["history"]
    assert s["fixed_threshold_source"]["instant"]==t["global_thresholds"]["instant"]
    assert s["global_operating"]["history"]["false_positives"]==50
    assert s["global_operating"]["history"]["true_positives"]==32
    for cell, values in s["architecture_vocabulary_cells"].items():
        assert values["auc_delta"]>0
        assert values["global_fpr"]["history"]<=.15
        assert values["frozen_threshold"]["history"]==t["global_thresholds"]["history"]
        assert all(v["delta_auc"]>0 for v in values["heldout_models"].values())
    assert all(len(v)==64 for v in s["reproducibility"].values() if isinstance(v,str) and len(v)==64)
    return {"status":"passed","heldout_auc":{k:round(v,6) for k,v in observed.items()},
            "primary_delta":round(observed["history"]-observed["instant"],6),
            "positive_cells":4,"positive_models":8,"fpr_transfer":True}


def verify_local_raw_receipts(folder, summary_path=SUMMARY):
    s=json.loads(Path(summary_path).read_text())
    root=Path(folder)
    failures=[]
    for name,expected in s["reproducibility"].items():
        if name in ("independent_replay",):continue
        p=root/f"shaft18_{name}.json"
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            failures.append(str(p))
    if failures:raise ValueError(f"missing/changed registered raw receipts: {failures}")
    return True


if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
