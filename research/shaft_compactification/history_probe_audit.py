"""Pure-standard-library SHAFT-COMP-17 receipt audit.

Independently reconstructs the frozen Mann–Whitney AUCs from positive-negative
pair counts and tests calibration/holdout threshold integrity. No PyTorch.
"""
import json
from pathlib import Path
DEFAULT="research/shaft_compactification/results/comp17_summary.json"

def mann_whitney(positive,negative,strict_wins,ties):
    if positive<1 or negative<1 or strict_wins<0 or ties<0 or strict_wins+ties>positive*negative:
        raise ValueError("invalid exact AUC counts")
    return (strict_wins+ties/2)/(positive*negative)

def audited_rate(successes,denominator):
    if denominator<1 or not 0<=successes<=denominator:
        raise ValueError("invalid rate")
    return successes/denominator

def audit(path=DEFAULT):
    s=json.loads(Path(path).read_text())
    assert s["experiment"]=="SHAFT-COMP-17"
    assert s["registration_commit"]=="938560c3cad9d2d8cf298f349311900a2510e0fe"
    assert s["threshold_freeze_commit"]=="603eb2640f5f4f98c99630495249d005c9dd2729"
    assert s["sha256_full_json"]==s["sha256_replay_json"]
    for phase in ("calibration","heldout"):
        counts=s["exact_auc_counts"][phase+"_pooled"]
        assert counts["positive"]==s[phase]["positives"]
        assert counts["positive"]+counts["negative"]==s[phase]["eligible"]
        for key,item in counts["scores"].items():
            value=mann_whitney(counts["positive"],counts["negative"],
                               item["strict_wins"],item["ties"])
            assert abs(value-s[phase]["auc"][key])<1e-6,(phase,key,value)
    for family in ("xor_lag","parity_gate"):
        counts=s["exact_auc_counts"]["heldout_"+family]
        original=s["heldout_families"][family]
        assert counts["positive"]==original["positives"]
        assert counts["positive"]+counts["negative"]==original["eligible"]
        for key,item in counts["scores"].items():
            value=mann_whitney(counts["positive"],counts["negative"],
                               item["strict_wins"],item["ties"])
            assert abs(value-original["auc"][key])<1e-6
    for key in ("history_direction","instant_direction"):
        cal=s["frozen_calibration_operating_points"][key]
        hold=s["heldout_fixed_operating_points"][key]
        assert cal["threshold"]==hold["threshold"]
        assert cal["negatives"]==437 and hold["negatives"]==492
        assert audited_rate(cal["false_positives"],cal["negatives"])<=.1
        assert audited_rate(hold["false_positives"],hold["negatives"])<.15
    assert len(s["heldout_model_signs"])==4
    assert all(r["auc_difference"]>0 for r in s["heldout_model_signs"])
    assert s["inference"]["bootstrap_95"][0]>0
    assert s["decision"]["prediction_transfer_supported"]
    assert s["decision"]["fixed_fpr_transfer_supported"]
    return {"calibration_auc":s["calibration"]["auc"],
            "heldout_auc":s["heldout"]["auc"],
            "auc_difference":s["inference"]["auc_difference"],
            "history_fpr":audited_rate(s["heldout_fixed_operating_points"]["history_direction"]["false_positives"],492),
            "history_tpr":audited_rate(s["heldout_fixed_operating_points"]["history_direction"]["true_positives"],31)}

if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
