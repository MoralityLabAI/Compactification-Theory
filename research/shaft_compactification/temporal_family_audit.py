"""SHAFT-COMP-16: stdlib audit of frozen temporal-family AUC and FPR results.

The audit reconstructs heldout positive-negative Mann–Whitney scores from
integer strict-win and tie counts. The source threshold remains unmodified.
No network, model training or PyTorch dependency.
"""
import json
from pathlib import Path

PATH=Path("research/shaft_compactification/results/comp16_summary.json")
FEATURES=("learned_direction","learned_rank","native_direction","native_rank",
          "learned_direction_normed","state_distance")
SOURCE_THRESHOLDS={"learned_direction":6.854760490520719,"learned_rank":2.5}


def auc_from_counts(positive,negative,wins,ties):
    total=positive*negative
    if total<1 or wins<0 or ties<0 or wins+ties>total:
        raise ValueError("invalid Mann–Whitney counts")
    return (wins+ties/2)/total


def verify(path=PATH):
    r=json.loads(Path(path).read_text())
    assert r["experiment"]=="SHAFT-COMP-16"
    assert r["protocol_commit"]=="83a436d3e301002491655a1d96af7bf1df88e4a7"
    assert r["calibration_source"]["thresholds"]==SOURCE_THRESHOLDS
    assert r["input_family_training"]["training_steps"]==1200
    assert r["input_family_training"]["per_goal"]==16
    assert len(r["model_hashes"])==8
    assert all(v["accuracy_12_steps_fresh"]>=.90 for v in r["model_training"].values())
    assert r["full_receipt_sha256"] and len(r["full_receipt_sha256"])==64
    for fam in ("lag_rotate","lag_match"):
        assert r["replays"][fam+"_sha256"]==r["per_family_receipt_sha256"][fam]
    out={}
    for label,c in sorted(r["exact_auc_counts"].items()):
        pos,neg=c["positive"],c["negative"]
        out[label]={}
        for metric in FEATURES:
            strict,ties=c["scores"][metric]
            value=auc_from_counts(pos,neg,strict,ties)
            out[label][metric]=round(value,6)
            parts=label.rsplit("_",1)
            if parts[0]=="pooled":
                if parts[1]=="heldout":
                    expected=r["heldout_pooled"]["auc"][metric]
                else:
                    continue
            else:
                expected=r["families"][parts[0]][parts[1]]["auc"][metric]
            assert abs(value-expected)<1e-6,(label,metric,value,expected)
    assert r["heldout_pooled"]["eligible"]==371
    assert r["heldout_pooled"]["positives"]==28
    assert r["families"]["lag_rotate"]["source_thresholds"]["learned_direction"]["false_positives"]==101
    assert r["families"]["lag_rotate"]["source_thresholds"]["learned_direction"]["negatives"]==195
    assert r["families"]["lag_match"]["source_thresholds"]["learned_direction"]["false_positives"]==17
    assert not r["decision"]["source_threshold_fpr_transfer_supported"]
    assert not r["decision"]["directional_prediction_transfer_supported"]
    assert r["inference"]["bootstrap_95"][0] < 0 < r["inference"]["bootstrap_95"][1]
    assert out["pooled_heldout"]["learned_direction"]==.551853
    assert out["pooled_heldout"]["learned_rank"]==.4293
    return {"verified":True,"new_family_heldout":out["pooled_heldout"],
            "source_threshold_failure":{"lag_rotate_fpr":101/195,"lag_match_fpr":17/148},
            "transfer_criteria_met":False}


if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True,indent=2))
