"""SHAFT-COMP-15: independently reconstruct exact AUCs from pair counts.

Pure Python standard library, no neural-model dependency. Histograms in
the frozen receipt were computed from the complete paired-state outputs.
Each strict win contributes 1, exact tie contributes 1/2.
"""
import json
from pathlib import Path

DEFAULT="research/shaft_compactification/results/comp15_summary.json"
METRICS=("learned_rank","learned_direction","native_rank","native_direction",
         "learned_direction_normed","native_direction_normed","state_distance")


def mann_whitney(positive,negative,wins,ties):
    pairs=positive*negative
    if min(positive,negative)<1 or min(wins,ties)<0 or wins+ties>pairs:
        raise ValueError("invalid pair counts")
    return (wins+ties/2)/pairs


def verify(path=DEFAULT):
    result=json.loads(Path(path).read_text())
    assert result["experiment"]=="SHAFT-COMP-15"
    assert result["protocol_commit"]=="0240f2f3378647d1106111b000e0aa425cdaee27"
    assert result["heldout_seeds"]==[707,808,909,1001]
    assert result["calibration_seeds"]==[101,202]
    assert result["rank_relative_threshold"]==0.001
    assert result["diagnostic_steps"]==5
    assert result["outcome_steps"]==32
    assert result["full_receipt_sha256"]==result["independent_replay_sha256"]
    calculated={}
    for split in ("calibration","heldout"):
        counts=result["strict_win_tie_counts"][split]
        pos,neg=counts["positive"],counts["negative"]
        assert pos==result[split]["positives"]
        assert pos+neg==result[split]["eligible"]
        calculated[split]={}
        for name in METRICS:
            wins,ties=counts["scores"][name]
            auc=mann_whitney(pos,neg,wins,ties)
            expected=result[split]["auc"][name]
            if abs(auc-expected)>1e-6:
                raise AssertionError((split,name,auc,expected))
            calculated[split][name]=round(auc,6)
    expected_delta=result["heldout"]["contrasts"]["learned_direction_minus_rank"]
    calculated_delta=calculated["heldout"]["learned_direction"]-calculated["heldout"]["learned_rank"]
    assert abs(calculated_delta-expected_delta)<1e-6
    assert sum(result["heldout_by_model"][str(seed)]["primary_delta"]>0
               for seed in result["heldout_seeds"])==4
    assert result["inference"]["positive_model_count"]==4
    assert result["inference"]["primary_difference_bootstrap95"][0]>0
    assert result["inference"]["prespecified_support_criteria_met"]
    return {"aucs":calculated,"primary_difference":round(calculated_delta,6),
            "primary_criteria_met":True}


if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True,indent=2))
