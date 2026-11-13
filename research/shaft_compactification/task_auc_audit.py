"""SHAFT-COMP-14: exact rank-AUC audit from archived positive-negative pair counts.

Reconstructs empirical Mann–Whitney AUC using strictly ordered and tied
positive-negative pairs, with all counts frozen prior to CI execution.
All data are synthetic toy GRU metrics. No PyTorch or checkpoint needed.
"""
from pathlib import Path
import json

def auc_from_pair_counts(positive,negative,strict_wins,ties):
    denom=positive*negative
    if denom<=0 or strict_wins<0 or ties<0 or strict_wins+ties>denom:
        raise ValueError("invalid positive-negative pair accounting")
    return (strict_wins+ties/2)/denom

def audit(path="research/shaft_compactification/results/comp14_summary.json"):
    r=json.loads(Path(path).read_text())
    for scope in ("calibration","heldout"):
        counts=r["exact_mann_whitney_pairs"][scope]
        pos,neg=counts["positive"],counts["negative"]
        assert pos==r[scope]["positives"]
        assert pos+neg==r[scope]["eligible"]
        for metric,(wins,ties) in counts["counts"].items():
            result=auc_from_pair_counts(pos,neg,wins,ties)
            target=r[scope]["auroc"][metric]
            assert abs(result-target)<1e-6,(scope,metric,result,target)
    assert r["protocol_sha"]=="221c225d2a3b9222b0cfd8b699a07f21b293452e"
    assert r["heldout_model_seeds"]==[505,606]
    assert r["uncertainty"]["task_minus_rank_auc"]>0
    return {k:r[k]["auroc"] for k in ("calibration","heldout")}

if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
