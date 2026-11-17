"""SHAFT-COMP-13: exact Mann–Whitney AUC from fixed binned rank scores.

Pure stdlib receipt check. The bins are counts from the frozen local GRU
experiment, including only eligible equal-current-behavior state pairs.
Higher rank deficit is the preregistered positive direction. This module
does not train a model, access a network, or change the threshold.
"""
import json
from pathlib import Path

def auc_from_bins(hist):
    keys=sorted(hist,key=float)
    pos=sum(hist[k]["positive"] for k in keys)
    neg=sum(hist[k]["negative"] for k in keys)
    if pos==0 or neg==0:return None
    negatives_below=0
    wins=0
    for key in keys:
        p,n=hist[key]["positive"],hist[key]["negative"]
        wins += p*negatives_below + (p*n)/2
        negatives_below += n
    return wins/(pos*neg)

def audit(path):
    data=json.loads(Path(path).read_text())
    result={}
    for scope in ("heldout","calibration"):
        result[scope]=auc_from_bins(data["rank_score_binned_counts"][scope])
    assert abs(result["heldout"]-data["test_results"]["heldout_pooled"]["auroc"]["rank_deficit"])<1e-6
    assert abs(result["calibration"]-data["test_results"]["calibration_pooled"]["auroc"]["rank_deficit"])<1e-6
    return result

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("receipt",nargs="?",default="research/shaft_compactification/results/comp13_summary.json")
    x=p.parse_args()
    print(json.dumps(audit(x.receipt),sort_keys=True))
