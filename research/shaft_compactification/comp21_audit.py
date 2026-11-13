"""SHAFT-COMP-21: dependency-free audit of frozen capability and AUC results."""
import json
from pathlib import Path

DEFAULT=Path("research/shaft_compactification/results/comp21_summary.json")


def exact_auc(pos,neg,wins,ties):
    total=pos*neg
    if total<=0 or wins<0 or ties<0 or wins+ties>total:
        raise ValueError("invalid Mann-Whitney accounting")
    return (wins+ties/2)/total


def audit(path=DEFAULT):
    s=json.loads(Path(path).read_text())
    assert s["experiment"]=="SHAFT-COMP-21"
    assert s["registration_commit"]=="b4596c1b5c0e4a63fc4dc3a3d03af524ad041f10"
    assert s["calibration_freeze_commit"]=="051b7f9c9cc317f3c40e61fce591ecfba2b73b64"
    assert len(s["frozen_calibration_sha256"])==64
    assert s["heldout"]["aux_gate_passed"]==4
    assert s["heldout"]["baseline_gate_passed"]==3
    assert s["heldout"]["capability_support_criterion_met"]
    seeds=[x["seed"] for x in s["heldout"]["per_seed"]]
    assert seeds==[2202,2203,2212,2213]
    assert all(x["aux_gate"] for x in s["heldout"]["per_seed"])
    assert sum(x["baseline_gate"] for x in s["heldout"]["per_seed"])==3
    assert all(x["aux_d32"]>x["baseline_d32"] for x in s["heldout"]["per_seed"])
    for key in ("d12","d32"):
        pairs=s["heldout"]["per_seed"]
        a=sum(x[f"aux_{key}"] for x in pairs)/len(pairs)
        b=sum(x[f"baseline_{key}"] for x in pairs)/len(pairs)
        assert abs(a-s["heldout"][f"mean_accuracy_{key}"]["aux"])<1e-10
        assert abs(b-s["heldout"][f"mean_accuracy_{key}"]["baseline"])<1e-10
    for group in ("aux","baseline"):
        r=s["conditional_probe"]
        counts=r[f"{group}_mann_whitney"]
        pos,neg=counts["positive"],counts["negative"]
        assert pos==r[f"{group}_positives" if group=="baseline" else "aux_positives"] if group=="aux" else pos==r["baseline_gate_passed_positives"]
        expected=r[f"{group}_auc"]
        for name in ("goal","duplicate","history","state_distance"):
            key="distance" if name=="state_distance" else name
            result=exact_auc(pos,neg,counts[f"{key}_strict_wins"],counts["all_ties"])
            assert abs(result-expected[name])<1e-10,(group,name,result,expected[name])
    v=s["conditional_probe"]
    assert abs(v["history_minus_goal_auc"]-
        (v["aux_auc"]["history"]-v["aux_auc"]["goal"]))<1e-12
    assert v["history_minus_goal_95_bootstrap"][0]<0<v["history_minus_goal_95_bootstrap"][1]
    assert v["source_threshold_transfer_aux"]["pooled"]["history"]["false_positives"]==4
    assert len(s["checks"]["exact_replay_proofs"])==2
    for proof in s["checks"]["exact_replay_proofs"]:
        assert len(proof["model_receipt_sha256"])==64
        assert len(proof["probe_receipt_sha256"])==64
    return {"status":"verified","heldout_gate":4,
            "paired_d32_gain":s["heldout"]["mean_accuracy_d32"]["paired_difference"],
            "history_minus_goal_auc":v["history_minus_goal_auc"]}


if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
