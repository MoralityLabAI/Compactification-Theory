"""SHAFT-COMP-19: standard-library audit of frozen capacity-matched results.

Checks exact positive-negative Mann-Whitney counts and preregistered thresholds,
without running or importing any neural network code.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path("research/shaft_compactification")
RESULT = ROOT / "results/comp19_summary.json"
THRESHOLD = ROOT / "results/comp19_calibration_thresholds.json"

def mann_whitney(positives, negatives, strict_wins, exact_ties):
    total = positives * negatives
    if total <= 0 or min(strict_wins, exact_ties) < 0 or strict_wins + exact_ties > total:
        raise ValueError("invalid exact AUC pair counts")
    return (strict_wins + exact_ties / 2) / total

def audit(result=RESULT, threshold=THRESHOLD):
    data = json.loads(Path(result).read_text())
    frozen = json.loads(Path(threshold).read_text())
    assert data["preregistration_commit"] == frozen["registration_commit"]
    assert data["thresholds_freeze_commit"] == "671a60e3f5502b9c11c66292ea900af58632384e"
    assert data["calibration_model_seeds"] == frozen["calibration_model_seeds"]
    assert data["fixed_source_thresholds"] == frozen["pooled_thresholds"]
    assert data["source_script_sha256"] == frozen["source_code_sha256"]
    assert data["full_pair_receipt_sha256"] == data["repeated_full_receipt_sha256"]
    assert data["heldout"]["eligible"] == 968 and data["heldout"]["positives"] == 323
    counts = data["exact_auc_comparisons"]["heldout"]
    assert counts["positive"] + counts["negative"] == data["heldout"]["eligible"]
    aucs = {}
    for name in ("history", "duplicate"):
        value = mann_whitney(
            counts["positive"], counts["negative"],
            counts[f"{name}_strict_wins"], counts[f"{name}_ties"]
        )
        assert abs(value - data["heldout"]["auc"][name]) < 1e-6
        aucs[name] = round(value, 6)
    assert abs((aucs["history"] - aucs["duplicate"]) - data["heldout"]["primary_delta"]) < 1e-6
    assert data["inference"]["permutation_p_two_sided"] == 0.003996
    assert data["inference"]["bootstrap_95_delta"][0] > 0
    assert data["inference"]["positive_cell_count"] == 4
    assert data["operating"]["history_fixed_source_fpr_pass_all_four_cells"]
    assert sum(x["positives"] for x in data["heldout_cells"].values()) == 323
    assert sum(x["history_fixed_true_positives"] for x in data["heldout_cells"].values()) == 27
    assert sum(x["history_fixed_false_positives"] for x in data["heldout_cells"].values()) == 35
    for cell in data["heldout_cells"].values():
        assert cell["delta"] > 0
        assert cell["history_fixed_fpr"] <= .15
    script = ROOT / "experiments/capacity_history.py"
    if script.exists():
        assert hashlib.sha256(script.read_bytes()).hexdigest() == data["source_script_sha256"]
    return {"aucs":aucs,"delta":round(aucs["history"]-aucs["duplicate"],6),
            "eligible":968,"positives":323,"fixed_false_positives":35,"fixed_true_positives":27}

if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
