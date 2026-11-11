"""SHAFT-COMP-24: stdlib exact-count and preregistration receipt audit."""
import json
from pathlib import Path

PATH=Path("research/shaft_compactification/results/comp24_summary.json")

def validate(path=PATH):
    data=json.loads(Path(path).read_text())
    assert data["experiment"]=="SHAFT-COMP-24"
    assert data["registration_commit"]=="92ad76ece8dafe37a6181b48ca53ee52f1554059"
    assert data["calibration_freeze_commit"]=="25c07861793ed747eb8b18e36a773fca56c3d753"
    assert data["calibration_freeze_sha256"]=="81fbcb7bc21c88810c8f70f7a1f1c0edb5d8485d8bf0e4c54f21f9d23ade5c92"
    assert data["original_experiment_source_sha256"]=="fe4b10ebedcb8074614f493ebe148b492663309331bd4adce125f63c4fbfcbd2"
    assert len(data["heldout_full_receipt_sha256"])==64
    assert data["heldout_full_receipt_sha256"]==data["heldout_independent_replay_sha256"]
    assert data["heldout_total_states"]==8192
    assert len(data["per_seed_depth40"])==4
    assert all(f["oracle_rule_entries_recovered"]==36 for f in data["family_calibration"].values())
    for depth in ("12","40"):
        cell=data["metrics_by_depth"][depth]
        assert cell["states"]==4096
        assert cell["learned"]==cell["oracle"]
        assert all(cell[name]["bits_total"]==147456
            for name in ("learned","oracle","zero","wrong_family","permuted_previous"))
    d=data["metrics_by_depth"]["40"]
    a=d["learned"]["bits_correct"]/d["learned"]["bits_total"]
    b=d["zero"]["bits_correct"]/d["zero"]["bits_total"]
    delta=a-b
    assert abs(delta-data["primary"]["difference"])<1e-8
    assert d["learned"]["exact_correct"]==3137
    assert d["zero"]["exact_correct"]==0
    assert data["primary"]["stratified_bootstrap95"][0]>0
    assert data["registered_support"]["overall"]
    return {"learned_bit_accuracy":a,"zero_bit_accuracy":b,
            "primary_difference":delta,"oracle_identical":True}

if __name__=="__main__":
    print(json.dumps(validate(),sort_keys=True,indent=2))
