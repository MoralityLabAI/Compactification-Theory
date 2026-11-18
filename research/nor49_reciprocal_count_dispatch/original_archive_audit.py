"""Read-only SHA and gate audit of the first NOR49 timing-clean CPU FAIL."""
from __future__ import annotations
import json, math, statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent / "results"
RAW_SHA = "c3f4a67b20df4504461c65168f6006041f2cc839aabdda054032a6f89bd24e4b"
ROSTER_SHA = "40929d52abddc5a15e82ec8e8101fba265d39e28239172c7e9fcaca4f6f0d2ae"
REGISTERED = "26b70716ec3d3830031919e593d3092cc31ae162"
TRAIN_SHA = "d5cf4b0e3860907eb58c2a37b84ff631f023caf2b69e6e0d0e5a322e2f0b2b99"
GATES = {
    "median_nor49_over_direct_less_than_0p75": False,
    "median_nor49_over_nor48_less_than_0p95": False,
    "signed_program_r_count_only_routing": True,
    "all_exact_source_program_and_roots": True,
    "first_source_freeze_reentrant_and_five_repeats": True,
}
ARMS = ("nor45_direct","nor47_gmp_full_trace",
        "nor48_any_R_dispatch","nor49_multiple_R_dispatch")
REPS = (101,102,103,105,106,107,109,110,111,113,114,115)

def audit(directory=HERE):
    raw = (directory / "nor49_first_original_cpu.json").read_bytes()
    roster_raw = (directory / "nor49_first_original_source_roster.json").read_bytes()
    receipt = json.loads((directory / "nor49_first_receipt.json").read_bytes())
    assert len(raw) == 84469 and sha256(raw).hexdigest() == RAW_SHA
    assert len(roster_raw) == 13081 and sha256(roster_raw).hexdigest() == ROSTER_SHA
    x, roster = json.loads(raw), json.loads(roster_raw)
    assert x["schema"] == "nor49.first_timing_clean_48_source_r_count_dispatch_cpu.v1"
    assert roster["schema"] == "nor49.first_fixed_timing_clean_nor42_source_roster.v1"
    assert x["registration_first_commit"] == roster["first_registration_sha"] == REGISTERED
    assert x["old_first_NOR48_training_sha256"] == TRAIN_SHA
    assert x["timing_clean_first_roster_sha256"] == ROSTER_SHA
    assert x["source_pairs"] == 48 and x["new_source_identities"] == 0
    assert x["repeat_count"] == 5 and x["five_repeat_interleaved_arms"] == list(ARMS)
    assert roster["timing_clean_source_count"] == 48
    assert roster["training_source_count"] == 16 and roster["no_holdout_timing_observed"]
    assert roster["selected_test_replicates"] == list(REPS)
    assert x["first_engineering_gates"] == GATES
    assert x["first_engineering_pass"] is False
    assert receipt["schema"] == "nor49.original_cpu_first_archive.v1"
    assert receipt["first_run"] == 44256305944
    assert receipt["first_artifact_id"] == 14405524398
    assert receipt["original_raw_sha256"] == RAW_SHA
    assert receipt["original_roster_sha256"] == ROSTER_SHA
    assert receipt["original_gates"] == GATES
    assert receipt["first_engineering_pass"] is False
    rows = x["source_records"]
    assert len(rows) == 48
    ids = [r["source_sha256"] for r in rows]
    assert len(set(ids)) == 48
    assert set(ids) == {r["source_sha256"] for r in roster["cases"]}
    assert Counter(r["replicate"] for r in rows) == {n:4 for n in REPS}
    assert Counter(r["active_R_count_from_authorized_program"] for r in rows) == {0:19,1:15,2:12,3:2}
    for r in rows:
        assert r["all_exact_outputs_roots_identical"]
        n = r["active_R_count_from_authorized_program"]
        ops = r["signed_program_ops"]
        recount = sum(ops[k] == "R" for k in ("base","u","v"))
        if ops["hidden_beta_nonzero"]:
            recount += (ops["hidden_transform"] == "R")
        assert n == recount
        assert r["program_backend"] == ("gmp_full_trace" if n >= 2 else "nor45_direct")
        for a in ARMS:
            t = r["interleaved_five_sample_ms"][a]
            assert len(t) == 5 and all(math.isfinite(v) and v > 0 for v in t)
            assert math.isclose(statistics.median(t),r["arm_median_ms"][a],rel_tol=1e-12)
        med = r["arm_median_ms"]
        assert math.isclose(r["nor49_over_direct"], med[ARMS[3]]/med[ARMS[0]],rel_tol=1e-12)
        assert math.isclose(r["nor49_over_nor48"],med[ARMS[3]]/med[ARMS[2]],rel_tol=1e-12)
    p = statistics.median(r["nor49_over_direct"] for r in rows)
    s = statistics.median(r["nor49_over_nor48"] for r in rows)
    assert math.isclose(p,0.9995949312073448,abs_tol=1e-12)
    assert math.isclose(s,0.9999505379197369,abs_tol=1e-12)
    totals = {arm:sum(r["arm_median_ms"][arm] for r in rows) for arm in ARMS}
    assert math.isclose(totals[ARMS[3]]/totals[ARMS[0]],0.7301753182536326,abs_tol=1e-12)
    assert math.isclose(totals[ARMS[3]]/totals[ARMS[2]],0.9094039954203512,abs_tol=1e-12)
    return {"original_sha":RAW_SHA,"first_registered_verdict":"FAIL (3/5)",
            "original_medians":(p,s),"posthoc_workload_ratios":(
             totals[ARMS[3]]/totals[ARMS[0]],totals[ARMS[3]]/totals[ARMS[2]])}

if __name__ == "__main__":
    print(audit())
