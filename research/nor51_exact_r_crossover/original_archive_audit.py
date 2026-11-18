"""NOR51 original first exact reciprocal CPU result: immutable FAIL (2/5).

Read-only over the first frozen source fixture, first completed raw CPU JSON,
and first archive receipt. Never regenerates or retimes any first source.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json,math,statistics

HERE=Path(__file__).resolve().parent/"results"
REGISTRATION="83fabe1c34c2bedbba35faa928349f41b964115a"
FIRST_RUN=44269058977
FIRST_CPU_ARTIFACT=14411604927
FIRST_SOURCE_ARTIFACT=14411647008
CPU_BYTES=86314
CPU_SHA="a0591f604ff0b2dfc1fa550ff87d22f62f9527b4ee96579674dee83618d7c9ea"
SOURCE_BYTES=1188767
SOURCE_SHA="0b0de5549de1e1815a40d56302cb01ca104e7ac8141f161bc8cab0b6501d2c93"
ARMS=("native_single_R_root","gmp_verify_reference_root",
      "newton_produce_verify_root")
GATES={
    "all_48_exact_inverse_merkle_and_negative_controls":True,
    "longest_m3073_gmp_verify_over_direct_under_0p75":False,
    "longest_m3073_newton_e2e_over_direct_under_0p95":False,
    "matched_long_over_short_gmp_ratio_slope_under_0p80":False,
    "public_only_source_first_reentrant_48_five_repeat_caps":True
}

def audit():
    archive=HERE/"nor51_first_original_hosted.zip"
    receipt=json.loads((HERE/"nor51_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor51.first_original_48_source_cpu_archive.v1"
    assert receipt["original_run"]==FIRST_RUN
    assert receipt["original_cpu_artifact_id"]==FIRST_CPU_ARTIFACT
    assert receipt["original_source_artifact_id"]==FIRST_SOURCE_ARTIFACT
    assert receipt["first_registration_sha"]==REGISTRATION
    assert receipt["raw_cpu_bytes"]==CPU_BYTES and receipt["raw_cpu_sha256"]==CPU_SHA
    assert receipt["raw_public_source_bytes"]==SOURCE_BYTES
    assert receipt["raw_public_source_sha256"]==SOURCE_SHA
    assert receipt["first_original_gates"]==GATES
    assert receipt["first_engineering_pass"] is False
    assert sha256(archive.read_bytes()).hexdigest()==receipt["original_archive_sha256"]
    assert archive.stat().st_size==receipt["original_archive_bytes"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor51_first_original_cpu.json","nor51_first_48_source_only.json"}
        raw=z.read("nor51_first_original_cpu.json")
        source_raw=z.read("nor51_first_48_source_only.json")
    assert len(raw)==CPU_BYTES and sha256(raw).hexdigest()==CPU_SHA
    assert len(source_raw)==SOURCE_BYTES and sha256(source_raw).hexdigest()==SOURCE_SHA
    x,fixture=json.loads(raw),json.loads(source_raw)
    assert x["schema"]=="nor51.first_source_frozen_48_exact_R_crossover_cpu.v1"
    assert fixture["schema"]=="nor51.first_public_only_48_exact_reciprocal_sources.v1"
    assert x["registration_first_commit"]==fixture["registration_first_commit"]==REGISTRATION
    assert x["first_frozen_public_only_source_sha256"]==SOURCE_SHA
    assert x["source_count"]==x["source_success_count"]==48
    assert x["source_fail_count"]==0
    assert x["new_public_only_source_identities"]==48
    assert fixture["source_count"]==48 and fixture["public_only"]
    assert fixture["no_inverse_root_trace_or_timings"]
    assert x["three_original_timed_arms"]==list(ARMS)
    assert x["same_run_interleaved_repetitions"]==5
    assert x["full_merkle_root_built_all_arms"] is True
    assert x["witness_generation_charged_in_newton_end_to_end_arm"] is True
    assert x["source_only_upstream_precedes_timing"] is True
    assert x["no_global_NOR46_verifier_override"] is True
    assert x["signed_authorization_in_kernel"] is False
    assert x["fresh_native_signed_grammar_extension"] is False
    assert x["original_five_registered_gates"]==GATES
    assert x["first_engineering_pass"] is False
    assert x["gmp_metadata"]["gmp"]["gmpy2_version"]=="2.3.2"
    rows=x["source_records"]
    assert len(rows)==48 and len({r["source_sha256"] for r in rows})==48
    assert Counter(r["length"] for r in rows)=={769:16,1537:16,3073:16}
    assert Counter(r["profile"] for r in rows)=={
        "s4-b2":12,"s4-b8":12,"s4-b16":12,"s1-b2":12}
    assert Counter(r["family"] for r in rows)=={
        "signed_random":24,"alternating":24}
    assert Counter(r["replicate"] for r in rows)=={0:24,1:24}
    public_ids={r["source_sha256"] for r in fixture["cases"]}
    assert len(public_ids)==48
    assert set(r["source_sha256"] for r in rows)==public_ids
    keyed={r["source_sha256"]:r for r in fixture["cases"]}
    for r in rows:
        inp=keyed[r["source_sha256"]]
        for col in ("replicate","length","profile","family","stride","coefficient_magnitude_bits"):
            assert r[col]==inp[col]
        assert r["source_success"] and r["full_exact_output_and_root_identical"]
        assert r["reference_max_abs_bit_length"]<=32768
        assert r["gmp_full_packed_bits"]<=120000000
        assert len(r["exact_inverse_merkle_root_sha256"])==64
        med=r["arm_medians_ms"]
        ts=r["five_interleaved_samples_ms"]
        assert set(med)==set(ts)==set(ARMS)
        for arm in ARMS:
            samples=ts[arm]
            assert len(samples)==5 and all(math.isfinite(t) and t>0 for t in samples)
            assert math.isclose(statistics.median(samples),med[arm],rel_tol=1e-12)
        assert math.isclose(r["gmp_verify_over_direct"],med[ARMS[1]]/med[ARMS[0]],rel_tol=1e-12)
        assert math.isclose(r["newton_end_to_end_over_direct"],med[ARMS[2]]/med[ARMS[0]],rel_tol=1e-12)
        assert r["outside_arm_witness_json_bytes"]>0
        assert r["outside_arm_native_reference_generation_ms"]>=0
    last=[r for r in rows if r["length"]==3073]
    assert len(last)==16
    gmp=statistics.median(r["gmp_verify_over_direct"] for r in last)
    newton=statistics.median(r["newton_end_to_end_over_direct"] for r in last)
    assert math.isclose(gmp,2.7979027868516457,abs_tol=1e-12)
    assert math.isclose(newton,7.439231736365985,abs_tol=1e-12)
    paired={}
    for r in rows:
        paired.setdefault((r["replicate"],r["profile"],r["family"]),{})[
            r["length"]]=r["gmp_verify_over_direct"]
    assert len(paired)==16 and all(len(v)==3 for v in paired.values())
    slope=statistics.median(v[3073]/v[769] for v in paired.values())
    assert math.isclose(slope,2.151120761901171,abs_tol=1e-12)
    assert math.isclose(slope,x["primary_matched_length_ratio_slope"],abs_tol=1e-12)
    assert receipt["first_original_longest_gmp_over_direct"]==gmp
    assert receipt["first_original_longest_newton_e2e_over_direct"]==newton
    assert receipt["first_original_long_short_slope"]==slope
    total=x["secondary_workload_totals_and_block_uncertainty"]["total_source_median_ms"]
    for a in ARMS:
        assert math.isclose(total[a],sum(r["arm_medians_ms"][a] for r in rows),rel_tol=1e-12)
    assert math.isclose(total[ARMS[1]]/total[ARMS[0]],3.328209112151982,abs_tol=1e-12)
    assert math.isclose(total[ARMS[2]]/total[ARMS[0]],8.498078735397563,abs_tol=1e-12)
    return {"original_run":FIRST_RUN,"first_raw_cpu_sha256":CPU_SHA,
            "first_verdict":"FAIL (2/5)", "all_exact":48,
            "longest_gmp_over_native":gmp,
            "longest_newton_e2e_over_native":newton,
            "long_vs_short_ratio_slope":slope}

if __name__=="__main__":
    print("NOR51 immutable first SHA and FAIL audit",audit())
