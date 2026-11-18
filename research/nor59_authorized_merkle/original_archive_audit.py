"""NOR59 immutable original first 60-program authenticated NOR45 PASS auditor.

Read-only SHA, signed program/source roster, template geometry, exact complete
root parity assertions and first five source-median gate replay. Does not
retime, regenerate first public-only artifacts, sign programs or reexecute a
first CPU benchmark. All original bytes preserved in the source-controlled ZIP.
"""
from __future__ import annotations
import json,math,statistics,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor59_authorized_merkle import cohort
HERE=Path(__file__).resolve().parent/"results"
REG="8f6eb3dccc4d4c4d9d58955387ea6f154ad05d84"
FIRST_RUN=44321699757
SOURCE_ARTIFACT=14435161653
CPU_ARTIFACT=14434139586
SOURCE_BYTES=1005307
SOURCE_SHA="be516ca2149cf6d5970842891280f7f7d3407706b9c08b02c4719d153418b0f9"
CPU_BYTES=219583
CPU_SHA="c56d65e676e7fd81668d3919eb9cbf29922fbe34cfe350418e3c67ae87c62b2b"
ARMS=("original_authorized_full_root","warm_authorized_static_root","cold_authorized_static_root")
GATES={
 "forty_sparse_authorized_warm_over_original_median_under_1p02":True,
 "forty_sparse_NOR45_root_only_warm_over_original_under_0p70":True,
 "twenty_dense_cold_whole_authorized_over_original_under_1p25":True,
 "all_60_signed_authorized_full_output_root_security_and_support":True,
 "first_60_public_program_sources_528_prior_exclusions_full_archive":True
}
PRIMARY={
 "sparse_warm_signed_full_over_original":0.9591404063964948,
 "sparse_root_only_warm_over_original":0.3095284814997522,
 "dense_cold_signed_full_over_original":1.0045494865756202,
 "all_60_sum_source_median_signed_warm_over_original":7637.712898/7702.7641459999995
}

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def audit():
    archive=HERE/"nor59_first_original_hosted.zip"
    receipt=json.loads((HERE/"nor59_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor59.first_original_60_signed_NOR45_CPU_archive.v1"
    assert receipt["original_first_run"]==FIRST_RUN
    assert receipt["original_source_artifact_id"]==SOURCE_ARTIFACT
    assert receipt["original_CPU_artifact_id"]==CPU_ARTIFACT
    assert receipt["original_first_registration_commit"]==REG
    assert receipt["raw_public_source_bytes"]==SOURCE_BYTES
    assert receipt["raw_public_source_sha256"]==SOURCE_SHA
    assert receipt["raw_first_CPU_bytes"]==CPU_BYTES
    assert receipt["raw_first_CPU_sha256"]==CPU_SHA
    assert receipt["original_first_five_gates"]==GATES
    assert receipt["first_original_engineering_pass"] is True
    assert receipt["original_primary"]==PRIMARY
    assert archive.stat().st_size==receipt["archive_zip_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["archive_zip_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor59_first_original_cpu.json","nor59_first_original_60_public_only.json"}
        cpu=z.read("nor59_first_original_cpu.json")
        source=z.read("nor59_first_original_60_public_only.json")
    assert len(cpu)==CPU_BYTES and sha256(cpu).hexdigest()==CPU_SHA
    assert len(source)==SOURCE_BYTES and sha256(source).hexdigest()==SOURCE_SHA
    x,fixture=json.loads(cpu),json.loads(source)
    assert canonical(x)==cpu and canonical(fixture)==source
    assert x["schema"]=="nor59.first_60_full_NOR45_signed_authorized_static_root_CPU.v1"
    assert fixture["schema"]==cohort.SCHEMA
    assert x["first_registration_sha"]==fixture["first_registration_sha"]==REG
    assert x["first_public_only_source_sha256"]==SOURCE_SHA
    assert fixture["source_domain"]==cohort.DOMAIN
    assert fixture["public_only"] and fixture["no_output_root_witness_session_signatures_templates_or_timings"]
    assert fixture["source_count"]==x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert fixture["excluded_first_source_IDs"]==x["excluded_prior_first_source_ID_count"]==528
    assert x["repeats"]==5 and x["arms"]==list(ARMS)
    assert x["first_original_five_registered_gates"]==GATES
    assert x["first_original_engineering_pass"] is True
    for field in (
        "source_first_SHA_frozen_upstream_before_reference_or_timing",
        "original_NOR45_authority_precommit_provider_checks_same_in_every_arm",
        "each_arm_recomputes_all_original_signed_program_integer_coefficients",
        "warm_template_internal_geometry_only_no_result_cache",
        "cold_template_build_charged_per_signed_verification",
        "original_NOR45_program_and_Merkle_helpers_unmodified_runtime",
        "fixture_signing_and_reference_production_outside_whole_verifier_timing",
        "stateless_NOR45_verifier_per_session_nonce_binding",
        "first_signed_authentication_private_keys_public_test_only"
    ):
        assert x[field] is True,field
    closure=x["source_free_support_closure"]
    assert closure["all_allowed_delay_and_spacing_sums_divisible_by_8"] is True
    assert closure["all_hidden_suffix_onsets_divisible_by_8"] is True
    assert closure["formal_Lean_kernel_proof_available"] is False
    sources=fixture["cases"]
    source_ids={r["source_sha256"] for r in sources}
    assert len(source_ids)==len(sources)==60
    assert source_ids.isdisjoint(cohort.historical_ids()) and len(cohort.historical_ids())==528
    original_by_sha={r["source_sha256"]:r for r in sources}
    rows=x["source_records"]
    assert len(rows)==60 and {r["source_sha256"] for r in rows}==source_ids
    assert Counter(r["observed_degree"] for r in rows)=={512:30,1024:30}
    assert Counter(r["source_stride"] for r in rows)=={1:20,4:20,8:20}
    assert Counter(r["program_profile"] for r in rows)=={p:12 for p in cohort.PROFILES}
    assert Counter(r["replicate"] for r in rows)=={0:30,1:30}
    geometry=x["geometry_template_ledger"]
    assert set(geometry)=={f"m{3*d//2+1}-s{s}" for d in cohort.DEGREES for s in cohort.STRIDES}
    for name,group in geometry.items():
        m,s=group["m"],group["source_stride"]
        assert name==f"m{m}-s{s}"
        assert m in (769,1537) and s in (1,4,8)
        assert group["trusted_geometry_only"] is True
        assert group["observed_degree"] in cohort.DEGREES
        assert group["build_ns_outside_all_timed_arms"]>=0
        assert len(group["template_SHA256"])==64
        assert group["canonical_template_bytes"]>0
        assert group["static_node_count"]>=group["static_zero_leaf_count"]
        assert group["dynamic_supported_leaf_count"]==(m-1)//s+1
        assert group["static_zero_leaf_count"]==m-group["dynamic_supported_leaf_count"]
    for row in rows:
        assert row["source_success"] is True
        assert row["all_three_full_signed_authorized_program_outputs_and_original_roots_equal"] is True
        case=original_by_sha[row["source_sha256"]]
        for field in ("observed_degree","full_length","source_stride",
                      "program_profile","coefficient_magnitude_bits","replicate","family"):
            assert row[field]==case[field]
        core={k:v for k,v in case.items() if k!="source_sha256"}
        assert sha256(canonical(core)).hexdigest()==case["source_sha256"]
        assert row["authorized_program"]==case["authorized_program"]==cohort.program(row["program_profile"])
        assert row["certified_public_support_stride"]==row["source_stride"]
        assert len(row["original_full_merkle_root_sha256"])==64
        assert row["original_program_authority_signer_public_research_fixture"] is True
        assert row["first_signed_certificate_bytes"]>0
        assert row["first_signed_verifier_receipt_bytes"]>0
        assert row["first_signed_provider_reference_bytes"]>0
        assert row["reference_program_execution_ns_outside_arms"]>=0
        assert row["reference_original_root_ns_outside_arms"]>=0
        assert row["fixture_five_signed_setup_ns_outside_arms"]>=0
        nonces=row["five_distinct_verifier_session_nonce_SHA256"]
        sessions=row["five_independent_signed_context_SHA256"]
        assert len(nonces)==len(set(nonces))==5
        assert len(sessions)==len(set(sessions))==5
        assert all(len(z)==64 for z in nonces+sessions)
        geom=geometry[row["template_geometry_only_key"]]
        assert row["template_geometry_only_key"]==f"m{row['full_length']}-s{row['source_stride']}"
        assert row["template_SHA256"]==geom["template_SHA256"]
        assert row["template_static_node_count"]==geom["static_node_count"]
        assert row["template_canonical_bytes"]==geom["canonical_template_bytes"]
        assert math.isclose(row["template_build_ms_outside_arms"],
                            geom["build_ns_outside_all_timed_arms"]/1e6,rel_tol=1e-12)
        samples,ms=row["five_interleaved_samples_ms"],row["arm_medians_ms"]
        assert set(samples)==set(ms)==set(ARMS)
        for arm in ARMS:
            times=samples[arm]
            assert len(times)==5 and all(math.isfinite(t) and t>0 for t in times)
            assert math.isclose(statistics.median(times),ms[arm],rel_tol=1e-12)
        ratios={
            "warm_over_original":ms[ARMS[1]]/ms[ARMS[0]],
            "cold_over_original":ms[ARMS[2]]/ms[ARMS[0]],
            "untimed_root_only_warm_over_original":
                row["untimed_root_only_warm_ns"]/row["untimed_root_only_original_ns"],
            "amortized_five_warm_with_template_build_over_original":
                (5*ms[ARMS[1]]+row["template_build_ms_outside_arms"])/(5*ms[ARMS[0]])
        }
        for k,val in ratios.items():
            assert math.isclose(row[k],val,rel_tol=1e-12)
    sparse=[r for r in rows if r["source_stride"]>1]
    dense=[r for r in rows if r["source_stride"]==1]
    assert len(sparse)==40 and len(dense)==20
    totals={a:sum(r["arm_medians_ms"][a] for r in rows) for a in ARMS}
    observed={
        "sparse_warm_signed_full_over_original":statistics.median(r["warm_over_original"] for r in sparse),
        "sparse_root_only_warm_over_original":statistics.median(
            r["untimed_root_only_warm_over_original"] for r in sparse),
        "dense_cold_signed_full_over_original":statistics.median(r["cold_over_original"] for r in dense),
        "all_60_sum_source_median_signed_warm_over_original":totals[ARMS[1]]/totals[ARMS[0]]
    }
    for k,v in PRIMARY.items():
        assert math.isclose(observed[k],v,abs_tol=1e-12)
    assert observed["sparse_warm_signed_full_over_original"]<1.02
    assert observed["sparse_root_only_warm_over_original"]<.70
    assert observed["dense_cold_signed_full_over_original"]<1.25
    assert receipt["original_source_median_wall_ms"]==totals
    assert x["overall"]["source_median_total_ms"]==totals
    assert x["secondary_ten_block_bootstrap"]["block_count"]==10
    return {"original_run":FIRST_RUN,"verdict":"PASS (5/5)",
            "original_CPU_SHA":CPU_SHA,"source_SHA":SOURCE_SHA,
            "primary":observed,"archive_SHA":receipt["archive_zip_sha256"]}

if __name__=="__main__":
    print("NOR59 original first NOR45 signed-authority static Merkle SHA auditor",audit())
