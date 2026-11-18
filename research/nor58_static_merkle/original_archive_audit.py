"""NOR58 independent permanent read-only first 60-source 5-of-5 PASS archive audit.

Verifies byte-exact first CPU/source JSON against SHA-pinned original ZIP,
reconstructs source identities/registered medians/geometry-only template
metadata and original five Boolean criteria. No inverse, nonce, signing,
source regeneration or first timing work occurs.
"""
from __future__ import annotations
import json,math,statistics,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor58_static_merkle import cohort

HERE=Path(__file__).resolve().parent/"results"
REG="ddbabc02cf8702fe2da2c9962174ddcfcbed39a4"
FIRST_RUN=44316658346
SOURCE_ARTIFACT=14431354312
CPU_ARTIFACT=14433176882
SOURCE_BYTES=1473865
SOURCE_SHA="6b34e964bc170f3e8efcdfae23403fead030992cf1fc08640ed13c2e34c6b5b3"
CPU_BYTES=204565
CPU_SHA="244d84accfd2821af25a3d1d98c114964a217925fa457451d2c83c1d6793342b"
ARMS=(
    "original_signed_GMP_full_Merkle",
    "warm_static_template_signed_GMP_full_Merkle",
    "cold_static_template_signed_GMP_full_Merkle"
)
GATES={
    "sparse_m3073_warm_over_original_under_0p93":True,
    "all_60_sum_source_medians_warm_over_original_under_0p95":True,
    "twelve_dense_cold_over_original_under_1p20":True,
    "all_60_full_signed_vectors_roots_and_hostile_protocol_controls":True,
    "first_60_source_468_excluded_repeats_static_template_caps_archive":True
}
PRIMARY={
    "sparse_m3073_median_sourcewise_warm_over_original":0.8024283640879615,
    "all_60_ratio_of_source_median_sums_warm_over_original":0.911459019843162,
    "dense_s1_median_sourcewise_cold_over_original":1.0326935475393229
}

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def audit():
    receipt=json.loads((HERE/"nor58_first_receipt.json").read_bytes())
    archive=HERE/"nor58_first_original_hosted.zip"
    assert receipt["schema"]=="nor58.original_first_60_static_merkle_CPU_archive.v1"
    assert receipt["first_run"]==FIRST_RUN
    assert receipt["original_source_artifact_id"]==SOURCE_ARTIFACT
    assert receipt["original_CPU_artifact_id"]==CPU_ARTIFACT
    assert receipt["first_registration_sha"]==REG
    assert receipt["raw_source_bytes"]==SOURCE_BYTES
    assert receipt["raw_source_sha256"]==SOURCE_SHA
    assert receipt["raw_CPU_bytes"]==CPU_BYTES
    assert receipt["raw_CPU_sha256"]==CPU_SHA
    assert receipt["original_first_five_registered_gates"]==GATES
    assert receipt["original_first_engineering_pass"] is True
    assert receipt["original_primary"]==PRIMARY
    assert archive.stat().st_size==receipt["archive_zip_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["archive_zip_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor58_first_original_cpu.json",
            "nor58_first_original_60_public_only.json"
        }
        cpu=z.read("nor58_first_original_cpu.json")
        source=z.read("nor58_first_original_60_public_only.json")
    assert len(cpu)==CPU_BYTES and sha256(cpu).hexdigest()==CPU_SHA
    assert len(source)==SOURCE_BYTES and sha256(source).hexdigest()==SOURCE_SHA
    x,fixture=json.loads(cpu),json.loads(source)
    assert canonical(x)==cpu and canonical(fixture)==source
    assert x["schema"]=="nor58.first_60_source_authenticated_static_merkle_cpu.v1"
    assert fixture["schema"]=="nor58.first_public_only_60_static_merkle_sources.v1"
    assert x["first_registration_sha"]==fixture["first_registration_sha"]==REG
    assert fixture["source_domain"]=="NOR58/SOURCE/v1"
    assert x["first_public_only_source_sha256"]==SOURCE_SHA
    assert x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert x["historical_original_first_source_ID_exclusions"]==468
    assert fixture["source_count"]==60 and fixture["excluded_original_source_IDs"]==468
    assert fixture["public_only"] and fixture["no_inverse_root_payload_signature_session_or_timing"]
    assert x["repeats"]==5 and x["signed_full_root_arms"]==list(ARMS)
    for flag in (
        "all_arms_receive_identical_signed_v2_wire_and_context",
        "all_arms_check_original_full_caps_GMP_identity_and_full_root",
        "warm_template_only_trusted_geometry_no_dynamic_hash_cache",
        "source_only_SHA_frozen_upstream_of_first_CPU",
        "reference_and_fixture_signing_outside_arm_timings",
        "template_built_outside_arm_timings_for_warm",
        "cold_template_build_charged_inside_arm",
        "root_only_diagnostics_after_signed_timed_arms",
        "nor46_global_exact_signed_convolution_unchanged"
    ):
        assert x[flag] is True,flag
    assert x["gmp_backend"]["gmpy2_version"]=="2.3.2"
    assert x["first_original_five_registered_gates"]==GATES
    assert x["first_original_engineering_pass"] is True
    ids={r["source_sha256"] for r in fixture["cases"]}
    assert len(ids)==len(fixture["cases"])==60
    assert ids.isdisjoint(cohort.historical_ids()) and len(cohort.historical_ids())==468
    cases={r["source_sha256"]:r for r in fixture["cases"]}
    rows=x["source_records"]
    assert len(rows)==60 and {r["source_sha256"] for r in rows}==ids
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert Counter(r["profile"] for r in rows)=={
        "s1-b2":12,"s4-b2":12,"s8-b2":12,"s4-b8":12,"s4-b16":12}
    assert Counter(r["family"] for r in rows)=={
        "signed_random":30,"alternating":30}
    assert Counter(r["replicate"] for r in rows)=={0:30,1:30}
    geometries=x["geometry_template_ledger"]
    assert set(geometries)=={
        f"m{m}-s{s}" for m in cohort.LENGTHS for s in (1,4,8)}
    for key,geom in geometries.items():
        m,s=geom["m"],geom["stride"]
        assert key==f"m{m}-s{s}" and m in cohort.LENGTHS and s in (1,4,8)
        assert geom["prebuilt_source_geometry_only"] is True
        assert len(geom["template_SHA256"])==64
        assert geom["canonical_template_bytes"]>0
        assert geom["build_ns_outside_all_arm_timings"]>=0
        assert geom["supported_dynamic_leaf_count"]==(m-1)//s+1
        assert geom["static_zero_leaf_count"]==m-geom["supported_dynamic_leaf_count"]
        assert geom["static_padding_leaf_count"]==(1<<(m-1).bit_length())-m
        assert geom["static_node_count"]>=(
            geom["static_zero_leaf_count"]+geom["static_padding_leaf_count"])
        assert geom["canonical_template_bytes"]<=250_000
    for r in rows:
        assert r["source_success"] and r["all_three_signed_full_exact_vectors_original_roots_equal"]
        case=cases[r["source_sha256"]]
        for key in ("replicate","length","profile","family","stride",
                    "coefficient_magnitude_bits"):
            assert r[key]==case[key]
        core={k:v for k,v in case.items() if k!="source_sha256"}
        assert sha256(canonical(core)).hexdigest()==case["source_sha256"]
        assert len(r["reference_full_original_merkle_root_sha256"])==64
        assert r["reference_full_native_generation_ns_outside_arms"]>=0
        assert r["reference_json_serialization_ns_outside_arms"]>=0
        assert r["full_reference_decimal_JSON_bytes"]>0
        assert r["signed_wire_bytes"]>r["signed_reduced_payload_bytes"]>0
        assert r["prebuilt_five_fixture_signer_encode_ns_outside_arms"]>=0
        nonces=r["five_distinct_verifier_nonce_SHA256"]
        assert len(nonces)==len(set(nonces))==5
        assert all(len(h)==64 for h in nonces)
        key=r["template_geometry_only_key"]
        g=geometries[key]
        assert key==f"m{r['length']}-s{r['stride']}"
        assert r["template_SHA256"]==g["template_SHA256"]
        assert r["template_canonical_bytes"]==g["canonical_template_bytes"]
        assert r["template_static_node_count"]==g["static_node_count"]
        assert r["template_dynamic_supported_leaf_count"]==g["supported_dynamic_leaf_count"]
        assert math.isclose(
            r["template_build_ms_outside_arms"],
            g["build_ns_outside_all_arm_timings"]/1e6,rel_tol=1e-12)
        caps=r["original_full_signed_resource_bounds"]
        assert caps["original_coefficient_count"]==r["length"]
        assert caps["compressed_coefficient_count"]==(r["length"]-1)//r["stride"]+1
        assert 0<caps["original_selected_packed_bits"]<=120_000_000
        assert 0<caps["compressed_selected_packed_bits"]<=120_000_000
        assert caps["original_witness_max_bits"]<=32768
        med,samples=r["arm_medians_ms"],r["five_interleaved_samples_ms"]
        assert set(med)==set(samples)==set(ARMS)
        for arm in ARMS:
            times=samples[arm]
            assert len(times)==5 and all(math.isfinite(t) and t>0 for t in times)
            assert math.isclose(statistics.median(times),med[arm],rel_tol=1e-12)
        ratios={
            "warm_over_original":med[ARMS[1]]/med[ARMS[0]],
            "cold_over_original":med[ARMS[2]]/med[ARMS[0]],
            "cold_over_warm":med[ARMS[2]]/med[ARMS[1]],
            "template_cost_amortized_over_five_warm_over_original":
                (5*med[ARMS[1]]+r["template_build_ms_outside_arms"])/(5*med[ARMS[0]])
        }
        for name,val in ratios.items():
            assert math.isclose(r[name],val,rel_tol=1e-12)
        assert r["untimed_root_only_original_ns"]>=0
        assert r["untimed_root_only_template_ns"]>=0
    longest=[r for r in rows if r["stride"]>1 and r["length"]==3073]
    dense=[r for r in rows if r["stride"]==1]
    assert len(longest)==16 and len(dense)==12
    totals={arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS}
    observed={
        "sparse_m3073_median_sourcewise_warm_over_original":
            statistics.median(r["warm_over_original"] for r in longest),
        "all_60_ratio_of_source_median_sums_warm_over_original":
            totals[ARMS[1]]/totals[ARMS[0]],
        "dense_s1_median_sourcewise_cold_over_original":
            statistics.median(r["cold_over_original"] for r in dense)
    }
    for k,val in PRIMARY.items():
        assert math.isclose(observed[k],val,abs_tol=1e-12)
    assert observed["sparse_m3073_median_sourcewise_warm_over_original"]<.93
    assert observed["all_60_ratio_of_source_median_sums_warm_over_original"]<.95
    assert observed["dense_s1_median_sourcewise_cold_over_original"]<1.20
    assert receipt["original_source_median_total_ms"]==totals
    assert x["overall"]["sum_source_median_ms_by_arm"]==totals
    assert x["primary_all_sixty_ratio_of_sum_of_source_medians"]==observed[
        "all_60_ratio_of_source_median_sums_warm_over_original"]
    assert receipt["first_original_five_registered_gates"]==GATES
    assert x["secondary_ten_block_descriptive_bootstrap"]["block_count"]==10
    return {"first_run":FIRST_RUN,"CPU_SHA":CPU_SHA,
            "original_verdict":"PASS (5/5)","source_count":60,
            "primary":observed,"archive_SHA":receipt["archive_zip_sha256"]}

if __name__=="__main__":
    print("NOR58 original first source/CPU exact Merkle 5-of-5 PASS audit",audit())
