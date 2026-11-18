"""NOR58 one-shot first exact CPU/source PASS archival publisher.

Copies only byte-identical original hosted artifacts from run #44316658346.
Generates a deterministic source-controlled ZIP, receipt and descriptive
report without regenerating any source, signature, inverse or timing.
"""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
SOURCE=Path("/tmp/nor58_original/nor58_first_original_60_public_only.json")
CPU=Path("/tmp/nor58_original/nor58_first_original_cpu.json")
REG="ddbabc02cf8702fe2da2c9962174ddcfcbed39a4"
SOURCE_BYTES=1473865
SOURCE_SHA="6b34e964bc170f3e8efcdfae23403fead030992cf1fc08640ed13c2e34c6b5b3"
CPU_BYTES=204565
CPU_SHA="244d84accfd2821af25a3d1d98c114964a217925fa457451d2c83c1d6793342b"
GATES={
    "sparse_m3073_warm_over_original_under_0p93":True,
    "all_60_sum_source_medians_warm_over_original_under_0p95":True,
    "twelve_dense_cold_over_original_under_1p20":True,
    "all_60_full_signed_vectors_roots_and_hostile_protocol_controls":True,
    "first_60_source_468_excluded_repeats_static_template_caps_archive":True
}

def exact(path,count,digest):
    raw=path.read_bytes()
    assert len(raw)==count and sha256(raw).hexdigest()==digest
    return raw

def main():
    outdir=HERE/"results"
    archive=outdir/"nor58_first_original_hosted.zip"
    if archive.exists():
        raise RuntimeError("first original NOR58 ZIP already preserved")
    src=exact(SOURCE,SOURCE_BYTES,SOURCE_SHA)
    cpu=exact(CPU,CPU_BYTES,CPU_SHA)
    fixture,x=json.loads(src),json.loads(cpu)
    assert x["schema"]=="nor58.first_60_source_authenticated_static_merkle_cpu.v1"
    assert fixture["schema"]=="nor58.first_public_only_60_static_merkle_sources.v1"
    assert fixture["first_registration_sha"]==x["first_registration_sha"]==REG
    assert x["first_public_only_source_sha256"]==SOURCE_SHA
    assert fixture["source_count"]==60 and fixture["excluded_original_source_IDs"]==468
    assert x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert x["historical_original_first_source_ID_exclusions"]==468
    assert len(x["source_records"])==60 and x["repeats"]==5
    assert x["first_original_five_registered_gates"]==GATES
    assert x["first_original_engineering_pass"] is True
    outdir.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=9) as z:
        for member,raw in (
            ("nor58_first_original_cpu.json",cpu),
            ("nor58_first_original_60_public_only.json",src)):
            zi=ZipInfo(member,date_time=(1980,1,1,0,0,0))
            zi.compress_type=ZIP_DEFLATED
            z.writestr(zi,raw,compress_type=ZIP_DEFLATED,compresslevel=9)
    sparse=x["primary_sparse_m3073"]
    dense=x["primary_twelve_dense_cold"]
    original_primary={
        "sparse_m3073_median_sourcewise_warm_over_original":
            sparse["median_sourcewise_warm_over_original"],
        "all_60_ratio_of_source_median_sums_warm_over_original":
            x["primary_all_sixty_ratio_of_sum_of_source_medians"],
        "dense_s1_median_sourcewise_cold_over_original":
            dense["median_sourcewise_cold_over_original"]
    }
    sums=x["overall"]["sum_source_median_ms_by_arm"]
    receipt={
        "schema":"nor58.original_first_60_static_merkle_CPU_archive.v1",
        "first_run":44316658346,
        "original_source_artifact_id":14431354312,
        "original_CPU_artifact_id":14433176882,
        "first_registration_sha":REG,
        "raw_source_bytes":SOURCE_BYTES,
        "raw_source_sha256":SOURCE_SHA,
        "raw_CPU_bytes":CPU_BYTES,
        "raw_CPU_sha256":CPU_SHA,
        "archive_zip_bytes":archive.stat().st_size,
        "archive_zip_sha256":sha256(archive.read_bytes()).hexdigest(),
        "original_first_five_registered_gates":GATES,
        "original_first_engineering_pass":True,
        "original_primary":original_primary,
        "original_source_median_total_ms":sums,
        "original_sourcewise_by_profile":x["by_profile"],
        "original_sourcewise_by_length":x["by_length"],
        "original_static_geometry_template_ledger":x["geometry_template_ledger"],
        "original_secondary_ten_block_descriptive_bootstrap":
            x["secondary_ten_block_descriptive_bootstrap"],
        "scope":"Same authenticated signed-v2 exact GMP witness and complete original Merkle root; immutable trusted geometry static leaves only; publicly known test Ed25519 private key, no real provider authority."
    }
    (outdir/"nor58_first_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    fmt=lambda v:f"{v:.6f}x"
    orig=sums["original_signed_GMP_full_Merkle"]
    warm=sums["warm_static_template_signed_GMP_full_Merkle"]
    cold=sums["cold_static_template_signed_GMP_full_Merkle"]
    lines=[
        "# NOR-58 first exact indexed static Merkle template evidence",
        "",
        "**First prospectively frozen engineering result: PASS (all 5/5 original gates).**",
        "",
        "The [first GitHub Actions run #44316658346](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44316658346) completed 60 new SHA-disjoint public-only sources, with all 468 original NOR51–57B source IDs excluded, and five interleaved exact signed full-output/full-root CPU repetitions of three matching-verifier arms per source. [Registration commit ddbabc02](https://github.com/MoralityLabAI/Compactification-Theory/commit/ddbabc02cf8702fe2da2c9962174ddcfcbed39a4) predates all NOR58 source/input and template timing.",
        "",
        "## Immutable source-first evidence",
        "",
        f"- First original CPU JSON: **{CPU_BYTES:,} exact bytes**, SHA-256 {CPU_SHA}.",
        f"- First upstream public-only 60-source JSON: **{SOURCE_BYTES:,} exact bytes**, SHA-256 {SOURCE_SHA}.",
        "- GitHub source artifact **#14431354312** was uploaded before the first dependent CPU artifact **#14433176882**.",
        "- Both raw first byte streams are retained within [nor58_first_original_hosted.zip](results/nor58_first_original_hosted.zip), SHA and first Boolean gates pinned in [receipt](results/nor58_first_receipt.json), independently checked by [original_archive_audit.py](original_archive_audit.py).",
        "",
        "## Exactly five registered first engineering gates",
        "",
        "| Registered original gate | First observation | Verdict |",
        "| --- | ---: | --- |",
        f"| 16 sparse m3073 sources, warm signed GMP verifier/full root vs original <0.93x | **{fmt(original_primary['sparse_m3073_median_sourcewise_warm_over_original'])}** | PASS |",
        f"| 60 source ratio of sums of source medians warm/original <0.95x | **{fmt(original_primary['all_60_ratio_of_source_median_sums_warm_over_original'])}** | PASS |",
        f"| 12 dense stride1 controls, cold/original sourcewise median <1.20x | **{fmt(original_primary['dense_s1_median_sourcewise_cold_over_original'])}** | PASS |",
        "| All original exact signed integer output vectors/full original NOR45 Merkle roots and malicious signing/source/witness controls | **60/60 plus source-free CI** | PASS |",
        "| SHA frozen source roster, 468 historical exclusions, five repeats, original resource caps, immutable static geometry and full evidence | **PASS** | PASS |",
        "",
        "The registered longest sparse sourcewise median warm/full signed root ratio corresponds to a **19.8% full-verifier CPU reduction**. Across all 60 first sources, the ratio of sums was **0.911459x**, an **8.9% total full-verifier CPU saving**. Every signed verifier uses identical signed compact-v2 bytes, pinned source, expected root, key, verifier nonce/sequence, original NOR47 full-sized resource bounds and NOR53 exact reduced signed GMP identity. The only timed algorithmic difference is the full original NOR45 Merkle-root construction.",
        "",
        "## Original matched workload sums",
        "",
        "| Whole authenticated exact output/full Merkle verifier | Sum of 60 source medians | Ratio to original |",
        "| --- | ---: | ---: |",
        f"| Original NOR45 full Merkle | {orig:.3f} ms | 1.000000x |",
        f"| Warm indexed-zero/pad immutable template | **{warm:.3f} ms** | **{warm/orig:.6f}x** |",
        f"| Cold fresh template constructed per call | {cold:.3f} ms | {cold/orig:.6f}x |",
        "",
        "Cold per-call template construction remains slightly slower over the complete workload, even though the registered dense cold-control overhead was only ~3.3%. Warm templates amortize geometry-only costs across repeated independently signed requests, with no reuse of dynamic hashes or completed roots. The ten-block descriptive uncertainty is in first CPU JSON; sourcewise and workload ratios are distinguished in registered scoring.",
        "",
        "## First by-profile warm/cold sourcewise medians",
        "",
        "| Fixed source profile | Sources | Warm / original | Cold / original |",
        "| --- | ---: | ---: | ---: |"
    ]
    for name in ("s1-b2","s4-b2","s8-b2","s4-b8","s4-b16"):
        item=x["by_profile"][name]
        lines.append(f"| {name} | {item['count']} | "+
            fmt(item["median_sourcewise_warm_over_original"])+" | "+
            fmt(item["median_sourcewise_cold_over_original"])+" |")
    lines+=[
        "",
        "## Immutable indexed-only templates and original root-only diagnostics",
        "",
        "Each fixed (full m, trusted support stride) template precomputes only SHA256 of canonical index-specific full-zero leaves, canonical index-specific padding leaves and internal nodes with exclusively fixed descendants. It stores no full signed witness or output-dependent Merkle root. Every supported coefficient leaf is hashed fresh on each invocation; NOR45's exact original leaf, padding, internal-node domains and full index framing are retained.",
        "",
        "On the 16 sparse m3073 first sources, the independently timed post-whole-arm root-only diagnostic gave median original full Merkle **{:.3f} ms** and median warm-template original root **{:.3f} ms**. This isolated root-only diagnostic is strictly secondary.".format(
            sparse["median_original_root_only_ns"]/1e6,
            sparse["median_template_root_only_ns"]/1e6),
        "",
        "### Template construction by original trusted geometry",
        "",
        "| Trusted geometry | One template build | Canonical static template bytes | Static node hashes |",
        "| --- | ---: | ---: | ---: |"
    ]
    for geom in ("m769-s1","m769-s4","m769-s8",
                 "m1537-s1","m1537-s4","m1537-s8",
                 "m3073-s1","m3073-s4","m3073-s8"):
        g=x["geometry_template_ledger"][geom]
        lines.append(
           f"| {geom} | {g['build_ns_outside_all_arm_timings']/1e6:.3f} ms | "
           f"{g['canonical_template_bytes']:,} | {g['static_node_count']:,} |")
    lines+=[
        "",
        "The original CPU JSON contains exact source-specific template geometry digests, signed original roots, original full-vs-compressed resource caps, all five interleaved arm samples, one-time reference witness and fixture signer costs, warm amortized five-request diagnostics, and 4,096-resample ten-block descriptive bootstrap results. The first five gate decisions remain independent of these descriptive breakdowns.",
        "",
        "## Security and formal limits",
        "",
        "This is a synthetic exact integer formal-series verifier with publicly reproducible research-fixture Ed25519 signer seed. Original full source and every supplied complete full signed witness are checked for unit, exact integer types, original full NOR47 coefficient and packed caps, and all off-support zeros **before** using a static template. Exact signed inverse identity and original full Merkle root must match the independently trusted reference to advance the replay guard. The observed speedup is a legitimate changed root-computation kernel under shared signing semantics, outside the earlier NOR57B fixed two-backend routing-only ceiling; the two studies used different source panels and their raw CPU times must not be treated as directly paired.",
        "",
        "The result proves neither genuine independent provider honesty nor secure deployed keys, off-host transport, Bitcoin consensus, zero knowledge or succinctness. m3073 remains outside the original NOR45 signed native-program grammar. Merkle-template extensional equivalence requires a separately checked Lean/MathCan theorem before being called machine-checked.",
        ""
    ]
    (HERE/"RESULTS.md").write_text("\n".join(lines))
    print("NOR58 immutable original CPU SHA",CPU_SHA)
    print("NOR58 original first source SHA",SOURCE_SHA)
    print("NOR58 original 5-of-5 engineering PASS",original_primary)
    print("NOR58 original ZIP SHA",receipt["archive_zip_sha256"])

if __name__=="__main__":
    main()
