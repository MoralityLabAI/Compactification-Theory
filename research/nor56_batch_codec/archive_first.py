"""One-shot NOR56 first evidence publisher: copy ORIGINAL GitHub Actions bytes.

The file must be invoked only by the temporary write-enabled archive workflow,
after downloading the exact original first run #44296895730 artifact. It never
regenerates source inputs, invokes any model, or reruns a timed benchmark.
"""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
SOURCE="/tmp/nor56_first/nor56_first_original_60_public_only.json"
CPU="/tmp/nor56_first/nor56_first_original_cpu.json"
REG="b186ee7478f27230478b099d7091b25817839464"
CPU_BYTES=400679
CPU_SHA="2ad2c0e149ca9f2405f7883fbb768fc72f380a03d717c2b265883343a6d7429d"
SOURCE_BYTES=1473496
SOURCE_SHA="f1326f4ac9cac314564c96eb82008177f30280b38ab8972eab696f51996b421c"
GATES={
 "sparse_m3073_signed_v2_over_v1_under_0p92":False,
 "sparse_m3073_signed_v2_over_JSON_under_0p82":False,
 "all_48_sparse_signed_v2_wire_over_v1_wire_under_1p05":True,
 "all_60_exact_full_roots_and_signed_canonical_attack_controls":True,
 "first_source_60_288_excluded_five_repeats_caps_reentrant_archive":True
}
FORMAT_ARMS=(
    "signed_compact_v1_verifier_root",
    "signed_json_verifier_root",
    "signed_compact_v2_verifier_root"
)
PHASES=(
    "header_and_replay_ns","ed25519_signature_ns",
    "canonical_payload_decode_ns","full_source_witness_support_caps_ns",
    "signed_exact_GMP_convolution_ns","full_original_Merkle_root_ns"
)
def assert_raw(path,bytes_len,digest):
    raw=Path(path).read_bytes()
    assert len(raw)==bytes_len and sha256(raw).hexdigest()==digest
    return raw
def main():
    archive_dir=ROOT/"results"
    zip_path=archive_dir/"nor56_first_original_hosted.zip"
    if zip_path.exists():
        raise RuntimeError("one-shot NOR56 original source archive already exists")
    cpu_raw=assert_raw(CPU,CPU_BYTES,CPU_SHA)
    source_raw=assert_raw(SOURCE,SOURCE_BYTES,SOURCE_SHA)
    data,fixture=json.loads(cpu_raw),json.loads(source_raw)
    assert data["schema"]=="nor56.first_60_source_signed_compact_v2_cpu.v1"
    assert fixture["schema"]=="nor56.first_public_only_60_batch_table_sources.v1"
    assert data["registration_first_sha"]==fixture["first_registration_sha"]==REG
    assert data["first_source_public_only_SHA256"]==SOURCE_SHA
    assert data["source_count"]==data["source_success_count"]==60
    assert data["source_failure_count"]==0
    assert data["excluded_previous_first_source_ID_count"]==288
    assert fixture["source_count"]==60 and fixture["excluded_original_source_IDs"]==288
    assert data["original_five_registered_gates"]==GATES
    assert data["first_original_engineering_pass"] is False
    assert len(data["source_records"])==60
    archive_dir.mkdir(parents=True,exist_ok=True)
    with ZipFile(zip_path,"w",compression=ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in (
            ("nor56_first_original_cpu.json",cpu_raw),
            ("nor56_first_original_60_public_only.json",source_raw)):
            info=ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=ZIP_DEFLATED
            z.writestr(info,raw,compress_type=ZIP_DEFLATED,compresslevel=9)
    sparse_long=data["primary_sixteen_sparse_m3073"]
    sparse=data["primary_48_sparse_wire"]
    primary={
        "sparse_m3073_v2_over_v1":sparse_long["median_signed_v2_over_signed_v1"],
        "sparse_m3073_v2_over_JSON":sparse_long["median_signed_v2_over_signed_json"],
        "sparse_48_v2_complete_wire_over_v1":
            sparse["median_signed_v2_complete_wire_over_signed_v1_complete_wire"]
    }
    # Exact original diagnostic from the first raw CPU JSON, never rerun.
    phase_totals={arm:{key:sum(
        r["secondary_untimed_six_phase_ledger_ns"][arm][key]
        for r in data["source_records"]) for key in PHASES}
        for arm in FORMAT_ARMS}
    phase_shares={}
    for arm,part in phase_totals.items():
        total=sum(part.values())
        phase_shares[arm]={key:part[key]/total for key in PHASES}
    receipt={
        "schema":"nor56.original_first_signed_v2_source_cpu_archive.v1",
        "first_run":44296895730,
        "original_source_artifact_id":14423891566,
        "original_cpu_artifact_id":14424252022,
        "first_registration_sha":REG,
        "raw_cpu_bytes":CPU_BYTES,
        "raw_cpu_sha256":CPU_SHA,
        "raw_source_bytes":SOURCE_BYTES,
        "raw_source_sha256":SOURCE_SHA,
        "original_archive_zip_bytes":zip_path.stat().st_size,
        "original_archive_zip_sha256":sha256(zip_path.read_bytes()).hexdigest(),
        "original_five_registered_gates":GATES,
        "first_original_engineering_pass":False,
        "original_primary":primary,
        "original_by_profile":data["by_profile"],
        "original_by_length":data["by_length"],
        "original_secondary_ten_block":data["secondary_source_median_ten_block_bootstrap"],
        "untimed_phase_total_ns_by_signed_arm":phase_totals,
        "untimed_phase_shares_by_signed_arm":phase_shares,
        "scientific_scope":"Transparent exact integer synthetic inverse and independent original full root. Public research signing key, no actual trusted external provider, source m3073 outside NOR45 signed grammar, no off-host network."
    }
    (archive_dir/"nor56_first_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    fmt=lambda x:f"{x:.6f}×"
    lines=[
        "# NOR-56: first prospectively source-locked signed compact-v2 batch decoder",
        "",
        "**Original engineering result: FAIL, 3/5 registered gates passed.**",
        "",
        "The [first original run #44296895730](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44296895730) completed 60 new SHA-frozen public-only source identities, with 288 first NOR51–55 source IDs excluded. Five source-seeded interleaved repetitions of four complete integer-output/full-NOR45-Merkle-root arms agree exactly. [Registration commit b186ee74](https://github.com/MoralityLabAI/Compactification-Theory/commit/b186ee7478f27230478b099d7091b25817839464) predates every first NOR56 source and benchmark timing.",
        "",
        "## Immutable original evidence",
        "",
        f"- First original CPU JSON: **{CPU_BYTES:,} bytes**, SHA-256 `{CPU_SHA}`.",
        f"- First public-only source JSON: **{SOURCE_BYTES:,} bytes**, SHA-256 `{SOURCE_SHA}`. Uploaded in a separate upstream CI job before any inverse or first timing.",
        "- First source artifact **14423891566** and first CPU artifact **14424252022**.",
        "- Both exact original JSON byte streams are preserved in [nor56_first_original_hosted.zip](results/nor56_first_original_hosted.zip), with its complete ZIP SHA, original gate verdict, source medians and independent phase ledger in [nor56_first_receipt.json](results/nor56_first_receipt.json). [Permanent read-only auditor](original_archive_audit.py) recomputes raw file SHAs, all gate ratios and full-root/sample evidence without source regeneration.",
        "- Runner Ubuntu 24.04, Python 3.12, gmpy2 2.3.2, cryptography 46.0.4. Full integer vectors, original signed caps, pinned test-fixture Ed25519 signature, canonical source/session/nonce and original full roots verified.",
        "",
        "## Five preregistered first engineering gates",
        "",
        "| Fixed gate | First observed result | Original verdict |",
        "| --- | ---: | --- |",
        f"| All 16 sparse m3073: signed v2/v1 verification time <0.92 | **{fmt(primary['sparse_m3073_v2_over_v1'])}** | **FAIL** |",
        f"| Same 16: signed v2/full JSON time <0.82 | **{fmt(primary['sparse_m3073_v2_over_JSON'])}** | **FAIL** |",
        f"| All 48 sparse: complete signed v2/v1 envelope bytes <1.05 | **{fmt(primary['sparse_48_v2_complete_wire_over_v1'])}** | PASS |",
        "| All full signed exact integer vectors/original Merkle roots, signed encoding, source, key and replay controls | **60/60 and source-free tests** | PASS |",
        "| First source-only SHA, 288 historical exclusions, five repeats, original caps and reentrancy | **PASS** | PASS |",
        "",
        "The sourcewise paired medians indicate v2's authenticated whole-operation verifier is about **2.18% faster** than v1 at m3073 in the 16 sparse cases. Signed v2/full JSON is **0.835501×**, which misses the original registered <0.82 gate. Complete signed v2 bytes are effectively equal to v1 on the 48 sparse cases, at **0.999223×**; both include the signature/header. These are finite original synthetic-source CPU measurements rather than an asymptotic theorem.",
        "",
        "## Fixed by-profile sourcewise median ratios",
        "",
        "| Source profile | Units | Signed v2/v1 | Signed v2/JSON | Signed v2/v1 wire bytes |",
        "| --- | ---: | ---: | ---: | ---: |"
    ]
    for profile in ("s1-b2","s4-b2","s8-b2","s4-b8","s4-b16"):
        q=data["by_profile"][profile]
        lines.append(
            f"| {profile} | {q['count']} | "
            f"{fmt(q['median_signed_v2_over_signed_v1'])} | "
            f"{fmt(q['median_signed_v2_over_signed_json'])} | "
            f"{fmt(q['median_signed_v2_complete_wire_over_signed_v1_complete_wire'])} |")
    totals=data["secondary_source_median_ten_block_bootstrap"]["source_median_total_ms"]
    baseline=totals["signed_compact_v1_verifier_root"]
    lines+=["",
            "## Secondary matched 60-source workload-total CPU",
            "",
            "| Exact full-output/root arm | Sum of source medians | Ratio to signed v1 |",
            "| --- | ---: | ---: |"]
    for arm,label in (
        ("native_compressed_R_root","NOR52 direct compressed native inversion"),
        ("signed_compact_v1_verifier_root","NOR55 authenticated compact-v1"),
        ("signed_json_verifier_root","NOR55 authenticated full JSON"),
        ("signed_compact_v2_verifier_root","NOR56 authenticated compact-v2")):
        lines.append(f"| {label} | {totals[arm]:.3f} ms | {fmt(totals[arm]/baseline)} |")
    lines+=["",
        "The totals above are predeclared **secondary sums of 60 source-median timings**. The first registered engineering score uses sourcewise ratios on the sixteen longest sparse inputs and the 48-source wire cohort. Ten matched (replicate,profile) block-bootstrap descriptive intervals remain in first CPU JSON; secondary totals cannot change original 3/5 FAIL.",
        "",
        "## Separately measured untimed phase replay",
        "",
        "After each source's first whole-arm timings, an independent signed verifier replay checked original root/vector parity while measuring canonical header/replay, Ed25519, payload decode, original support/size caps, exact signed GMP identity and complete full Merkle reconstruction. These are diagnostic original CPU phase observations and do not prove one phase caused the observed whole-arm difference.",
        "",
        "| Authenticated format | Canonical decode phase | Exact GMP phase | Full Merkle phase |",
        "| --- | ---: | ---: | ---: |"]
    for arm,name in (
        ("signed_compact_v1_verifier_root","Compact-v1"),
        ("signed_json_verifier_root","Full decimal JSON"),
        ("signed_compact_v2_verifier_root","Batch-table v2")):
        p=phase_shares[arm]
        lines.append(
            f"| {name} | {p['canonical_payload_decode_ns']*100:.2f}% | "
            f"{p['signed_exact_GMP_convolution_ns']*100:.2f}% | "
            f"{p['full_original_Merkle_root_ns']*100:.2f}% |")
    reduction=(1-phase_totals["signed_compact_v2_verifier_root"][
        "canonical_payload_decode_ns"]/
        phase_totals["signed_compact_v1_verifier_root"][
        "canonical_payload_decode_ns"])*100
    lines+=["",
        f"In this **untimed phase replay**, total decoder phase time for the v2 panel was about **{reduction:.1f}% lower** than v1's. Exact GMP and full-root reconstruction account for most measured remaining work, limiting whole-operation gains. The phase attribution is a descriptive replay under the same source mix, not a registered causal phase intervention.",
        "",
        "## Boundary and next work",
        "",
        "The complete source-bound v2 Ed25519 signed protocol uses a **public-by-construction research-test private key**, which is unsuitable for deployment. The test proves conditional source/session and message integrity given an independently pinned public key, plus an independent exact inverse identity and full Merkle reconstruction. It does not establish an honest remote provider, signed NOR45 native grammar at m3073, cross-host transport, succinct proof size or zero knowledge.",
        "",
        "The next experiment should target the **GMP identity/root work** rather than further table parsing, with a clean same-run source-locked arithmetic change. A Lean kernel formalization of the sparse-support substitution and no-carry signed correction remains separate from this runtime result.",
        ""
    ]
    (ROOT/"RESULTS.md").write_text("\n".join(lines))
    print("NOR56 original immutable CPU SHA",CPU_SHA)
    print("NOR56 first immutable source SHA",SOURCE_SHA)
    print("NOR56 original 3/5 engineering FAIL",primary)
    print("NOR56 archive ZIP SHA",receipt["original_archive_zip_sha256"])

if __name__=="__main__":
    main()
