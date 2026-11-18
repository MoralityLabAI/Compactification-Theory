"""NOR59 one-shot byte-identical first hosted original CPU/source publisher.

Never regenerates any public source, signed certificate, program output,
session, template, first timing, or first PASS/FAIL. It copies exact original
GitHub Actions first source/CPU artifact JSON bytes, checks pinned hashes,
writes deterministic source-controlled ZIP and an independently auditable
first receipt and report.
"""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
SOURCE=Path("/tmp/nor59_original/nor59_first_original_60_public_only.json")
CPU=Path("/tmp/nor59_original/nor59_first_original_cpu.json")
REG="8f6eb3dccc4d4c4d9d58955387ea6f154ad05d84"
SOURCE_BYTES=1005307
SOURCE_SHA="be516ca2149cf6d5970842891280f7f7d3407706b9c08b02c4719d153418b0f9"
CPU_BYTES=219583
CPU_SHA="c56d65e676e7fd81668d3919eb9cbf29922fbe34cfe350418e3c67ae87c62b2b"
GATES={
 "forty_sparse_authorized_warm_over_original_median_under_1p02":True,
 "forty_sparse_NOR45_root_only_warm_over_original_under_0p70":True,
 "twenty_dense_cold_whole_authorized_over_original_under_1p25":True,
 "all_60_signed_authorized_full_output_root_security_and_support":True,
 "first_60_public_program_sources_528_prior_exclusions_full_archive":True
}

def raw_checked(path,size,digest):
    content=path.read_bytes()
    assert len(content)==size and sha256(content).hexdigest()==digest
    return content

def main():
    destination=HERE/"results"
    archive=destination/"nor59_first_original_hosted.zip"
    if archive.exists():
        raise RuntimeError("immutable first original NOR59 evidence already exists")
    source=raw_checked(SOURCE,SOURCE_BYTES,SOURCE_SHA)
    cpu=raw_checked(CPU,CPU_BYTES,CPU_SHA)
    fixture,x=json.loads(source),json.loads(cpu)
    assert fixture["schema"]=="nor59.first_public_only_60_signed_authorized_program_sources.v1"
    assert x["schema"]=="nor59.first_60_full_NOR45_signed_authorized_static_root_CPU.v1"
    assert fixture["first_registration_sha"]==x["first_registration_sha"]==REG
    assert x["first_public_only_source_sha256"]==SOURCE_SHA
    assert fixture["source_count"]==x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert fixture["excluded_first_source_IDs"]==x["excluded_prior_first_source_ID_count"]==528
    assert x["repeats"]==5 and len(x["arms"])==3
    assert x["first_original_five_registered_gates"]==GATES
    assert x["first_original_engineering_pass"] is True
    destination.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=9) as z:
        for member,raw in (
            ("nor59_first_original_cpu.json",cpu),
            ("nor59_first_original_60_public_only.json",source)
        ):
            entry=ZipInfo(member,date_time=(1980,1,1,0,0,0))
            entry.compress_type=ZIP_DEFLATED
            z.writestr(entry,raw,compress_type=ZIP_DEFLATED,compresslevel=9)
    sparse=x["primary_forty_sparse_signed_authorized"]
    dense=x["primary_twenty_dense_cold"]
    overall=x["overall"]
    ms=overall["source_median_total_ms"]
    primary={
        "sparse_warm_signed_full_over_original":
            sparse["median_sourcewise_warm_over_original"],
        "sparse_root_only_warm_over_original":
            sparse["median_sourcewise_root_only_warm_over_original"],
        "dense_cold_signed_full_over_original":
            dense["median_sourcewise_cold_over_original"],
        "all_60_sum_source_median_signed_warm_over_original":
            ms["warm_authorized_static_root"]/ms["original_authorized_full_root"]
    }
    receipt={
        "schema":"nor59.first_original_60_signed_NOR45_CPU_archive.v1",
        "original_first_run":44321699757,
        "original_source_artifact_id":14435161653,
        "original_CPU_artifact_id":14434139586,
        "original_first_registration_commit":REG,
        "raw_public_source_bytes":SOURCE_BYTES,
        "raw_public_source_sha256":SOURCE_SHA,
        "raw_first_CPU_bytes":CPU_BYTES,
        "raw_first_CPU_sha256":CPU_SHA,
        "archive_zip_bytes":archive.stat().st_size,
        "archive_zip_sha256":sha256(archive.read_bytes()).hexdigest(),
        "original_first_five_gates":GATES,
        "first_original_engineering_pass":True,
        "original_primary":primary,
        "original_source_median_wall_ms":ms,
        "original_by_stride":x["by_stride"],
        "original_by_degree":x["by_degree"],
        "original_by_profile":x["by_program_profile"],
        "original_geometry_template_ledger":x["geometry_template_ledger"],
        "secondary_ten_block_descriptive_uncertainty":x["secondary_ten_block_bootstrap"],
        "trust_boundary":"Genuine NOR45 independently authorized program, NOR44 verifier precommit and provider-signed root, original full interpreter; deterministic public test-only private keys; authority honesty and external true world remain explicit premises.",
        "formal_Lean_kernel_proof_available":False
    }
    (destination/"nor59_first_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    fmt=lambda v:f"{v:.6f}x"
    original=ms["original_authorized_full_root"]
    warm=ms["warm_authorized_static_root"]
    cold=ms["cold_authorized_static_root"]
    lines=[
        "# NOR-59: first genuine NOR45 program-authorized static full Merkle roots",
        "",
        "**Original prospectively registered engineering result: PASS (5/5).**",
        "",
        "The [original first Actions run #44321699757](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44321699757) froze 60 new source/program SHA identities before any program execution/signature/root timings, then ran five source/repetition-interleaved full NOR45 signed verifier calls per mode, 60/60 exact original integer-output and full Merkle-root parity. The [preregistration commit 8f6eb3dc](https://github.com/MoralityLabAI/Compactification-Theory/commit/8f6eb3dccc4d4c4d9d58955387ea6f154ad05d84) precedes the first source job. Every arm verifies original NOR45 program-authority signature and original NOR44 verifier-precommit and signed output-provider reference before executing the same entire authentic program and checking its full original NOR45 Merkle root.",
        "",
        "## Immutable original source and CPU evidence",
        "",
        f"- First CPU JSON: **{CPU_BYTES:,} exact raw bytes**, SHA-256 `{CPU_SHA}` (GitHub artifact **14434139586**).",
        f"- Upstream public-only first source/program JSON: **{SOURCE_BYTES:,} raw bytes**, SHA-256 `{SOURCE_SHA}` (source artifact **14435161653**). Original first 60 source identities are SHA-disjoint from 528 NOR51–58 first archived source IDs.",
        "- Both original raw byte streams appear **unchanged** inside [results/nor59_first_original_hosted.zip](results/nor59_first_original_hosted.zip), whose SHA-256 and original five first Boolean gates are recorded in [results/nor59_first_receipt.json](results/nor59_first_receipt.json) and checked by [read-only independent original auditor](original_archive_audit.py).",
        "- Original runner: Ubuntu24.04, CPython3.12, cryptography46.0.4 and gmpy2 2.3.2. All signing keys and deterministic session nonces are **public research-test fixtures**, not deployed credentials.",
        "",
        "## Exactly five preregistered first engineering gates",
        "",
        "| Frozen original gate | First registered observation | Verdict |",
        "| --- | ---: | --- |",
        f"| All 40 sparse s4/s8 actual authorized programs, warm/original full signed verifier median <1.02 | **{fmt(primary['sparse_warm_signed_full_over_original'])}** | PASS |",
        f"| Same 40 sparse, post-timing original full Merkle root-only warm/original median <0.70 | **{fmt(primary['sparse_root_only_warm_over_original'])}** | PASS |",
        f"| All 20 dense s1 original authorized programs, cold/original whole signed verifier median <1.25 | **{fmt(primary['dense_cold_signed_full_over_original'])}** | PASS |",
        "| Full exact output vectors, original roots and hostile program-authority/provider/query/source/session/nonce/signature controls | **60/60 plus predata** | PASS |",
        "| First SHA before timing, 528 old source exclusions, five samples, original signed NOR45 semantics, geometry-only immutable cache and archival integrity | **PASS** | PASS |",
        "",
        "The authentic signed NOR45 whole-program verifier's median sourcewise warm/original ratio was **0.959140x** across 40 sparse cases, about **4.1% faster** including full original program-authority, provider and verifier signatures, complete program execution and original Merkle root. The separate root-only ratio was **0.309528x**, or about **69.0% less full-root computation time**. These are different registered endpoints and are not interchangeable.",
        "",
        "## Complete matched 60-source workload totals",
        "",
        "| Authentic NOR45 signed full-program/full-root arm | Sum of source-median wall time | Ratio to original |",
        "| --- | ---: | ---: |",
        f"| Original independent full Merkle root | **{original:.3f} ms** | 1.000000x |",
        f"| Source-certified warm static root | **{warm:.3f} ms** | **{fmt(warm/original)}** |",
        f"| Source-certified cold rebuilt static root | **{cold:.3f} ms** | {fmt(cold/original)} |",
        "",
        "Across all 60 cases, dense signed program recomputation dominated the runtime; the sum-of-source-median warm/original was **0.991555x**, a **0.84%** complete authorized-workload reduction. Registered overall workload ratio is descriptive for NOR59; the first sparse whole-operation performance gate uses median sourcewise ratios on the fixed 40 sparse cases. The ten-block bootstrap intervals and all raw five samples remain in original first CPU JSON.",
        "",
        "## Original source support stratification",
        "",
        "| Signed input support | Source/program cases | Warm/original whole signed verifier | Warm/original full-root-only |",
        "| --- | ---: | ---: | ---: |"
    ]
    for stride in (1,4,8):
        group=x["by_stride"][str(stride)]
        lines.append(
            f"| s={stride} | {group['count']} | "+
            fmt(group["median_sourcewise_warm_over_original"])+
            " | "+fmt(group["median_sourcewise_root_only_warm_over_original"])+" |")
    lines+=["",
        "## Signed program grammar and cache trust boundary",
        "",
        "Original signed NOR45 permits only D512/1024, m769/1537, I/N/R/S unary operations, delays 64/80/96/112, spacings 24/32 and optional D+32 suffix. All original shifts are multiples of eight, so on a source independently verified to have support stride4 or stride8, every allowed operation and original signed program output preserves that stride support. The source-closure checker independently checks the **entire query** and the **entire recomputed output** for off-support zeros before a template can calculate a root. The full original NOR45 root is still compared to the separately signed program-authority certificate and original provider reference.",
        "",
        "The warm cache contains only index-bound zero/padding leaf hashes and static-only subtrees for verified (m,stride), with no prior dynamic leaf hash, witness value, accepted root or provider/session data. The original NOR45 verifier's default path and signature checks remain unchanged; only its final exact root calculation changes under a fixed internal root policy. Root-only diagnostics were taken **after** whole-verifier timing, with one-time warm template preparation outside timed calls and cold preparation inside each timed call. All reference output, signing setup, template digests, session context/nonce hashes and raw input programs are preserved in the original CPU JSON.",
        "",
        "## Limits and formal follow-up",
        "",
        "This is genuine faithful NOR45 **test-fixture program authorization** with separately signed program authority, verifier-precommit and provider reference. The keys and nonce fixtures are publicly reproducible, and the verifier's original per-call cryptographic binding is stateless without a deployment nonce-use ledger. A malicious but correctly pinned authority can still sign the wrong program and remain crypto-valid; true-world semantic honesty is external. There is no zero knowledge, succinctness, off-host provider-security claim or expansion of NOR45 grammar to m3073.",
        "",
        "The Python support-closure check and exhaustive source-free tests are not a Lean kernel proof. A separately verified Lean/MathCan theorem is still needed for supported truncated series inversion, NOR45 program shift/operation closure and indexed subtree root extensional equality.",
        ""
    ]
    (HERE/"RESULTS.md").write_text("\n".join(lines))
    print("NOR59 first immutable CPU SHA",CPU_SHA)
    print("NOR59 first immutable source SHA",SOURCE_SHA)
    print("NOR59 first 5/5 signed-authority PASS",primary)
    print("NOR59 original ZIP SHA",receipt["archive_zip_sha256"])

if __name__=="__main__":
    main()
