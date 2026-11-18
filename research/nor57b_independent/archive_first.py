"""NOR57B one-shot original FIRST source/CPU evidence publisher.

Copies byte-identical GitHub Actions first CPU and source JSON from run
44305694146. The original NOR57 reporting FAIL remains separate and unscored.
No new source generation, inverse, signature or timing is performed here.
"""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
REG="0a9467ae18df3296d8fdf04971f0ffbbb4c6e3d7"
SOURCE_PATH=Path("/tmp/nor57b_first/nor57b_first_original_60_public_only.json")
CPU_PATH=Path("/tmp/nor57b_first/nor57b_first_original_cpu.json")
SOURCE_BYTES=1474323
SOURCE_SHA="bca8cddca58a4c8a234df4f6bba2b31c44c7e114c2f07f67a545638dad5b9a63"
CPU_BYTES=173922
CPU_SHA="83fbda5e6b94c3e434203def01f3a2092235a73dc3e235f7d2187450dc421890"
EXPECTED_GATES={
 "all_60_selector_sum_source_median_over_GMP_under_0p95":False,
 "eight_native_selected_native_over_GMP_median_under_0p85":True,
 "fifty_two_GMP_selected_selector_over_GMP_median_under_1p10":True,
 "all_60_signed_exact_full_vectors_original_roots_and_tamper_controls":True,
 "first_source_60_348_old_excluded_5_repeats_fixed_route_caps_archive":True
}

def verify_exact_file(path,size,digest):
    raw=path.read_bytes()
    assert len(raw)==size and sha256(raw).hexdigest()==digest
    return raw

def main():
    outdir=ROOT/"results"
    archive=outdir/"nor57b_first_original_hosted.zip"
    if archive.exists():raise RuntimeError("original NOR57B first immutable ZIP already exists")
    src=verify_exact_file(SOURCE_PATH,SOURCE_BYTES,SOURCE_SHA)
    cpu=verify_exact_file(CPU_PATH,CPU_BYTES,CPU_SHA)
    fixture,x=json.loads(src),json.loads(cpu)
    assert fixture["first_registration_sha"]==x["first_registration_sha"]==REG
    assert fixture["source_domain"]=="NOR57B/SOURCE/v1"
    assert fixture["source_count"]==x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert fixture["excluded_original_source_IDs"]==x["previous_original_source_exclusions"]==408
    assert x["selector_native_count"]==8 and x["selector_gmp_count"]==52
    assert len(x["source_records"])==60 and x["repeat_count"]==5
    assert x["original_five_registered_gates"]==EXPECTED_GATES
    assert x["first_original_engineering_pass"] is False
    assert x["source_first_SHA_frozen_before_witness_and_timing"]
    outdir.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in (
            ("nor57b_first_original_cpu.json",cpu),
            ("nor57b_first_original_60_public_only.json",src)):
            info=ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=ZIP_DEFLATED
            z.writestr(info,raw,compress_type=ZIP_DEFLATED,compresslevel=9)
    native=x["primary_native_selected"]
    other=x["primary_GMP_selected"]
    primary={
       "whole_source_median_selector_over_GMP":
          x["primary_selector_ratio_of_sum_of_sixty_source_medians"],
       "eight_native_selected_native_over_GMP":
          native["median_sourcewise_always_native_over_always_GMP"],
       "fifty_two_GMP_selected_selector_over_GMP":
          other["median_sourcewise_frozen_selector_over_always_GMP"]
    }
    totals=x["overall"]["sum_source_median_ms_by_arm"]
    receipt={
        "schema":"nor57b.first_original_signed_source_selector_CPU_archive.v1",
        "first_run":44305694146,
        "original_source_artifact_id":14428291671,
        "original_CPU_artifact_id":14427713264,
        "registration_sha_before_new_source":REG,
        "raw_source_bytes":SOURCE_BYTES,
        "raw_source_sha256":SOURCE_SHA,
        "raw_CPU_bytes":CPU_BYTES,
        "raw_CPU_sha256":CPU_SHA,
        "archive_bytes":archive.stat().st_size,
        "archive_sha256":sha256(archive.read_bytes()).hexdigest(),
        "original_five_registered_gates":EXPECTED_GATES,
        "first_original_engineering_pass":False,
        "original_primary":primary,
        "original_workload_ms":totals,
        "original_profile_stats":x["by_profile"],
        "original_secondary_ten_block":x["secondary_ten_block_workload"],
        "prior_original_NOR57_unscored_FAIL_archive_SHA256":
            "d6feae134d340a4f15535b4bcc14e9800da3d559213f51203895c1c0c70b72c5",
        "limitations":"Synthetic exact integer transparent signed test-fixture witness; original NOR57 first run unscored reporting FAIL; m3073 outside signed NOR45 grammar; no off-host honest provider."
    }
    (outdir/"nor57b_first_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    fmt=lambda y:f"{y:.6f}×"
    lines=[
      "# NOR-57B: first prospective signed native/GMP public-source selector follow-up",
      "",
      "**Original NOR57B engineering result: FAIL, 4/5 first registered gates passed.**",
      "",
      "This is a **new SHA-disjoint first NOR57B experiment** after NOR57's original first CPU reporter crashed before creating a CPU JSON or scoring any of its five gates. The [original NOR57 negative first job/source evidence](../nor57_authenticated_router/RESULTS.md) remains **unscored** and independently preserved. The existing signed exact verifier, selector and five quantitative thresholds were fixed again in [NOR57B registration commit 0a9467ae](https://github.com/MoralityLabAI/Compactification-Theory/commit/0a9467ae18df3296d8fdf04971f0ffbbb4c6e3d7) before any NOR57B source generation/timing.",
      "",
      "The [first NOR57B hosted run #44305694146](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44305694146) completed all 60 new public-only source IDs, excluding all 408 original NOR51–57 first IDs. The separate source-only job uploaded the first SHA before the dependent CPU job. All three signed compact-v2 verifier arms accepted the **same signed envelope bytes** and returned identical complete exact signed integer vectors and original full NOR45-format Merkle roots with source/nonce/key/session and original resource controls.",
      "",
      "## Immutable first NOR57B source/CPU evidence",
      "",
      f"- Original first CPU JSON **{CPU_BYTES:,} bytes**, SHA-256 `{CPU_SHA}`.",
      f"- Original public-only 60-source JSON **{SOURCE_BYTES:,} bytes**, SHA-256 `{SOURCE_SHA}`.",
      "- First source artifact **14428291671**, first original CPU artifact **14427713264**, same [first GitHub Actions run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44305694146).",
      "- Both raw byte streams preserved unmodified in [nor57b_first_original_hosted.zip](results/nor57b_first_original_hosted.zip), with original archive SHA and five first registered Boolean results in [receipt](results/nor57b_first_receipt.json) and permanent [read-only source/SHA/ratio audit](original_archive_audit.py).",
      "",
      "## Five original NOR57B registered engineering gates",
      "",
      "| Fixed criterion | Original first result | Verdict |",
      "| --- | ---: | --- |",
      f"| 60-source **ratio of sums** selector/always-GMP <0.95 | **{fmt(primary['whole_source_median_selector_over_GMP'])}** | **FAIL** |",
      f"| Eight native-selected **median sourcewise** native/GMP <0.85 | **{fmt(primary['eight_native_selected_native_over_GMP'])}** | PASS |",
      f"| Fifty-two GMP-selected **median sourcewise** selector/GMP <1.10 | **{fmt(primary['fifty_two_GMP_selected_selector_over_GMP'])}** | PASS |",
      "| All 60 full exact integer vectors/full original Merkle roots and source/security tamper controls | **60/60 plus predata** | PASS |",
      "| First SHA, 408 historical ID exclusions, five repeats, fixed 8/52 public rule, original caps and reentrancy | **PASS** | PASS |",
      "",
      "The frozen high-width/short-source selector successfully identified eight cases with **0.738331×** median sourcewise native/GMP whole-verifier time, a 26.2% reduction. The other 52 GMP-routed sources had **1.021210×** median selector/GMP overhead. The registered **whole-workload** ratio of sums was **0.989961×**, about a 1.0% improvement, missing the strict 0.95× (5% improvement) goal. This FAIL remains original.",
      "",
      "## How the complete workload gain was diluted",
      "",
      "| Signed whole-output/full-root source group | Always GMP sum of source-median ms | Fixed selector sum of source-median ms |",
      "| --- | ---: | ---: |",
      f"| Exactly 8 native-selected inputs | {native['sum_source_median_ms_by_arm']['always_fast_GMP_signed_v2']:.3f} | {native['sum_source_median_ms_by_arm']['frozen_public_selector_signed_v2']:.3f} |",
      f"| Exactly 52 GMP-selected inputs | {other['sum_source_median_ms_by_arm']['always_fast_GMP_signed_v2']:.3f} | {other['sum_source_median_ms_by_arm']['frozen_public_selector_signed_v2']:.3f} |",
      f"| **All 60 inputs** | **{totals['always_fast_GMP_signed_v2']:.3f}** | **{totals['frozen_public_selector_signed_v2']:.3f}** |",
      "",
      "Native-selected cases saved about **38.9 ms**, while remaining GMP-routed cases incurred about **14.5 ms** in aggregate. The net **24.4 ms** saved across the 60-source matched workload explains the 0.989961× total ratio. The *always-native* whole-panel sum was much larger, making unrestricted native recomputation expensive on this source distribution. All totals are sums of original per-source five-repeat medians. Ten matched replicate/profile-block descriptive bootstrap intervals remain in the first raw CPU JSON.",
      "",
      "## Original fixed source profiles",
      "",
      "| Public source profile | Units | Native/GMP median sourcewise | Fixed selector/GMP median sourcewise |",
      "| --- | ---: | ---: | ---: |"]
    for p in ("s1-b2","s4-b2","s8-b2","s4-b8","s4-b16"):
        y=x["by_profile"][p]
        lines.append(
            f"| {p} | {y['count']} | "
            f"{fmt(y['median_sourcewise_always_native_over_always_GMP'])} | "
            f"{fmt(y['median_sourcewise_frozen_selector_over_always_GMP'])} |")
    lines+=["",
       "## Exact signed acceptance and research scope",
       "",
       "Every arm parses and authenticates an **identical complete NOR56 compact-v2 signed envelope**, verifying the pinned research key fingerprint, trusted full source/source SHA, original expected root, exact unit/support/coefficient/packed caps, monotone verifier-issued nonce and session, and a complete original full NOR45 Merkle root before accepting. The native arm independently recomputes a reduced exact reciprocal from trusted source and compares every reduced/full supplied signed coefficient. The GMP arm checks exact signed no-carry polynomial identity. The frozen selector reads only authenticated full public length, stride and actual coefficient bit length, using the **same rule as original NOR57**. First source-level five-repeat timing samples, roots, source hashes and signer generation/serialization costs are all preserved in the raw first JSON.",
       "",
       "This tests a transparent nonsuccinct exact integer algebra witness with a **public deterministic research Ed25519 fixture seed**, not real secure source or remote publisher trust. m3073 remains outside original NOR45 signed-native program grammar. Deployed source/session authority, actual off-host network costs, and true-world provider honesty remain external. A follow-up must be independently preregistered if experimenting with selector routing to increase workload-wide savings; it cannot retroactively change this NOR57B first 4/5 FAIL.",
       ""]
    (ROOT/"RESULTS.md").write_text("\n".join(lines))
    print("NOR57B first original CPU raw SHA",CPU_SHA)
    print("NOR57B first original source raw SHA",SOURCE_SHA)
    print("NOR57B original 4-of-5 engineering FAIL",primary)
    print("NOR57B original source-controlled ZIP SHA",receipt["archive_sha256"])

if __name__=="__main__":
    main()
