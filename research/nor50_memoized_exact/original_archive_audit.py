"""NOR50 original first CPU audit: byte-locked FAIL 4/5; no benchmark replay."""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json,math,statistics

HERE=Path(__file__).resolve().parent/"results"
RAW_SHA="69344bc6be09246296a5d6c0ac2e22b58031da2ed303f67be69d49bd78cb7f2e"
SOURCE_SHA="5a64a6d0ff9d9a4d74535443bc02b1d069cb6bc76b441da35acdb016dd1d9a93"
REGISTRATION="a3b7a211e312235f8e9363612aff4173f50a3a4e"
RUN=44265466762
ARTIFACT=14409649467
ARMS=("nor45_original_direct","nor50_memoized_direct",
      "nor47_gmp_full_trace","nor49_count_R_dispatch")
GATES={
 "multi_R_cached_over_original_under_0p70_at_least_8":True,
 "multi_R_gmp_over_cached_under_0p90_at_least_8":False,
 "low_R_cached_over_original_under_1p10_at_least_8":True,
 "all_full_exact_outputs_roots_and_training_controls":True,
 "source_first_64_reentrant_five_interleaved_repeats":True
}

def audit():
    archive=HERE/"nor50_first_original_hosted.zip"
    zipraw=archive.read_bytes()
    receipt=json.loads((HERE/"nor50_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor50.first_original_cpu_archive.v1"
    assert receipt["first_run"]==RUN and receipt["first_artifact_id"]==ARTIFACT
    assert receipt["first_registration_sha"]==REGISTRATION
    assert receipt["archive_zip_sha256"]==sha256(zipraw).hexdigest()
    assert receipt["raw_cpu_sha256"]==RAW_SHA and receipt["raw_cpu_bytes"]==118129
    assert receipt["raw_source_sha256"]==SOURCE_SHA and receipt["raw_source_bytes"]==9428144
    assert receipt["original_gates"]==GATES and receipt["original_engineering_pass"] is False
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={"nor50_first_original_cpu.json",
                                  "nor50_first_public_only_fixture.json"}
        raw=z.read("nor50_first_original_cpu.json")
        public=z.read("nor50_first_public_only_fixture.json")
    assert len(raw)==118129 and sha256(raw).hexdigest()==RAW_SHA
    assert len(public)==9428144 and sha256(public).hexdigest()==SOURCE_SHA
    x,src=json.loads(raw),json.loads(public)
    assert x["schema"]=="nor50.first_64_new_source_memoization_cpu.v1"
    assert src["schema"]=="nor50.first_64_public_only_memoization_sources.v1"
    assert x["registration_first_commit"]==src["registration_first_commit"]==REGISTRATION
    assert x["source_only_first_sha256"]==SOURCE_SHA
    assert x["source_count"]==64 and src["public_source_units"]==64
    assert x["historical_excluded_source_count"]==src["excluded_original_source_count"]==452
    assert x["repeat_count"]==5 and x["first_five_interleaved_arms"]==list(ARMS)
    assert x["no_global_NOR46_convolution_override"] is True
    assert x["source_only_precedes_first_timing"] is True
    assert x["transcript_production_measured_separately"] is True
    assert x["five_first_registered_gates"]==GATES
    assert x["first_engineering_pass"] is False
    rows=x["source_records"]
    public_ids={r["full_public_input_sha256"] for r in src["cases"]}
    ids=[r["source_sha256"] for r in rows]
    assert len(rows)==len(public_ids)==len(set(ids))==64
    assert set(ids)==public_ids
    assert Counter(r["replicate"] for r in rows)=={i:4 for i in range(200,216)}
    assert Counter(r["authorized_active_R_count"] for r in rows)=={0:17,1:28,2:16,3:3}
    assert all(not set(r).intersection(("hidden_beta","hidden_transform","trace",
       "program","future","target","timings")) for r in src["cases"])
    for r in rows:
        assert r["all_exact_outputs_and_merkle_roots_identical"]
        n=r["authorized_active_R_count"]
        ops=r["authorized_signed_program_ops"]
        computed=sum(ops[k]=="R" for k in ("base","u","v"))
        if ops["hidden_beta_nonzero"]:
            computed+=(ops["hidden_transform"]=="R")
        assert computed==n
        for a in ARMS:
            timings=r["five_interleaved_samples_ms"][a]
            assert len(timings)==5
            assert all(math.isfinite(t) and t>0 for t in timings)
            assert math.isclose(statistics.median(timings),r["arm_medians_ms"][a],rel_tol=1e-12)
        m=r["arm_medians_ms"]
        assert math.isclose(r["cached_over_original"],m[ARMS[1]]/m[ARMS[0]],rel_tol=1e-12)
        assert math.isclose(r["gmp_over_cached"],m[ARMS[2]]/m[ARMS[1]],rel_tol=1e-12)
    multi=[r for r in rows if r["authorized_active_R_count"]>=2]
    low=[r for r in rows if r["authorized_active_R_count"]<=1]
    assert len(multi)==19 and len(low)==45
    med_cached=statistics.median(r["cached_over_original"] for r in multi)
    med_gmp=statistics.median(r["gmp_over_cached"] for r in multi)
    med_low=statistics.median(r["cached_over_original"] for r in low)
    assert math.isclose(med_cached,0.5116361341527716,abs_tol=1e-12)
    assert math.isclose(med_gmp,1.183263144025194,abs_tol=1e-12)
    assert math.isclose(med_low,1.0020723074206739,abs_tol=1e-12)
    assert med_cached<.70 and med_gmp>=.90 and med_low<1.10
    totals={arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS}
    for a in ARMS:
        assert math.isclose(totals[a],x["secondary_workload_totals_and_bootstrap"][
            "sum_source_medians_ms"][a],rel_tol=1e-12)
    assert math.isclose(totals[ARMS[1]]/totals[ARMS[0]],0.7333710138719598,abs_tol=1e-12)
    assert math.isclose(totals[ARMS[3]]/totals[ARMS[0]],0.8853976560435939,abs_tol=1e-12)
    return {"first_SHA":RAW_SHA,"first_verdict":"FAIL 4/5",
            "multi_R_n":19,"cached_original":med_cached,"gmp_cached":med_gmp,
            "exploratory_workload_cached_original":totals[ARMS[1]]/totals[ARMS[0]]}

if __name__=="__main__":
    print("NOR50 permanent read-only first CPU evidence audit",audit())
