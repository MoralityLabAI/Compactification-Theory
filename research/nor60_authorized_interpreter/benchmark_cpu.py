"""NOR60 first registered four-arm signed-program CPU experiment.

The first source fixture must already exist and be archived upstream. Every
source and every timed call is journaled with fsync outside its timed region.
Existing output or journal paths are refused, including an interrupted run.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor45_authorized_program import program_authority as authority
from research.nor44_session_protocol import session_protocol as session
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor59_authorized_merkle import authorized_root
from research.nor59_authorized_merkle.benchmark_cpu import prewarm_original_signed_geometries
from research.nor60_authorized_interpreter import cohort, optimized_execution

SCHEMA = "nor60.first_60_full_signed_authorized_four_arm_CPU.v1"
REPEATS = 5
ARMS = ("original_interpreter_full_root", "original_interpreter_warm_root",
        "optimized_interpreter_full_root", "optimized_interpreter_warm_root")
MODES = {ARMS[0]: ("original", "original"), ARMS[1]: ("original", "warm"),
         ARMS[2]: ("support_memoized", "original"), ARMS[3]: ("support_memoized", "warm")}
SECURITY_MODULES = (
    "research.nor60_authorized_interpreter.test_optimized_execution",
    "research.nor60_authorized_interpreter.test_cohort_integrity",
    "research.nor60_authorized_interpreter.test_reporting",
    "research.nor59_authorized_merkle.test_authorized_root",
    "research.nor45_authorized_program.test_program_authority",
    "research.nor44_session_protocol.test_session_protocol",
    "research.nor58_static_merkle.test_template",
)


def canonical(obj):
    return cohort.canonical(obj)


def require(condition, message):
    cohort.require(condition, message)


class Journal:
    """Exclusive append-only progress with a SHA chain and durable flush."""
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("xb")
        self.previous = "0" * 64
        self.count = 0

    def write(self, event, **fields):
        core = {"ordinal": self.count, "event": event, "previous_sha256": self.previous, **fields}
        digest = sha256(canonical(core)).hexdigest()
        record = {**core, "event_sha256": digest}
        self.stream.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.previous = digest
        self.count += 1

    def close(self):
        self.stream.close()


def summarize(rows):
    if not rows:
        return {"count": 0}
    totals = {arm: sum(row["arm_medians_ms"][arm] for row in rows) for arm in ARMS}
    return {"count": len(rows), "source_median_total_ms": totals,
            "source_median_ratio_of_sums_to_original": {arm: totals[arm] / totals[ARMS[0]] for arm in ARMS},
            "median_sourcewise_ratio_to_original": {
                arm: statistics.median(row["arm_medians_ms"][arm] / row["arm_medians_ms"][ARMS[0]]
                                       for row in rows) for arm in ARMS},
            "median_cold_combined_single_call_over_original_median": statistics.median(
                row["secondary_cold_combined_ms"] / row["arm_medians_ms"][ARMS[0]] for row in rows)}


def bootstrap(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["observed_degree"], row["program_profile"])].append(row)
    if len(groups) != 10 or any(len(block) != 6 for block in groups.values()):
        return {"status": "incomplete registered ten-block panel"}
    keys = sorted(groups)
    result = {}
    for index, arm in enumerate(ARMS[1:]):
        rng = random.Random(202611600010 + index)
        draws = []
        for _ in range(4096):
            selection = [row for key in (rng.choice(keys) for _ in keys) for row in groups[key]]
            draws.append(sum(row["arm_medians_ms"][arm] for row in selection) /
                         sum(row["arm_medians_ms"][ARMS[0]] for row in selection))
        draws.sort()
        result[arm] = {"descriptive_95pct_ten_block_interval": [draws[102], draws[3993]]}
    return {"block_count": 10, "bootstrap_repeats": 4096, "descriptive_intervals": result}


def source_code_hashes():
    paths = {"research/nor60_authorized_interpreter/optimized_execution.py",
             "research/nor60_authorized_interpreter/cohort.py",
             "research/nor60_authorized_interpreter/benchmark_cpu.py",
             "research/nor45_authorized_program/independent_execution_verifier.py",
             "research/nor45_authorized_program/program_authority.py",
             "research/nor44_session_protocol/session_protocol.py",
             "research/nor43_authenticated_witness/signed_witness.py",
             "research/nor59_authorized_merkle/authorized_root.py",
             "research/nor58_static_merkle/template.py",
             "research/nor59_authorized_merkle/benchmark_cpu.py",
             "research/nor59_authorized_merkle/cohort.py",
             "research/rrc_compactification/compound25.py"}
    paths.update(name.replace(".", "/") + ".py" for name in SECURITY_MODULES)
    # Include every transitively imported repository Python file. This binds
    # NOR52/NOR51 algebra, GMP/NOR46 helpers, and provider verification too.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT):
                paths.add(path.relative_to(ROOT).as_posix())
    return {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(paths)}


def security_preflight(journal):
    command = [sys.executable, "-m", "unittest", *SECURITY_MODULES, "-v"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    receipt = {"command": ["python", "-m", "unittest", *SECURITY_MODULES, "-v"],
               "returncode": result.returncode, "log_sha256": sha256(log.encode()).hexdigest(),
               "complete_log": log, "passed": result.returncode == 0}
    journal.write("source_free_security_preflight", **receipt)
    require(receipt["passed"], "source-free signed security preflight failed")
    return receipt


def make_signed_bundles(query, program, degree, source_sha, output, expected_root):
    authority_signer = authority.test_authority_private_key()
    provider_signer, verifier_signer = trees.test_private_key(), session.test_verifier_private_key()
    authority_pub = authority.public_key_of(authority_signer)
    provider_pub, verifier_pub = trees.test_reference_public_key(), session.public_key_bytes(verifier_signer)
    tree = trees.OutputTree(output)
    require(tree.root.hex() == expected_root, "provider and independent original Merkle roots differ")
    bundles = []
    for rep in range(REPEATS):
        nonce = sha256(f"NOR60/VERIFIER-ISSUED-RESEARCH-NONCE/v1|{source_sha}|{rep}".encode("ascii")).digest()
        begin = time.perf_counter_ns()
        verifier_session = session.VerifierSession(source_sha256=source_sha, observed_degree=degree,
            coefficient_count=len(query), provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=provider_pub, verifier_signing_key=verifier_signer,
            expires_epoch=100, nonce=nonce, test_only=True)
        precommit = verifier_session.precommit(session.proposal_digest(query, verifier_session.context), now_epoch=1)
        provider = session.sign_provider_reference(tree, precommit, verifier_session.context,
                                                   verifier_pub, provider_signer, now_epoch=2)
        certificate = authority.sign_authorized_program(program=program, query=query,
            context=verifier_session.context, precommit_receipt=precommit,
            pinned_verifier_public_key=verifier_pub, expected_output_merkle_root=expected_root,
            program_authority_signer=authority_signer, now_epoch=2)
        elapsed = time.perf_counter_ns() - begin
        kwargs = {"certificate": certificate, "query": query, "session_context": verifier_session.context,
                  "precommit_receipt": precommit, "provider_reference": provider,
                  "pinned_program_authority_key": authority_pub,
                  "expected_program_authority_id": authority.TEST_AUTHORITY_ID,
                  "pinned_verifier_public_key": verifier_pub, "pinned_provider_public_key": provider_pub,
                  "now_epoch": 2}
        bundles.append({"kwargs": kwargs, "signed_setup_ns_outside_timing": elapsed,
                        "session_nonce_sha256": sha256(nonce).hexdigest(),
                        "signed_context_sha256": sha256(session.canonical(verifier_session.context)).hexdigest(),
                        "authority_certificate": certificate, "verifier_precommit": precommit,
                        "provider_reference": provider, "session_context": verifier_session.context,
                        "authority_certificate_bytes": len(session.canonical(certificate)),
                        "verifier_precommit_bytes": len(session.canonical(precommit)),
                        "provider_reference_bytes": len(session.canonical(provider))})
    require(len({bundle["session_nonce_sha256"] for bundle in bundles}) == REPEATS,
            "signed repetition sessions are not distinct")
    return bundles


def run_arm(arm, bundle):
    interpreter, root = MODES[arm]
    return nor45.verify_authorized_execution(**bundle["kwargs"],
        _interpreter_policy=interpreter, _root_policy=root)


def run(public_file: Path, journal: Journal, repeats=REPEATS):
    require(repeats == REPEATS, "first NOR60 requires exactly five repeats")
    registered = cohort.registration_sha()
    fixture, source_sha = cohort.read_fixture(public_file)
    original_program, original_root = nor45.execute_authorized_program, nor45.independently_computed_root
    original_optimized = optimized_execution.execute_authorized_program
    code_before = source_code_hashes()
    journal.write("experiment_started", registration_sha=registered, source_sha256=source_sha,
                  source_count=60, repeats=REPEATS, arms=list(ARMS), source_code_sha256=code_before)
    security = security_preflight(journal)
    authorized_root._geometry_only_warm_template.cache_clear()
    geometry = prewarm_original_signed_geometries()
    journal.write("geometry_prepared_outside_timed_calls", geometry=geometry)
    records = []
    for ordinal, case in enumerate(fixture["cases"], start=1):
        identity = {key: case[key] for key in ("source_sha256", "query_content_sha256", "program_input_sha256",
            "observed_degree", "full_length", "source_stride", "program_profile",
            "coefficient_magnitude_bits", "replicate", "family")}
        journal.write("source_started", source_ordinal=ordinal, **identity)
        samples = {arm: [] for arm in ARMS}
        try:
            query = tuple(int(value) for value in case["public_query_coefficients_decimal"])
            degree, program = case["observed_degree"], case["authorized_program"]
            structure = optimized_execution.execution_structure(program, query, degree)
            require(structure["source_derived_stride"] == case["source_stride"], "registered input support changed")
            begin = time.perf_counter_ns()
            output = original_program(program, query, degree)
            execution_ns = time.perf_counter_ns() - begin
            begin = time.perf_counter_ns()
            root = original_root(output)
            root_ns = time.perf_counter_ns() - begin
            output_vector_sha = nor45.query_digest(output)
            stride = authorized_root.certify_actual_NOR45_program_support(program=program, query=query,
                        observed_degree=degree, output=output)
            signed = make_signed_bundles(query, program, degree, case["source_sha256"], output, root)
            for arm in ARMS:
                actual = run_arm(arm, signed[0])
                require(actual["values"] == output and actual["root_sha256"] == root,
                        "pretime full vector/root parity failed: " + arm)
            orders = []
            for rep in range(REPEATS):
                order = list(ARMS)
                seed = int.from_bytes(sha256(f"NOR60/ARM-ORDER/v1|{case['source_sha256']}|{rep}".encode("ascii")).digest(), "big")
                random.Random(seed).shuffle(order)
                orders.append(order)
                for arm in order:
                    begin = time.perf_counter_ns()
                    actual = run_arm(arm, signed[rep])
                    elapsed = (time.perf_counter_ns() - begin) / 1e6
                    require(math.isfinite(elapsed) and elapsed > 0, "invalid timer sample")
                    require(actual["values"] == output and actual["root_sha256"] == root,
                            "timed full vector/root parity failed: " + arm)
                    require(nor45.execute_authorized_program is original_program and
                            nor45.independently_computed_root is original_root and
                            optimized_execution.execute_authorized_program is original_optimized,
                            "global verifier/interpreter helper changed during benchmark")
                    samples[arm].append(elapsed)
                    journal.write("timed_arm_completed", source_ordinal=ordinal,
                        source_sha256=case["source_sha256"], repetition=rep, arm=arm,
                        elapsed_ms=elapsed, exact_output_vector_sha256=output_vector_sha,
                        original_full_root_sha256=root,
                        signed_context_sha256=signed[rep]["signed_context_sha256"])
            medians = {arm: statistics.median(samples[arm]) for arm in ARMS}
            # One separately labelled cold construction call AFTER all primary samples.
            begin = time.perf_counter_ns()
            cold = nor45.verify_authorized_execution(**signed[0]["kwargs"],
                       _interpreter_policy="support_memoized", _root_policy="cold")
            cold_ms = (time.perf_counter_ns() - begin) / 1e6
            require(cold["values"] == output and cold["root_sha256"] == root, "cold combined output/root differs")
            geometry_key = f"m{len(query)}-s{stride}"
            record = {**identity, "source_success": True, "authorized_program": program,
                "all_four_full_signed_outputs_and_original_roots_equal": True,
                "certified_public_support_stride": stride,
                "original_full_merkle_root_sha256": root, "exact_output_vector_sha256": output_vector_sha,
                "exact_output_coefficients_decimal": [str(value) for value in output],
                "reference_execution_ns_outside_arms": execution_ns,
                "reference_root_ns_outside_arms": root_ns,
                "operation_structure": structure, "five_interleaved_orders": orders,
                "five_interleaved_samples_ms": samples, "arm_medians_ms": medians,
                "interpreter_over_original": medians[ARMS[2]] / medians[ARMS[0]],
                "combined_over_original": medians[ARMS[3]] / medians[ARMS[0]],
                "root_only_change_whole_verifier_over_original": medians[ARMS[1]] / medians[ARMS[0]],
                "secondary_cold_combined_ms": cold_ms,
                "secondary_cold_single_call_after_primary_samples": True,
                "template_geometry_key": geometry_key,
                "template_build_ms_outside_arms": geometry[geometry_key]["build_ns_outside_all_timed_arms"] / 1e6,
                "five_signed_bundles": [{k: v for k, v in bundle.items() if k != "kwargs"} for bundle in signed]}
            records.append(record)
            journal.write("source_completed", source_ordinal=ordinal, source_sha256=case["source_sha256"],
                          source_success=True, arm_medians_ms=medians, secondary_cold_combined_ms=cold_ms)
            print("NOR60 FIRST CPU", ordinal, "/60", f"D{degree}", f"s{stride}",
                  f"P{case['program_profile']}", "interpreter/original", round(record["interpreter_over_original"], 5),
                  "combined/original", round(record["combined_over_original"], 5), flush=True)
        except Exception as exc:
            record = {**identity, "source_success": False, "failure_type": type(exc).__name__,
                      "failure_message": str(exc)[:1000], "partial_timed_samples_ms": samples}
            records.append(record)
            journal.write("source_failed", source_ordinal=ordinal, **record)
            print("NOR60 FIRST SOURCE FAIL", ordinal, type(exc).__name__, str(exc)[:180], flush=True)
    success = [row for row in records if row["source_success"]]
    sparse = [row for row in success if row["source_stride"] > 1]
    p4 = [row for row in success if row["program_profile"] == 4]
    all_summary, sparse_summary, p4_summary = summarize(success), summarize(sparse), summarize(p4)
    same_code = code_before == source_code_hashes()
    same_helpers = nor45.execute_authorized_program is original_program and nor45.independently_computed_root is original_root and optimized_execution.execute_authorized_program is original_optimized
    gates = {
        "sparse40_interpreter_ratio_median_under_0p70": len(sparse) == 40 and sparse_summary["median_sourcewise_ratio_to_original"][ARMS[2]] < .70,
        "all60_combined_ratio_of_sums_under_0p80": len(success) == 60 and all_summary["source_median_ratio_of_sums_to_original"][ARMS[3]] < .80,
        "profile4_12_interpreter_ratio_median_under_0p65": len(p4) == 12 and p4_summary["median_sourcewise_ratio_to_original"][ARMS[2]] < .65,
        "all60_exact_signed_output_root_and_security": len(success) == 60 and security["passed"] and all(
            row["all_four_full_signed_outputs_and_original_roots_equal"] and row["certified_public_support_stride"] == row["source_stride"] for row in success),
        "first_source_and_evidence_integrity": len(success) == 60 and same_code and same_helpers
            and fixture["excluded_first_source_IDs"] == 588 and fixture["new_query_content_disjoint_from_all_historical_rows"]
            and all(all(len(row["five_interleaved_samples_ms"][arm]) == REPEATS for arm in ARMS)
                    and len({b["session_nonce_sha256"] for b in row["five_signed_bundles"]}) == REPEATS
                    for row in success),
    }
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        rss = None
    return {"schema": SCHEMA, "design_issue": 280, "first_registration_sha": registered,
        "first_public_only_source_sha256": source_sha, "source_count": 60,
        "source_success_count": len(success), "source_failure_count": 60 - len(success),
        "excluded_prior_first_source_ID_count": 588,
        "excluded_prior_query_content_unique_count": fixture["historical_query_content_unique_count"],
        "excluded_prior_program_input_unique_count": fixture["historical_program_input_unique_count"],
        "repeats": REPEATS, "arms": list(ARMS), "source_code_sha256": code_before,
        "source_free_security_preflight": security,
        "source_first_SHA_frozen_upstream_before_reference_or_timing": True,
        "same_original_authority_precommit_provider_query_checks_every_arm": True,
        "all_dynamic_results_scoped_to_single_invocation": True,
        "original_NOR45_helpers_and_source_files_unchanged_during_run": same_helpers and same_code,
        "fixture_signing_and_reference_production_outside_whole_verifier_timing": True,
        "stateless_verifier_nonce_binding_without_deployed_consumption_ledger": True,
        "all_signing_private_keys_public_test_only": True, "formal_Lean_kernel_proof_available": False,
        "geometry_template_ledger": geometry,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "cpu_model": platform.processor() or "unknown", "logical_cpu_count": os.cpu_count(),
        "maxrss_kib": rss, "dependencies": {name: importlib.metadata.version(name) for name in ("cryptography", "gmpy2")},
        "github_run_id": os.environ.get("GITHUB_RUN_ID"), "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_sha": os.environ.get("GITHUB_SHA"), "overall": all_summary,
        "primary_forty_sparse": sparse_summary, "primary_twelve_profile4": p4_summary,
        "secondary_four_dense_profile4": summarize([row for row in p4 if row["source_stride"] == 1]),
        "by_degree": {str(d): summarize([row for row in success if row["observed_degree"] == d]) for d in cohort.DEGREES},
        "by_stride": {str(s): summarize([row for row in success if row["source_stride"] == s]) for s in cohort.STRIDES},
        "by_program_profile": {str(p): summarize([row for row in success if row["program_profile"] == p]) for p in cohort.PROFILES},
        "secondary_ten_block_bootstrap": bootstrap(success),
        "secondary_total_one_time_geometry_build_ms": sum(g["build_ns_outside_all_timed_arms"] / 1e6 for g in geometry.values()),
        "first_original_five_registered_gates": gates, "first_original_engineering_pass": all(gates.values()),
        "source_records": records,
        "scope": "Synthetic exact NOR45 signed-program CPU engineering; honest authority/source premises; public test keys; no deployed nonce store, zero knowledge, succinctness, universal speedup or external-world semantic claim."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-file", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args()
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    journal_path = args.journal or args.out.with_suffix(".progress.jsonl")
    failure_path = args.out.with_suffix(".failure.json")
    require(not args.out.exists() and not journal_path.exists() and not failure_path.exists(),
            "first measurement paths already exist; overwrite or silent rerun forbidden")
    journal = Journal(journal_path)
    try:
        result = run(args.public_file, journal, args.repeats)
        journal.write("experiment_completed", source_count=result["source_count"],
                      source_success_count=result["source_success_count"],
                      gates=result["first_original_five_registered_gates"])
        journal.close()
        result["progress_journal_sha256"] = sha256(journal_path.read_bytes()).hexdigest()
        result["progress_journal_event_count"] = journal.count
        raw = canonical(result)
        with args.out.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print("NOR60 FIRST ORIGINAL CPU BYTES SHA256", len(raw), sha256(raw).hexdigest(), flush=True)
        print("NOR60 FIVE REGISTERED GATES", result["first_original_five_registered_gates"], flush=True)
        print("NOR60 FIRST ENGINEERING PASS", result["first_original_engineering_pass"], flush=True)
    except BaseException as exc:
        failure = {"schema": "nor60.first_attempt_failure.v1", "failure_type": type(exc).__name__,
                   "failure_message": str(exc)[:2000], "first_original_engineering_pass": False,
                   "partial_evidence_journal": journal_path.name, "rerun_is_not_original_first_attempt": True}
        if not journal.stream.closed:
            journal.write("experiment_fatal_failure", **failure)
            journal.close()
        failure["progress_journal_sha256"] = sha256(journal_path.read_bytes()).hexdigest()
        with failure_path.open("xb") as stream:
            stream.write(canonical(failure))
        raise


if __name__ == "__main__":
    main()
