"""Read-only audit of NOR60's original source, CPU result and progress journal.

This module recomputes exact reference outputs and signature bindings. It
never measures runtime, invokes the CPU benchmark or writes a source fixture.
The final receipt and archive SHA anchors are filled from observed original
artifact bytes before this auditor is published.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import statistics
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor44_session_protocol import session_protocol as session
from research.nor45_authorized_program import independent_execution_verifier as original
from research.nor45_authorized_program import program_authority as authority
from research.nor59_authorized_merkle import authorized_root
from research.nor58_static_merkle import template
from research.nor60_authorized_interpreter import cohort

HERE = Path(__file__).resolve().parent / "results"
RECEIPT_NAME = "nor60_first_receipt.json"
ARCHIVE_NAME = "nor60_first_original_hosted.zip"
SOURCE_NAME = "nor60_first_original_60_public_only.json"
CPU_NAME = "nor60_first_original_cpu.json"
JOURNAL_NAME = "nor60_first_original_progress.jsonl"
EXPECTED_RECEIPT_SHA256 = "0bc8cdbe1c50737a12863186620d7ef40faa96509e4347392a66bad58ae471cb"
EXPECTED_ARCHIVE_SHA256 = "c92e2c4e3bfcfd8e1371d1504c89d6812ac263ef270fc9d62e89d335f23da53c"
RECEIPT_SCHEMA = "nor60.first_original_60_signed_NOR45_CPU_archive.v1"
REGISTRATION = "363ef19abeba68e2897c593697ad56e2aaf4fefc"
ARMS = ("original_interpreter_full_root", "original_interpreter_warm_root",
        "optimized_interpreter_full_root", "optimized_interpreter_warm_root")
SOURCE_FIELDS = {"case_schema", "observed_degree", "full_length", "source_stride", "program_profile",
                 "coefficient_magnitude_bits", "replicate", "family", "authorized_program",
                 "public_query_coefficients_decimal", "source_sha256", "query_content_sha256",
                 "program_input_sha256"}
PROFILES = (
    {"base": "I", "u": "R", "v": "N", "delay": 64, "spacing": 24, "amplitude": 1, "hidden_beta": 0, "hidden_transform": None},
    {"base": "N", "u": "S", "v": "I", "delay": 80, "spacing": 32, "amplitude": -2, "hidden_beta": 1, "hidden_transform": "R"},
    {"base": "R", "u": "I", "v": "S", "delay": 96, "spacing": 24, "amplitude": 2, "hidden_beta": -1, "hidden_transform": "S"},
    {"base": "S", "u": "N", "v": "R", "delay": 112, "spacing": 32, "amplitude": -1, "hidden_beta": 0, "hidden_transform": None},
    {"base": "R", "u": "R", "v": "R", "delay": 80, "spacing": 24, "amplitude": 1, "hidden_beta": 2, "hidden_transform": "N"},
)
IDENTITY_FIELDS = ("source_sha256", "query_content_sha256", "program_input_sha256", "observed_degree",
                   "full_length", "source_stride", "program_profile", "coefficient_magnitude_bits",
                   "replicate", "family")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def canonical_loaded(raw, label):
    value = json.loads(raw)
    require(canonical(value) == raw, label + " is not canonical JSON")
    return value


def checked_sha(value, label):
    require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
            label + " lacks a valid SHA-256 anchor")
    return value


def checked_decimals(values, length, label):
    require(type(values) is list and len(values) == length, label + " length mismatch")
    require(all(type(v) is str and re.fullmatch(r"0|-?[1-9][0-9]*", v) is not None for v in values),
            label + " is not a canonical signed decimal vector")
    return tuple(int(v) for v in values)


def query_identity(values):
    return sha256(canonical({"length": len(values), "coefficients_decimal": values})).hexdigest()


def program_identity(row):
    return sha256(canonical({"query_content_sha256": row["query_content_sha256"],
                            "observed_degree": row["observed_degree"],
                            "authorized_program": row["authorized_program"]})).hexdigest()


def validate_source(fixture):
    require(type(fixture) is dict and set(fixture) == {"schema", "first_registration_sha", "source_domain",
        "degrees", "strides", "profiles", "replicates", "public_only", "source_count", "excluded_first_source_IDs",
        "historical_query_content_unique_count", "historical_program_input_unique_count",
        "query_content_hash_ignores_experiment_schema_and_metadata", "new_query_content_disjoint_from_all_historical_rows",
        "no_output_root_witness_session_signatures_templates_or_timings", "historical_archives", "cases"},
        "source manifest schema changed or contains extra fields")
    expected_grid = [(d, s, p, r) for d in (512, 1024) for s in (1, 4, 8)
                     for p in range(5) for r in (0, 1)]
    require(fixture["schema"] == "nor60.first_public_only_60_signed_authorized_program_sources.v1",
            "wrong first source schema")
    require(fixture["first_registration_sha"] == REGISTRATION, "wrong source registration")
    require(fixture["source_domain"] == "NOR60/SOURCE/v1", "wrong source domain")
    require(fixture["degrees"] == [512, 1024] and fixture["strides"] == [1, 4, 8]
            and fixture["profiles"] == list(range(5)) and fixture["replicates"] == [0, 1], "wrong source grid")
    for field in ("public_only", "query_content_hash_ignores_experiment_schema_and_metadata",
                  "new_query_content_disjoint_from_all_historical_rows",
                  "no_output_root_witness_session_signatures_templates_or_timings"):
        require(fixture[field] is True, "source-only invariant failed: " + field)
    rows = fixture["cases"]
    require(type(rows) is list and len(rows) == fixture["source_count"] == 60, "expected all 60 first sources")
    ids, queries, inputs = set(), set(), set()
    for row, (degree, stride, profile, rep) in zip(rows, expected_grid):
        require(type(row) is dict and set(row) == SOURCE_FIELDS, "source record schema changed")
        require((row["observed_degree"], row["source_stride"], row["program_profile"], row["replicate"])
                == (degree, stride, profile, rep), "source order or factorial cell changed")
        length, bits = 3 * degree // 2 + 1, (2, 8, 16, 2, 8)[profile]
        require(row["full_length"] == length and row["coefficient_magnitude_bits"] == bits,
                "source geometry or bit magnitude changed")
        require(row["case_schema"] == "nor60.public_exact_signed_NOR45_source.v1"
                and row["authorized_program"] == PROFILES[profile]
                and row["family"] == ("signed_random" if rep == 0 else "alternating"),
                "source program or sign family changed")
        values = checked_decimals(row["public_query_coefficients_decimal"], length, "source")
        seed = int.from_bytes(sha256(f"NOR60/SOURCE/v1|{degree}|{stride}|{profile}|{rep}".encode("ascii")).digest(), "big")
        rng = random.Random(seed)
        expected = [1]
        for index in range(1, length):
            if index % stride:
                expected.append(0)
            else:
                mag = rng.randint(1 << (bits - 1), (1 << bits) - 1)
                sign = (-1 if rng.randrange(2) else 1) if rep == 0 else (-1 if (index // stride) % 2 else 1)
                expected.append(sign * mag)
        require(values == tuple(expected), "source differs from independently replayed frozen seed")
        core = {k: v for k, v in row.items() if k not in ("source_sha256", "query_content_sha256", "program_input_sha256")}
        require(row["source_sha256"] == sha256(canonical(core)).hexdigest(), "schema-bearing source SHA mismatch")
        require(row["query_content_sha256"] == query_identity(row["public_query_coefficients_decimal"]),
                "schema-independent query content SHA mismatch")
        require(row["program_input_sha256"] == program_identity(row), "program/input content SHA mismatch")
        ids.add(row["source_sha256"])
        queries.add(row["query_content_sha256"])
        inputs.add(row["program_input_sha256"])
    require(len(ids) == len(queries) == len(inputs) == 60, "duplicate first source or exact content")
    history = cohort.historical_evidence()
    require(history["source_row_count"] == len(history["source_ids"]) == len(history["query_content_ids"]) == 588,
            "historical source/content audit did not cover all 588")
    require(len(history["program_input_ids"]) == 60, "historical authorized-program coverage changed")
    require(ids.isdisjoint(history["source_ids"]) and queries.isdisjoint(history["query_content_ids"])
            and inputs.isdisjoint(history["program_input_ids"]), "historical input was reused")
    require(fixture["historical_archives"] == history["archive_records"], "historical archive inventory changed")
    require(fixture["excluded_first_source_IDs"] == 588 and fixture["historical_query_content_unique_count"] == 588
            and fixture["historical_program_input_unique_count"] == 60, "historical audit counts changed")
    return rows


def validate_journal_bytes(raw):
    require(raw.endswith(b"\n") and bool(raw), "progress journal is empty or truncated")
    events, previous = [], "0" * 64
    for ordinal, line in enumerate(raw.splitlines(keepends=True)):
        event = json.loads(line)
        require(type(event) is dict, "journal event is not an object")
        compact = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        require(line == compact, "journal event is not canonical compact JSON")
        require(event["ordinal"] == ordinal and type(event["ordinal"]) is int, "journal ordinal changed")
        require(event["previous_sha256"] == previous, "journal chain link changed")
        digest = checked_sha(event["event_sha256"], "journal event")
        core = {k: v for k, v in event.items() if k != "event_sha256"}
        require(sha256(canonical(core)).hexdigest() == digest, "journal event SHA mismatch")
        previous = digest
        events.append(event)
    return events


def validate_samples(samples, medians):
    require(type(samples) is dict and set(samples) == set(ARMS), "timed arm set changed")
    require(type(medians) is dict and set(medians) == set(ARMS), "median arm set changed")
    for arm in ARMS:
        values = samples[arm]
        require(type(values) is list and len(values) == 5, "expected exactly five samples per arm")
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values),
                "nonpositive, nonfinite or untyped timer sample")
        require(medians[arm] == statistics.median(values), "stored source median differs from raw samples")


def validate_signed_bundles(case, row):
    query = checked_decimals(case["public_query_coefficients_decimal"], case["full_length"], "query")
    values = checked_decimals(row["exact_output_coefficients_decimal"], case["full_length"], "output")
    root = checked_sha(row["original_full_merkle_root_sha256"], "original full root")
    require(original.query_digest(values) == row["exact_output_vector_sha256"], "stored exact output vector SHA mismatch")
    require(original.independently_computed_root(values) == root, "stored full indexed Merkle root mismatch")
    program, degree = case["authorized_program"], case["observed_degree"]
    authority_pub = authority.public_key_of(authority.test_authority_private_key())
    provider_pub = trees.test_reference_public_key()
    verifier_pub = session.public_key_bytes(session.test_verifier_private_key())
    bundles = row["five_signed_bundles"]
    require(type(bundles) is list and len(bundles) == 5, "expected five original signed bundles")
    nonces = set()
    for rep, bundle in enumerate(bundles):
        context = bundle["session_context"]
        nonce = sha256(f"NOR60/VERIFIER-ISSUED-RESEARCH-NONCE/v1|{case['source_sha256']}|{rep}".encode("ascii")).digest()
        expected_context = {"protocol": session.CONTEXT_PROTOCOL, "nonce_hex": nonce.hex(),
            "source_sha256": case["source_sha256"], "observed_degree": degree,
            "coefficient_count": len(query), "provider_id": trees.PROVIDER_ID,
            "provider_key_id": sha256(provider_pub).hexdigest(), "verifier_key_id": sha256(verifier_pub).hexdigest(),
            "expires_epoch": 100, "test_only": True}
        require(context == expected_context, "signed context differs from registered source/session")
        require(bundle["session_nonce_sha256"] == sha256(nonce).hexdigest(), "signed nonce digest mismatch")
        require(bundle["signed_context_sha256"] == sha256(session.canonical(context)).hexdigest(),
                "signed context digest mismatch")
        nonces.add(bundle["session_nonce_sha256"])
        precommit, provider = bundle["verifier_precommit"], bundle["provider_reference"]
        receipt = session.verify_precommit_receipt(precommit, verifier_pub, context, 2)
        require(receipt["proposal_sha256"] == session.proposal_digest(query, context),
                "precommit proposal does not bind the original query")
        provider_root = session.verify_provider_reference(provider, context, precommit, provider_pub, verifier_pub, 2)
        require(provider_root["merkle_root_sha256"] == root, "provider signature binds a different full root")
        certificate = bundle["authority_certificate"]
        require(type(certificate) is dict and set(certificate) == {"statement", "signature_hex"},
                "authority certificate envelope changed")
        expected_statement = {"protocol": original.PROTOCOL, "semantics": original.CODE_VERSION,
            "authority_id": authority.TEST_AUTHORITY_ID, "authority_key_id": sha256(authority_pub).hexdigest(),
            "context_sha256": sha256(session.canonical(context)).hexdigest(), "session_nonce_hex": nonce.hex(),
            "source_sha256": case["source_sha256"], "observed_degree": degree, "coefficient_count": len(query),
            "provider_key_id": sha256(provider_pub).hexdigest(),
            "precommit_receipt_sha256": sha256(session.canonical(precommit)).hexdigest(),
            "query_sha256": original.query_digest(query), "expected_merkle_root_sha256": root,
            "program": program}
        require(certificate["statement"] == expected_statement, "signed authority statement differs from exact source/program")
        signature = bytes.fromhex(certificate["signature_hex"])
        require(len(signature) == 64 and signature.hex() == certificate["signature_hex"], "noncanonical authority signature")
        Ed25519PublicKey.from_public_bytes(authority_pub).verify(signature, session.canonical(expected_statement))
        for object_key, count_key in (("authority_certificate", "authority_certificate_bytes"),
                                      ("verifier_precommit", "verifier_precommit_bytes"),
                                      ("provider_reference", "provider_reference_bytes")):
            require(bundle[count_key] == len(session.canonical(bundle[object_key])), "signed object byte count mismatch")
        if rep == 0:
            verified = original.verify_authorized_execution(certificate=certificate, query=query,
                session_context=context, precommit_receipt=precommit, provider_reference=provider,
                pinned_program_authority_key=authority_pub, expected_program_authority_id=authority.TEST_AUTHORITY_ID,
                pinned_verifier_public_key=verifier_pub, pinned_provider_public_key=provider_pub, now_epoch=2)
            require(verified["values"] == values and verified["root_sha256"] == root,
                    "stored output differs from original full interpreter and signed verifier replay")
    require(len(nonces) == 5, "repeated signed session nonce")
    stride = authorized_root.certify_actual_NOR45_program_support(program=program, query=query,
        observed_degree=degree, output=values)
    require(stride == case["source_stride"] == row["certified_public_support_stride"], "support closure differs")


def summary(rows):
    if not rows:
        return {"count": 0}
    totals = {arm: sum(row["arm_medians_ms"][arm] for row in rows) for arm in ARMS}
    return {"count": len(rows), "source_median_total_ms": totals,
        "source_median_ratio_of_sums_to_original": {arm: totals[arm] / totals[ARMS[0]] for arm in ARMS},
        "median_sourcewise_ratio_to_original": {arm: statistics.median(
            row["arm_medians_ms"][arm] / row["arm_medians_ms"][ARMS[0]] for row in rows) for arm in ARMS},
        "median_cold_combined_single_call_over_original_median": statistics.median(
            row["secondary_cold_combined_ms"] / row["arm_medians_ms"][ARMS[0]] for row in rows)}


def replay_gates(rows, *, security_passed, source_integrity):
    successful = [row for row in rows if row["source_success"]]
    sparse = [row for row in successful if row["source_stride"] > 1]
    p4 = [row for row in successful if row["program_profile"] == 4]
    return {
        "sparse40_interpreter_ratio_median_under_0p70": len(sparse) == 40 and summary(sparse)["median_sourcewise_ratio_to_original"][ARMS[2]] < .70,
        "all60_combined_ratio_of_sums_under_0p80": len(successful) == 60 and summary(successful)["source_median_ratio_of_sums_to_original"][ARMS[3]] < .80,
        "profile4_12_interpreter_ratio_median_under_0p65": len(p4) == 12 and summary(p4)["median_sourcewise_ratio_to_original"][ARMS[2]] < .65,
        "all60_exact_signed_output_root_and_security": len(successful) == 60 and security_passed and all(
            row["all_four_full_signed_outputs_and_original_roots_equal"] is True
            and row["certified_public_support_stride"] == row["source_stride"] for row in successful),
        "first_source_and_evidence_integrity": len(successful) == 60 and source_integrity and all(
            all(len(row["five_interleaved_samples_ms"][arm]) == 5 for arm in ARMS)
            and len({b["session_nonce_sha256"] for b in row["five_signed_bundles"]}) == 5 for row in successful),
    }


def replay_bootstrap(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["observed_degree"], row["program_profile"]), []).append(row)
    if len(groups) != 10 or any(len(block) != 6 for block in groups.values()):
        return {"status": "incomplete registered ten-block panel"}
    keys, intervals = sorted(groups), {}
    for index, arm in enumerate(ARMS[1:]):
        rng = random.Random(202611600010 + index)
        draws = []
        for _ in range(4096):
            selected = [row for key in (rng.choice(keys) for _ in keys) for row in groups[key]]
            draws.append(sum(row["arm_medians_ms"][arm] for row in selected) /
                         sum(row["arm_medians_ms"][ARMS[0]] for row in selected))
        draws.sort()
        intervals[arm] = {"descriptive_95pct_ten_block_interval": [draws[102], draws[3993]]}
    return {"block_count": 10, "bootstrap_repeats": 4096, "descriptive_intervals": intervals}


def validate_journal_against_cpu(events, cpu):
    require(len(events) == cpu["progress_journal_event_count"], "journal event count differs from first CPU")
    require(events[0]["event"] == "experiment_started" and events[-1]["event"] == "experiment_completed",
            "first CPU does not have complete original attempt framing")
    first = events[0]
    require(first["registration_sha"] == REGISTRATION and first["source_sha256"] == cpu["first_public_only_source_sha256"]
            and first["source_count"] == 60 and first["repeats"] == 5 and first["arms"] == list(ARMS)
            and first["source_code_sha256"] == cpu["source_code_sha256"], "journal attempt framing differs")
    require(events[1]["event"] == "source_free_security_preflight", "missing original security preflight event")
    security = {k: v for k, v in events[1].items() if k not in ("ordinal", "event", "previous_sha256", "event_sha256")}
    require(security == cpu["source_free_security_preflight"], "journal security preflight differs from CPU")
    require(events[2]["event"] == "geometry_prepared_outside_timed_calls"
            and events[2]["geometry"] == cpu["geometry_template_ledger"], "geometry setup journal differs")
    cursor, timed_count = 3, 0
    for ordinal, row in enumerate(cpu["source_records"], start=1):
        require(cursor < len(events), "source progress missing")
        started = events[cursor]
        cursor += 1
        require(started["event"] == "source_started" and started["source_ordinal"] == ordinal,
                "source-start order differs")
        require(all(started[k] == row[k] for k in IDENTITY_FIELDS), "source-start identity differs")
        if row["source_success"]:
            for rep, order in enumerate(row["five_interleaved_orders"]):
                expected_order = list(ARMS)
                seed = int.from_bytes(sha256(f"NOR60/ARM-ORDER/v1|{row['source_sha256']}|{rep}".encode("ascii")).digest(), "big")
                random.Random(seed).shuffle(expected_order)
                require(order == expected_order, "interleaved arm order differs from registered seed")
                for arm in order:
                    require(cursor < len(events), "completed sample missing from journal")
                    sample = events[cursor]
                    cursor += 1
                    timed_count += 1
                    expected = {"event": "timed_arm_completed", "source_ordinal": ordinal,
                        "source_sha256": row["source_sha256"], "repetition": rep, "arm": arm,
                        "elapsed_ms": row["five_interleaved_samples_ms"][arm][rep],
                        "exact_output_vector_sha256": row["exact_output_vector_sha256"],
                        "original_full_root_sha256": row["original_full_merkle_root_sha256"],
                        "signed_context_sha256": row["five_signed_bundles"][rep]["signed_context_sha256"]}
                    require(all(sample.get(k) == value for k, value in expected.items()), "journal sample or signed context differs")
            completed = events[cursor]
            cursor += 1
            require(completed["event"] == "source_completed" and completed["source_ordinal"] == ordinal
                    and completed["source_sha256"] == row["source_sha256"] and completed["source_success"] is True
                    and completed["arm_medians_ms"] == row["arm_medians_ms"]
                    and completed["secondary_cold_combined_ms"] == row["secondary_cold_combined_ms"],
                    "source completion differs from CPU record")
        else:
            partial = {arm: [] for arm in ARMS}
            while events[cursor]["event"] == "timed_arm_completed":
                sample = events[cursor]
                require(sample["source_ordinal"] == ordinal and sample["source_sha256"] == row["source_sha256"],
                        "partial sample source mismatch")
                partial[sample["arm"]].append(sample["elapsed_ms"])
                cursor += 1
                timed_count += 1
            require(partial == row["partial_timed_samples_ms"], "retained failed-source partial samples changed")
            failure = events[cursor]
            cursor += 1
            require(failure["event"] == "source_failed" and failure["source_ordinal"] == ordinal
                    and all(failure[k] == v for k, v in row.items()), "first source failure was changed or omitted")
    require(cursor == len(events) - 1, "unexpected events after original 60 sources")
    end = events[-1]
    require(end["source_count"] == 60 and end["source_success_count"] == cpu["source_success_count"]
            and end["gates"] == cpu["first_original_five_registered_gates"], "journal final score differs")
    if cpu["source_success_count"] == 60:
        require(timed_count == 1200 and len(events) == 1324, "complete original run requires all 1200 timed rows")
    return timed_count


def validate_cpu(cpu, sources):
    require(cpu["schema"] == "nor60.first_60_full_signed_authorized_four_arm_CPU.v1" and cpu["design_issue"] == 280,
            "wrong first CPU schema or experiment")
    require(cpu["first_registration_sha"] == REGISTRATION and cpu["source_count"] == 60
            and cpu["repeats"] == 5 and cpu["arms"] == list(ARMS), "CPU registered panel changed")
    require(cpu["excluded_prior_first_source_ID_count"] == cpu["excluded_prior_query_content_unique_count"] == 588
            and cpu["excluded_prior_program_input_unique_count"] == 60, "CPU input exclusion coverage changed")
    require(cpu["python_version"].startswith("3.12.")
            and cpu["dependencies"] == {"cryptography": "46.0.4", "gmpy2": "2.3.2"}, "registered CPU dependencies changed")
    hashes = cpu["source_code_sha256"]
    require(type(hashes) is dict and bool(hashes), "missing frozen source code hashes")
    for filename, digest in hashes.items():
        require(type(filename) is str and filename.startswith("research/") and filename.endswith(".py"),
                "unsafe source registry path")
        path = (ROOT / filename).resolve()
        require(path.is_relative_to(ROOT), "source registry path escapes repository")
        require(sha256(path.read_bytes()).hexdigest() == checked_sha(digest, "source code"),
                "current source differs from original measured code: " + filename)
    required_code = {"research/nor60_authorized_interpreter/benchmark_cpu.py",
        "research/nor60_authorized_interpreter/cohort.py", "research/nor60_authorized_interpreter/optimized_execution.py",
        "research/nor45_authorized_program/independent_execution_verifier.py",
        "research/nor44_session_protocol/session_protocol.py", "research/nor59_authorized_merkle/authorized_root.py"}
    require(required_code <= set(hashes), "source hash registry lacks a required measured component")
    for field in ("source_first_SHA_frozen_upstream_before_reference_or_timing",
                  "same_original_authority_precommit_provider_query_checks_every_arm",
                  "all_dynamic_results_scoped_to_single_invocation",
                  "original_NOR45_helpers_and_source_files_unchanged_during_run",
                  "fixture_signing_and_reference_production_outside_whole_verifier_timing",
                  "stateless_verifier_nonce_binding_without_deployed_consumption_ledger",
                  "all_signing_private_keys_public_test_only"):
        require(cpu[field] is True, "measured trust/scope invariant failed: " + field)
    require(cpu["formal_Lean_kernel_proof_available"] is False, "unwarranted kernel-proof claim")
    security = cpu["source_free_security_preflight"]
    require(security["passed"] is True and security["returncode"] == 0
            and sha256(security["complete_log"].encode()).hexdigest() == security["log_sha256"],
            "first source-free security preflight failed or log changed")
    geometry = cpu["geometry_template_ledger"]
    require(set(geometry) == {f"m{3*d//2+1}-s{s}" for d in (512, 1024) for s in (1, 4, 8)},
            "registered geometry cache inventory changed")
    for degree in (512, 1024):
        for stride in (1, 4, 8):
            length = 3 * degree // 2 + 1
            row = geometry[f"m{length}-s{stride}"]
            rebuilt = template.build_template(length, stride)
            expected = {"m": length, "observed_degree": degree, "source_stride": stride,
                "template_SHA256": rebuilt.template_sha256, "canonical_template_bytes": rebuilt.canonical_template_bytes,
                "static_node_count": rebuilt.static_node_count, "static_zero_leaf_count": rebuilt.static_zero_leaf_count,
                "static_padding_leaf_count": rebuilt.static_padding_leaf_count,
                "dynamic_supported_leaf_count": len(rebuilt.dynamic_leaf_indices), "trusted_geometry_only": True}
            require(set(row) == set(expected) | {"build_ns_outside_all_timed_arms"}
                    and all(row[k] == v for k, v in expected.items()), "static geometry differs from frozen construction")
            require(type(row["build_ns_outside_all_timed_arms"]) is int and row["build_ns_outside_all_timed_arms"] > 0,
                    "invalid one-time geometry setup sample")
    require(cpu["secondary_total_one_time_geometry_build_ms"] == sum(
        geometry[f"m{3*d//2+1}-s{s}"]["build_ns_outside_all_timed_arms"] / 1e6
        for d in (512, 1024) for s in (1, 4, 8)), "geometry setup total changed")
    rows = cpu["source_records"]
    require(type(rows) is list and len(rows) == 60, "missing first source outcomes")
    for case, row in zip(sources, rows):
        require(all(row[k] == case[k] for k in IDENTITY_FIELDS), "CPU source identity differs from public roster")
        require(type(row["source_success"]) is bool, "untyped source outcome")
        if not row["source_success"]:
            require(type(row["failure_type"]) is str and type(row["failure_message"]) is str,
                    "source failure lacks original diagnostic")
            continue
        require(row["authorized_program"] == case["authorized_program"], "CPU program differs from source-only roster")
        validate_samples(row["five_interleaved_samples_ms"], row["arm_medians_ms"])
        require(type(row["five_interleaved_orders"]) is list and len(row["five_interleaved_orders"]) == 5,
                "missing interleaved repetition order")
        validate_signed_bundles(case, row)
        require(row["all_four_full_signed_outputs_and_original_roots_equal"] is True, "first vector/root parity failed")
        require(row["interpreter_over_original"] == row["arm_medians_ms"][ARMS[2]] / row["arm_medians_ms"][ARMS[0]]
                and row["combined_over_original"] == row["arm_medians_ms"][ARMS[3]] / row["arm_medians_ms"][ARMS[0]]
                and row["root_only_change_whole_verifier_over_original"] == row["arm_medians_ms"][ARMS[1]] / row["arm_medians_ms"][ARMS[0]],
                "stored sourcewise ratios differ from original samples")
        require(type(row["secondary_cold_combined_ms"]) in (int, float)
                and math.isfinite(row["secondary_cold_combined_ms"]) and row["secondary_cold_combined_ms"] > 0
                and row["secondary_cold_single_call_after_primary_samples"] is True, "cold secondary sample is invalid")
        operations = [case["authorized_program"][key] for key in ("base", "u", "v")]
        if case["authorized_program"]["hidden_beta"]:
            operations.append(case["authorized_program"]["hidden_transform"])
        expected_structure = {"source_derived_stride": case["source_stride"],
            "reduced_inverse_length": (case["full_length"] - 1) // case["source_stride"] + 1,
            "original_unary_call_count": len(operations), "optimized_unary_call_count": len(set(operations)),
            "original_R_call_count": operations.count("R"), "optimized_R_call_count": int("R" in operations),
            "R_calls_eliminated": max(0, operations.count("R") - 1),
            "distinct_unary_tokens": list(dict.fromkeys(operations)), "memo_scope": "one_authenticated_program_invocation"}
        require(row["operation_structure"] == expected_structure, "operation count or source-derived support claim changed")
        geometry_key = f"m{case['full_length']}-s{case['source_stride']}"
        require(row["template_geometry_key"] == geometry_key and row["template_build_ms_outside_arms"] ==
                geometry[geometry_key]["build_ns_outside_all_timed_arms"] / 1e6, "source geometry setup binding changed")
    successful = [row for row in rows if row["source_success"]]
    require(cpu["source_success_count"] == len(successful) and cpu["source_failure_count"] == 60 - len(successful),
            "first source failure count changed")
    groups = {"overall": successful,
              "primary_forty_sparse": [r for r in successful if r["source_stride"] > 1],
              "primary_twelve_profile4": [r for r in successful if r["program_profile"] == 4],
              "secondary_four_dense_profile4": [r for r in successful if r["program_profile"] == 4 and r["source_stride"] == 1]}
    for name, selection in groups.items():
        require(cpu[name] == summary(selection), "stored aggregate differs from original samples: " + name)
    for name, field, values in (("by_degree", "observed_degree", (512, 1024)),
                                ("by_stride", "source_stride", (1, 4, 8)),
                                ("by_program_profile", "program_profile", range(5))):
        require(cpu[name] == {str(v): summary([r for r in successful if r[field] == v]) for v in values},
                "stored subgroup summaries differ: " + name)
    require(cpu["secondary_ten_block_bootstrap"] == replay_bootstrap(successful),
            "descriptive block-bootstrap intervals differ from original samples")
    gates = replay_gates(rows, security_passed=True, source_integrity=True)
    require(cpu["first_original_five_registered_gates"] == gates and
            cpu["first_original_engineering_pass"] is all(gates.values()), "first registered score changed")
    return gates


def _utc(value):
    require(type(value) is str, "provenance timestamp is missing")
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(timestamp.tzinfo is not None, "provenance timestamp lacks UTC offset")
    return timestamp


def audit(directory=HERE):
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    directory = Path(directory)
    checked_sha(EXPECTED_RECEIPT_SHA256, "published receipt")
    checked_sha(EXPECTED_ARCHIVE_SHA256, "published archive")
    receipt_raw = (directory / RECEIPT_NAME).read_bytes()
    require(sha256(receipt_raw).hexdigest() == EXPECTED_RECEIPT_SHA256, "externally pinned original receipt SHA mismatch")
    receipt = canonical_loaded(receipt_raw, "first receipt")
    require(receipt["schema"] == RECEIPT_SCHEMA, "wrong first receipt schema")
    require(receipt["original_first_registration_commit"] == REGISTRATION, "receipt registration changed")
    require(cohort.registration_sha() == REGISTRATION, "current registration marker changed")
    archive_path = directory / ARCHIVE_NAME
    archive_raw = archive_path.read_bytes()
    require(sha256(archive_raw).hexdigest() == EXPECTED_ARCHIVE_SHA256 == receipt["archive_zip_sha256"],
            "externally pinned original archive SHA mismatch")
    require(len(archive_raw) == receipt["archive_zip_bytes"], "original archive byte count mismatch")
    with ZipFile(archive_path) as archive:
        require(archive.testzip() is None, "original ZIP integrity failure")
        require(len(archive.namelist()) == 3 and set(archive.namelist()) == {SOURCE_NAME, CPU_NAME, JOURNAL_NAME},
                "original ZIP members differ or contain duplicates")
        source_raw, cpu_raw, journal_raw = (archive.read(name) for name in (SOURCE_NAME, CPU_NAME, JOURNAL_NAME))
    for raw, prefix in ((source_raw, "raw_public_source"), (cpu_raw, "raw_first_CPU"), (journal_raw, "raw_progress_journal")):
        require(len(raw) == receipt[prefix + "_bytes"] and sha256(raw).hexdigest() == receipt[prefix + "_sha256"],
                "original member SHA or byte count differs: " + prefix)
    fixture, cpu = canonical_loaded(source_raw, "first public source"), canonical_loaded(cpu_raw, "first CPU")
    sources = validate_source(fixture)
    require(cpu["first_public_only_source_sha256"] == sha256(source_raw).hexdigest(), "CPU source SHA binding changed")
    require(cpu["progress_journal_sha256"] == sha256(journal_raw).hexdigest(), "CPU progress journal SHA binding changed")
    gates = validate_cpu(cpu, sources)
    events = validate_journal_bytes(journal_raw)
    timed_count = validate_journal_against_cpu(events, cpu)
    require(receipt["original_first_five_gates"] == gates and
            receipt["first_original_engineering_pass"] is all(gates.values()), "receipt changed original engineering outcome")
    primary = {
        "sparse40_interpreter_median": cpu["primary_forty_sparse"].get("median_sourcewise_ratio_to_original", {}).get(ARMS[2]),
        "all60_combined_ratio_of_sums": cpu["overall"].get("source_median_ratio_of_sums_to_original", {}).get(ARMS[3]),
        "profile4_12_interpreter_median": cpu["primary_twelve_profile4"].get("median_sourcewise_ratio_to_original", {}).get(ARMS[2]),
    }
    require(receipt["original_primary"] == primary, "receipt primary metrics differ from original samples")
    for field in ("original_first_run", "original_source_artifact_id", "original_CPU_artifact_id"):
        require(type(receipt[field]) is int and receipt[field] > 0, "invalid original provenance ID: " + field)
    require(str(receipt["original_first_run"]) == cpu["github_run_id"] and cpu["github_run_attempt"] == "1",
            "first CPU is not original workflow attempt one")
    require(receipt["original_trigger_commit"] == cpu["github_sha"] and
            re.fullmatch("[0-9a-f]{40}", cpu["github_sha"]) is not None, "trigger commit binding changed")
    require(_utc(receipt["upstream_source_uploaded_at"]) < _utc(receipt["cpu_job_started_at"]),
            "upstream source was not uploaded before CPU job start")
    return {"schema": "nor60.original_first_archive_audit.v1", "integrity_pass": True,
            "source_count": 60, "source_success_count": cpu["source_success_count"],
            "original_run_id": receipt["original_first_run"], "signed_bundles_authenticated": cpu["source_success_count"] * 5,
            "original_interpreter_vectors_replayed": cpu["source_success_count"], "journal_events": len(events),
            "timed_rows_preserved": timed_count, "registered_gates": gates,
            "first_original_engineering_pass": all(gates.values()),
            "source_sha256": sha256(source_raw).hexdigest(), "cpu_sha256": sha256(cpu_raw).hexdigest(),
            "journal_sha256": sha256(journal_raw).hexdigest(), "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "retimed": False, "source_fixture_written": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
