"""Source-free adversarial tests for the independent immutable archive reader."""
import copy
import json
import random
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
from cryptography.exceptions import InvalidSignature
from research.nor45_authorized_program import independent_execution_verifier as original
from research.nor59_authorized_merkle.test_authorized_root import source, program
from research.nor60_authorized_interpreter import benchmark_cpu as benchmark
from research.nor60_authorized_interpreter import original_archive_audit as audit


def chain(events):
    result, previous = [], "0" * 64
    for ordinal, payload in enumerate(events):
        core = {"ordinal": ordinal, "previous_sha256": previous, **payload}
        previous = sha256(audit.canonical(core)).hexdigest()
        result.append({**core, "event_sha256": previous})
    return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii") for row in result)


class ArchiveStructureTests(unittest.TestCase):
    def test_published_entrypoint_refuses_missing_sha_anchors(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, "EXPECTED_RECEIPT_SHA256", "UNSET"):
            with self.assertRaisesRegex(ValueError, "anchor"):
                audit.audit(Path(directory))

    def test_canonical_bytes_and_decimal_encodings_fail_closed(self):
        self.assertEqual(audit.canonical_loaded(b'{\n  "a": 1\n}\n', "toy"), {"a": 1})
        for raw in (b'{"a":1}', b'{"a":0,"a":1}\n'):
            with self.assertRaises(ValueError):
                audit.canonical_loaded(raw, "toy")
        self.assertEqual(audit.checked_decimals(["1", "0", "-12"], 3, "toy"), (1, 0, -12))
        for value in (["1", "00"], ["1", "-0"], ["1", "+2"], ["1", "2.0"], ["1", False]):
            with self.assertRaises(ValueError):
                audit.checked_decimals(value, 2, "toy")

    def test_journal_rejects_tampering_reordering_and_truncation(self):
        raw = chain([{"event": "started"}, {"event": "sample", "elapsed_ms": 1.25}])
        self.assertEqual(len(audit.validate_journal_bytes(raw)), 2)
        for bad in (raw.replace(b"1.25", b"1.26"), b"".join(reversed(raw.splitlines(keepends=True))), raw[:-1]):
            with self.assertRaises(ValueError):
                audit.validate_journal_bytes(bad)

    def test_samples_reject_missing_nonfinite_or_changed_medians(self):
        samples = {arm: [1.0, 2.0, 3.0, 4.0, 5.0] for arm in audit.ARMS}
        medians = {arm: 3.0 for arm in audit.ARMS}
        audit.validate_samples(samples, medians)
        for value in ([1.0] * 4, [1.0, 2.0, float("nan"), 4.0, 5.0],
                      [1.0, 2.0, float("inf"), 4.0, 5.0], [1.0, 2.0, 0, 4.0, 5.0],
                      [1.0, 2.0, True, 4.0, 5.0]):
            with self.assertRaises(ValueError):
                audit.validate_samples(dict(samples, **{audit.ARMS[0]: value}), medians)
        with self.assertRaises(ValueError):
            audit.validate_samples(samples, dict(medians, **{audit.ARMS[0]: 2.0}))

    def test_journal_sample_context_and_registered_order_match_saved_cpu(self):
        row = {key: 0 for key in audit.IDENTITY_FIELDS}
        row.update(source_sha256="a" * 64, source_success=True,
            exact_output_vector_sha256="b" * 64, original_full_merkle_root_sha256="c" * 64,
            arm_medians_ms={arm: 3.0 for arm in audit.ARMS}, secondary_cold_combined_ms=4.0,
            five_interleaved_samples_ms={arm: [1.0, 2.0, 3.0, 4.0, 5.0] for arm in audit.ARMS},
            five_signed_bundles=[{"signed_context_sha256": str(rep)} for rep in range(5)],
            five_interleaved_orders=[])
        cpu = {"first_public_only_source_sha256": "d" * 64, "source_code_sha256": {},
               "source_free_security_preflight": {"passed": True}, "geometry_template_ledger": {},
               "source_records": [row], "source_success_count": 1,
               "first_original_five_registered_gates": {"toy": False}}
        events = [{"event": "experiment_started", "registration_sha": audit.REGISTRATION,
                   "source_sha256": cpu["first_public_only_source_sha256"], "source_count": 60,
                   "repeats": 5, "arms": list(audit.ARMS), "source_code_sha256": {}},
                  {"event": "source_free_security_preflight", "passed": True},
                  {"event": "geometry_prepared_outside_timed_calls", "geometry": {}},
                  {"event": "source_started", "source_ordinal": 1,
                   **{key: row[key] for key in audit.IDENTITY_FIELDS}}]
        for rep in range(5):
            order = list(audit.ARMS)
            seed = int.from_bytes(sha256(f"NOR60/ARM-ORDER/v1|{row['source_sha256']}|{rep}".encode("ascii")).digest(), "big")
            random.Random(seed).shuffle(order)
            row["five_interleaved_orders"].append(order)
            for arm in order:
                events.append({"event": "timed_arm_completed", "source_ordinal": 1,
                    "source_sha256": row["source_sha256"], "repetition": rep, "arm": arm,
                    "elapsed_ms": float(rep + 1), "exact_output_vector_sha256": "b" * 64,
                    "original_full_root_sha256": "c" * 64, "signed_context_sha256": str(rep)})
        events.extend([{"event": "source_completed", "source_ordinal": 1, "source_sha256": "a" * 64,
                        "source_success": True, "arm_medians_ms": row["arm_medians_ms"], "secondary_cold_combined_ms": 4.0},
                       {"event": "experiment_completed", "source_count": 60, "source_success_count": 1,
                        "gates": {"toy": False}}])
        cpu["progress_journal_event_count"] = len(events)
        self.assertEqual(audit.validate_journal_against_cpu(audit.validate_journal_bytes(chain(events)), cpu), 20)
        changed = copy.deepcopy(events)
        changed[4]["signed_context_sha256"] = "wrong-session"
        with self.assertRaisesRegex(ValueError, "context"):
            audit.validate_journal_against_cpu(audit.validate_journal_bytes(chain(changed)), cpu)
        changed = copy.deepcopy(events)
        changed[4], changed[5] = changed[5], changed[4]
        with self.assertRaises(ValueError):
            audit.validate_journal_against_cpu(audit.validate_journal_bytes(chain(changed)), cpu)

    def test_registered_gates_use_all_first_sources_and_strict_thresholds(self):
        rows = []
        for degree in (512, 1024):
            for stride in (1, 4, 8):
                for profile in range(5):
                    for rep in (0, 1):
                        medians = {audit.ARMS[0]: 1.0, audit.ARMS[1]: .9,
                                   audit.ARMS[2]: .3, audit.ARMS[3]: .2}
                        rows.append({"source_success": True, "source_stride": stride, "program_profile": profile,
                            "all_four_full_signed_outputs_and_original_roots_equal": True,
                            "certified_public_support_stride": stride, "arm_medians_ms": medians,
                            "secondary_cold_combined_ms": .25,
                            "five_interleaved_samples_ms": {arm: [value] * 5 for arm, value in medians.items()},
                            "five_signed_bundles": [{"session_nonce_sha256": str(n)} for n in range(5)]})
        self.assertTrue(all(audit.replay_gates(rows, security_passed=True, source_integrity=True).values()))
        failed = copy.deepcopy(rows)
        failed[0]["source_success"] = False
        gates = audit.replay_gates(failed, security_passed=True, source_integrity=True)
        self.assertFalse(gates["all60_combined_ratio_of_sums_under_0p80"])
        self.assertFalse(gates["all60_exact_signed_output_root_and_security"])
        boundary = copy.deepcopy(rows)
        for row in boundary:
            row["arm_medians_ms"][audit.ARMS[2]] = .70
        self.assertFalse(audit.replay_gates(boundary, security_passed=True, source_integrity=True)[
            "sparse40_interpreter_ratio_median_under_0p70"])


class SignedArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        query, p = source(512, 8, 1), program(4)
        values = original.execute_authorized_program(p, query, 512)
        root = original.independently_computed_root(values)
        source_sha = sha256(b"NOR60-ARCHIVE-AUDIT-SOURCE-FREE-TOY").hexdigest()
        bundles = benchmark.make_signed_bundles(query, p, 512, source_sha, values, root)
        cls.case = {"public_query_coefficients_decimal": [str(v) for v in query], "full_length": len(query),
                    "authorized_program": p, "observed_degree": 512, "source_sha256": source_sha, "source_stride": 8}
        cls.row = {"exact_output_coefficients_decimal": [str(v) for v in values],
                   "original_full_merkle_root_sha256": root, "exact_output_vector_sha256": original.query_digest(values),
                   "certified_public_support_stride": 8,
                   "five_signed_bundles": [{k: v for k, v in b.items() if k != "kwargs"} for b in bundles]}

    def test_replay_original_vector_and_all_five_signed_bindings(self):
        audit.validate_signed_bundles(self.case, self.row)

    def test_session_replay_forgery_or_changed_exact_output_rejected(self):
        altered = []
        bad = copy.deepcopy(self.row)
        bad["five_signed_bundles"][4] = copy.deepcopy(bad["five_signed_bundles"][0])
        altered.append(bad)
        bad = copy.deepcopy(self.row)
        bad["five_signed_bundles"][4]["authority_certificate"]["signature_hex"] = "00" * 64
        altered.append(bad)
        bad = copy.deepcopy(self.row)
        bad["five_signed_bundles"][3]["provider_reference"]["signature_hex"] = "00" * 64
        altered.append(bad)
        bad = copy.deepcopy(self.row)
        bad["exact_output_coefficients_decimal"][8] = "999"
        altered.append(bad)
        bad = copy.deepcopy(self.row)
        bad["five_signed_bundles"][1]["verifier_precommit"]["payload"]["proposal_sha256"] = "0" * 64
        altered.append(bad)
        for candidate in altered:
            with self.assertRaises((ValueError, InvalidSignature)):
                audit.validate_signed_bundles(self.case, candidate)


if __name__ == "__main__":
    unittest.main()
