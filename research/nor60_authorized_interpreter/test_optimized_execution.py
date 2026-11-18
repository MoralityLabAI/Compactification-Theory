"""Source-free algebra and signed boundary controls. No NOR60 cohort is read."""
from __future__ import annotations
import copy
import itertools
import random
import unittest
from unittest.mock import patch

from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor59_authorized_merkle.test_authorized_root import program, signed_bundle, source
from research.nor60_authorized_interpreter import optimized_execution as optimized


class ExactInterpreterTests(unittest.TestCase):
    def test_finite_inverse_matches_independent_convolution_and_original(self):
        rng = random.Random(606001)
        for length, stride in itertools.product((1, 2, 3, 7, 8, 9, 16, 19, 33), (1, 4, 8)):
            query = tuple([1] + [rng.randrange(-7, 8) if i % stride == 0 else 0
                                  for i in range(1, length)])
            result = optimized.supported_inverse(query)
            self.assertEqual(result, nor45.unary(query, "R"))
            self.assertEqual(len(result), length)
            for n in range(length):
                self.assertEqual(sum(query[k] * result[n-k] for k in range(n+1)), int(n == 0))
            self.assertTrue(all(result[i] == 0 for i in range(length) if i % stride))

    def test_original_indices_sign_doubling_hidden_shift_and_truncation(self):
        for degree, stride, profile in itertools.product((512, 1024), (1, 4, 8), range(5)):
            query = source(degree, stride, 1)
            p = program(profile)
            actual = optimized.execute_authorized_program(p, query, degree)
            expected = nor45.execute_authorized_program(p, query, degree)
            self.assertEqual(actual, expected)
            self.assertEqual(nor45.independently_computed_root(actual),
                             nor45.independently_computed_root(expected))
            if stride > 1:
                self.assertEqual(nor45.unary(query, "N"), query)

    def test_every_grammar_operator_combination_with_sparse_toy(self):
        query = source(512, 8, 1)
        for base, left, right in itertools.product(nor45.OPS, repeat=3):
            p = program(0)
            p.update(base=base, u=left, v=right)
            self.assertEqual(optimized.execute_authorized_program(p, query, 512),
                             nor45.execute_authorized_program(p, query, 512))

    def test_unary_reuse_is_one_invocation_and_query_sensitive(self):
        p, q1 = program(4), source(512, 8, 1)
        q2 = list(q1)
        q2[8] += 3
        original = optimized.supported_inverse
        with patch.object(optimized, "supported_inverse", wraps=original) as call:
            first = optimized.execute_authorized_program(p, q1, 512)
            second = optimized.execute_authorized_program(p, q2, 512)
            again = optimized.execute_authorized_program(p, q1, 512)
            self.assertEqual(call.call_count, 3)
        self.assertEqual(first, again)
        self.assertNotEqual(first, second)
        self.assertEqual(second, nor45.execute_authorized_program(p, q2, 512))
        structure = optimized.execution_structure(p, q1, 512)
        self.assertEqual(structure["original_R_call_count"], 3)
        self.assertEqual(structure["optimized_R_call_count"], 1)
        self.assertEqual(structure["R_calls_eliminated"], 2)

    def test_reject_nonunit_types_geometry_and_unsigned_program_grammar(self):
        p, q = program(0), source(512, 8, 1)
        for bad in ((), (0, 1), (True, 0), (1, False), (1, 1.0), (1, "2"), iter(q)):
            with self.assertRaises((ValueError, TypeError)):
                optimized.supported_inverse(bad)
        for bad in (q[:-1], q + (0,), q * 2, list(q[:1]) + [False] + list(q[2:])):
            with self.assertRaises(ValueError):
                optimized.execute_authorized_program(p, bad, 512)
        for degree in (True, 513, 2048):
            with self.assertRaises(ValueError):
                optimized.execute_authorized_program(p, q, degree)
        for field, value in (("delay", 65), ("spacing", 25), ("base", "callback"),
                             ("amplitude", True), ("hidden_transform", "R")):
            badp = dict(p, **{field: value})
            with self.assertRaises(ValueError):
                optimized.execute_authorized_program(badp, q, 512)


class SignedBoundaryTests(unittest.TestCase):
    def test_all_fixed_interpreter_root_policies_share_signed_output(self):
        for stride, profile in ((1, 0), (4, 1), (8, 2), (4, 3), (8, 4)):
            kwargs, expected = signed_bundle(source(512, stride, 1), profile=profile)
            baseline = nor45.verify_authorized_execution(**kwargs)
            self.assertEqual(baseline["values"], expected)
            for root_policy in ("original", "warm", "cold"):
                actual = nor45.verify_authorized_execution(**kwargs,
                    _root_policy=root_policy, _interpreter_policy=optimized.POLICY)
                self.assertEqual(actual, baseline)
            with self.assertRaises(ValueError):
                nor45.verify_authorized_execution(**kwargs, _interpreter_policy="caller_callback")

    def test_authentication_failures_happen_before_optimized_execution(self):
        q = source(512, 8, 1)
        kwargs, _ = signed_bundle(q, profile=1)
        changed_source = list(q)
        changed_source[1] = 1
        bad_context = copy.deepcopy(kwargs["session_context"])
        bad_context["nonce_hex"] = "ff" * 32
        bad_certificate = copy.deepcopy(kwargs["certificate"])
        bad_certificate["signature_hex"] = "00" * 64
        bad_program = copy.deepcopy(kwargs["certificate"])
        bad_program["statement"]["program"]["base"] = "R"
        bad_precommit = copy.deepcopy(kwargs["precommit_receipt"])
        bad_precommit["signature_hex"] = "00" * 64
        bad_provider = copy.deepcopy(kwargs["provider_reference"])
        bad_provider["signature_hex"] = "00" * 64
        swaps = (("query", changed_source), ("session_context", bad_context),
                 ("certificate", bad_certificate), ("certificate", bad_program),
                 ("precommit_receipt", bad_precommit), ("provider_reference", bad_provider),
                 ("pinned_program_authority_key", b"0" * 32),
                 ("pinned_verifier_public_key", b"1" * 32),
                 ("pinned_provider_public_key", b"2" * 32), ("now_epoch", 101))
        with patch.object(optimized, "execute_authorized_program") as execute:
            for root_policy, (field, value) in itertools.product(("original", "warm", "cold"), swaps):
                with self.assertRaises((ValueError, TypeError)):
                    nor45.verify_authorized_execution(**dict(kwargs, **{field: value}),
                        _root_policy=root_policy, _interpreter_policy=optimized.POLICY)
            execute.assert_not_called()

    def test_cross_session_bundles_and_false_signed_provider_root_rejected(self):
        q = source(512, 8, 1)
        first, _ = signed_bundle(q, profile=1, nonce=b"a" * 32)
        second, _ = signed_bundle(q, profile=1, nonce=b"b" * 32)
        false, _ = signed_bundle(q, profile=1, nonce=b"c" * 32, provider_output=q)
        for root_policy in ("original", "warm", "cold"):
            for field in ("certificate", "precommit_receipt", "provider_reference"):
                with self.assertRaises(ValueError):
                    nor45.verify_authorized_execution(**dict(first, **{field: second[field]}),
                        _root_policy=root_policy, _interpreter_policy=optimized.POLICY)
            with self.assertRaises(ValueError):
                nor45.verify_authorized_execution(**false, _root_policy=root_policy,
                                                  _interpreter_policy=optimized.POLICY)


if __name__ == "__main__":
    unittest.main()
