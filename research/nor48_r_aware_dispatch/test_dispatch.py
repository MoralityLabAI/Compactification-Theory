"""NOR-48 signed R dispatcher on original 16 historical source programs."""
from __future__ import annotations
import copy,unittest
from hashlib import sha256

from research.nor48_r_aware_dispatch import dispatch
from research.nor48_r_aware_dispatch import historical_cases as cases
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as previous


class SignedProgramDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=cases.historical_signed_16()

    def test_signed_r_classification_all_original_16(self):
        self.assertEqual(len(self.cases),16)
        for case in self.cases:
            opts=case["opts"]
            reported=dispatch.authenticated_requires_reciprocal(
                certificate=opts["certificate"],
                pinned_authority_public_key=opts["pinned_program_authority_key"],
                expected_authority_id=opts["expected_program_authority_id"],
                query=opts["query"],
                observed_degree=opts["session_context"]["observed_degree"])
            self.assertEqual(reported,case["requires_R"])
        self.assertEqual(sum(x["requires_R"] for x in self.cases),12)

    def test_all_16_dispatch_match_original_root_and_direct_program(self):
        previous_fn=previous.exact_convolution_prefix
        for case in self.cases:
            opts=case["opts"]
            actual=dispatch.verified_historical_dispatch(**opts)
            self.assertEqual(actual["values"],case["true_output"])
            self.assertEqual(actual["root_sha256"],case["signed_root"])
            self.assertEqual(actual["source_sha256"],case["source_sha256"])
            self.assertEqual(actual["requires_R"],case["requires_R"])
            self.assertEqual(actual["authority_id"],opts["expected_program_authority_id"])
            self.assertTrue(case["requires_R"]==("GMP" in actual["backend"]))
            self.assertIs(previous.exact_convolution_prefix,previous_fn)
            # The separately authorized NOR45 direct whole execution is the
            # reference for every old hidden world in the historical cohort.
            baseline=native.verify_authorized_execution(**{
                k:v for k,v in opts.items() if k!="trace_witness"})
            self.assertEqual(actual["values"],baseline["values"])
            self.assertEqual(actual["root_sha256"],baseline["root_sha256"])

    def test_bad_program_authority_source_session_and_model_selected_root(self):
        case=next(z for z in self.cases if z["requires_R"])
        args=case["opts"]
        invalid=copy.deepcopy(args["certificate"])
        invalid["statement"]["program"]["base"]="I"
        with self.assertRaises(ValueError):
            dispatch.verified_historical_dispatch(**{
                **args,"certificate":invalid})
        with self.assertRaises(ValueError):
            dispatch.verified_historical_dispatch(**{
                **args,"pinned_program_authority_key":sha256(b"attacker key").digest()})
        bad_context=copy.deepcopy(args["session_context"])
        bad_context["nonce_hex"]="00"*32
        with self.assertRaises(ValueError):
            dispatch.verified_historical_dispatch(**{
                **args,"session_context":bad_context})
        bad_query=list(args["query"])
        bad_query[11]+=1
        with self.assertRaises(ValueError):
            dispatch.verified_historical_dispatch(**{
                **args,"query":bad_query})
        bad_reference=copy.deepcopy(args["provider_reference"])
        # Provider root substitution must fail even if trace itself is valid.
        bad_reference["signed_reference"]["manifest"]["merkle_root_sha256"]="00"*32
        with self.assertRaises((ValueError,KeyError,TypeError)):
            dispatch.verified_historical_dispatch(**{
                **args,"provider_reference":bad_reference})

    def test_mutated_R_trace_rejected_direct_path_ignores_trace(self):
        case=next(z for z in self.cases if z["requires_R"])
        args=case["opts"]
        witness=copy.deepcopy(args["trace_witness"])
        witness["output_coefficients"][-1]=str(
            int(witness["output_coefficients"][-1])+1)
        with self.assertRaises(ValueError):
            dispatch.verified_historical_dispatch(**{
                **args,"trace_witness":witness})
        case=next(z for z in self.cases if not z["requires_R"])
        args=case["opts"]
        witness=copy.deepcopy(args["trace_witness"])
        witness["output_coefficients"][-1]=str(
            int(witness["output_coefficients"][-1])+7)
        correct=dispatch.verified_historical_dispatch(**{
            **args,"trace_witness":witness})
        self.assertEqual(correct["values"],case["true_output"])
        self.assertFalse(correct["requires_R"])

    def test_untrusted_backend_selector_cannot_be_supplied(self):
        opts=self.cases[0]["opts"]
        with self.assertRaises(TypeError):
            dispatch.verified_historical_dispatch(**{**opts,"backend":"direct"})

    def test_gmp_benchmark_global_is_restored_on_exception(self):
        original=previous.exact_convolution_prefix
        with self.assertRaises(RuntimeError):
            with dispatch.temporary_exact_gmp_backend():
                self.assertIs(previous.exact_convolution_prefix,
                              dispatch.gmp_kernel.full_gmp)
                raise RuntimeError("synthetic fault")
        self.assertIs(previous.exact_convolution_prefix,original)


if __name__=="__main__":
    unittest.main()
