"""NOR48 source-locked reentrant GMP dispatch and negative trace controls."""
from __future__ import annotations
import copy, unittest
from collections import Counter
from research.nor47_gmp_exact_convolution import benchmark_cpu as old
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp
from research.nor46_exact_integer_trace import trace_producer as producer
from research.nor46_exact_integer_trace import trace_verifier as verifier
from research.nor48_program_dispatch import dispatcher as new


class NOR48Dispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture,grade=old.nor45.load_original()
        prior=old.archive.audit()
        archived=old.json.loads((old.ROOT/"research/nor46_exact_integer_trace/results/nor46_first_historical_cpu_cost.json").read_bytes())
        identities={r["source_sha256"] for r in archived["source_records"]}
        cls.examples=[]
        for row,receipt in zip(fixture["cases"],grade["source_records"],strict=True):
            if row["full_public_input_sha256"] not in identities:continue
            proposal=old.nor42.nor37.public_proposal(row)
            clean=proposal["pred"]
            _,hidden,beta,op=old.nor42.restore_worlds(row,clean)
            program=old.nor45.spec_from_public_proposal(proposal,row["degree"],beta,op)
            query=tuple(row["public_query_input"])
            trace=producer.transcript(program,query,row["degree"])
            cls.examples.append((program,query,row["degree"],trace,hidden))
        assert len(cls.examples)==16

    def test_route_only_by_authorized_program_and_no_global_patch(self):
        counts=Counter()
        baseline=verifier.exact_convolution_prefix
        for program,query,d,trace,expected in self.examples:
            expected_route="gmp_full_trace" if "R" in (
                program["base"],program["u"],program["v"],
                program["hidden_transform"] if program["hidden_beta"] else None
            ) else "nor45_direct"
            self.assertEqual(new.authorized_routing_key(program,d,len(query)),expected_route)
            for hint in (None,"gmp_full_trace","nor45_direct","MODEL_FAKE_BACKEND"):
                result=new.dispatch_verified_arithmetic(
                    program,query,d,trace,unsigned_backend_hint=hint)
                self.assertEqual(result["route"],expected_route)
                self.assertEqual(result["output"],expected)
                self.assertEqual(result["merkle_root"],
                    old.native.independently_computed_root(expected))
                self.assertIs(verifier.exact_convolution_prefix,baseline)
            counts[expected_route]+=1
        self.assertEqual(counts,{"gmp_full_trace":12,"nor45_direct":4})

    def test_R_tamper_signature_statement_and_wrong_trace_rejected(self):
        baseline=verifier.exact_convolution_prefix
        for program,query,d,trace,hidden in self.examples:
            route=new.authorized_routing_key(program,d,len(query))
            if route=="gmp_full_trace":
                wrong=copy.deepcopy(trace)
                wrong["unary_transforms"]["R"][-1]=str(
                    int(wrong["unary_transforms"]["R"][-1])+1)
                with self.assertRaises((verifier.TraceError,ValueError)):
                    new.dispatch_verified_arithmetic(program,query,d,wrong)
                bad=copy.deepcopy(trace)
                bad["output_coefficients"][-1]=str(
                    int(bad["output_coefficients"][-1])+1)
                with self.assertRaises((verifier.TraceError,ValueError)):
                    new.dispatch_verified_arithmetic(program,query,d,bad)
                bad=copy.deepcopy(trace)
                bad["unary_transforms"].pop("R")
                with self.assertRaises((verifier.TraceError,ValueError)):
                    new.dispatch_verified_arithmetic(program,query,d,bad)
                with self.assertRaises((verifier.TraceError,ValueError)):
                    new.dispatch_verified_arithmetic(program,query,d,None)
            else:
                # Direct source-authorized execution ignores malicious traces
                # and never selects a different backend from a model hint.
                bad=copy.deepcopy(trace)
                bad["query_sha256"]="f"*64
                v=new.dispatch_verified_arithmetic(
                    program,query,d,bad,unsigned_backend_hint="gmp_full_trace")
                self.assertEqual(v["output"],hidden)
            invalid=dict(program);invalid["delay"]=13
            with self.assertRaises((ValueError,verifier.TraceError)):
                new.dispatch_verified_arithmetic(invalid,query,d,trace)
            self.assertIs(verifier.exact_convolution_prefix,baseline)

    def test_explicit_convolution_dependency_and_error_does_not_mutate_global(self):
        initial=verifier.exact_convolution_prefix
        example=next((x for x in self.examples
                      if new.authorized_routing_key(x[0],x[2],len(x[1]))=="gmp_full_trace"))
        program,query,d,trace,hidden=example
        times=[0]
        def custom(a,b):
            times[0]+=1
            return gmp.full_gmp(a,b)
        self.assertEqual(new.verified_gmp_trace_values(trace,query,program,d,custom),hidden)
        self.assertGreater(times[0],0)
        def blow_up(a,b):
            raise RuntimeError("forced injection failure")
        with self.assertRaises(RuntimeError):
            new.verified_gmp_trace_values(trace,query,program,d,blow_up)
        self.assertIs(verifier.exact_convolution_prefix,initial)
        self.assertEqual(verifier.verified_trace_values(trace,query,program,d),hidden)


if __name__=="__main__":
    unittest.main()
