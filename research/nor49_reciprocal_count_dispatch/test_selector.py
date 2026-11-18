"""NOR49 pre-timing tests use only ORIGINAL NOR48 16 timed TRAINING programs.

Never call NOR49 first 48-source benchmark or inspect new holdout timing data.
"""
from __future__ import annotations
import copy,unittest
from collections import Counter
from research.nor48_program_dispatch import test_dispatch as original_tests
from research.nor48_program_dispatch import dispatcher as original
from research.nor46_exact_integer_trace import trace_verifier as previous
from research.nor49_reciprocal_count_dispatch import selector as new


class ProspectiveSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_tests.NOR48Dispatch.setUpClass()
        cls.training=original_tests.NOR48Dispatch.examples
        assert len(cls.training)==16

    def test_frozen_count_rule_on_only_training_16(self):
        groups=Counter()
        for program,query,d,trace,target in self.training:
            count=new.reciprocal_count(program,d,len(query))
            groups[count]+=1
            predicted=new.authorized_backend(program,d,len(query))
            self.assertEqual(predicted,
                "gmp_full_trace" if count>=2 else "nor45_direct")
            for hint in (None,"gmp_full_trace","nor45_direct","attacker-override"):
                result=new.verify_math(program,query,d,trace,hint)
                self.assertEqual(result["route"],predicted)
                self.assertEqual(result["output"],target)
                self.assertEqual(result["merkle_root"],
                                 original.native.independently_computed_root(target))
        self.assertEqual(groups,{0:4,1:5,2:6,3:1})
        self.assertEqual(new.THRESHOLD,2)

    def test_source_only_R_one_remains_direct_even_with_poisoned_trace(self):
        program,query,d,trace,target=next(
            case for case in self.training if
            new.reciprocal_count(case[0],case[2],len(case[1]))==1)
        poisoned=copy.deepcopy(trace)
        poisoned["unary_transforms"]["R"][-1]="0000"
        result=new.verify_math(program,query,d,poisoned,"gmp_full_trace")
        self.assertEqual(result["route"],"nor45_direct")
        self.assertEqual(result["output"],target)
        modified=dict(program)
        modified["spacing"]=1
        with self.assertRaises((ValueError,TypeError)):
            new.verify_math(modified,query,d,trace)

    def test_two_or_more_R_reject_malicious_trace_and_preserve_global(self):
        program,query,d,trace,target=next(
            case for case in self.training if
            new.reciprocal_count(case[0],case[2],len(case[1]))>=2)
        saved=previous.exact_convolution_prefix
        self.assertEqual(new.verify_math(program,query,d,trace)["output"],target)
        corrupted=copy.deepcopy(trace)
        corrupted["unary_transforms"]["R"][-1]=str(
            int(corrupted["unary_transforms"]["R"][-1])+1)
        with self.assertRaises((ValueError,TypeError)):
            new.verify_math(program,query,d,corrupted)
        with self.assertRaises((ValueError,TypeError)):
            new.verify_math(program,query,d,None)
        self.assertIs(previous.exact_convolution_prefix,saved)


if __name__=="__main__":
    unittest.main()
