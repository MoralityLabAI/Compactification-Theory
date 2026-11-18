"""NOR50 pre-data mechanism tests using only the prior NOR48 16 TRAIN programs.

Do not generate NOR50 new source IDs or time any first NOR50 holdout cases
in this module. NOR49 full signed controls run alongside these tests in CI.
"""
from __future__ import annotations
import unittest
from collections import Counter
from research.nor48_program_dispatch import test_dispatch as original
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor50_memoized_exact import cached

class CachedDirectControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original.NOR48Dispatch.setUpClass()
        cls.training=original.NOR48Dispatch.examples
        assert len(cls.training)==16

    def test_full_16_training_vectors_root_and_per_call_unique_ops(self):
        saved_global=nor46.exact_convolution_prefix
        r_counts=Counter()
        for program,query,degree,trace,target in self.training:
            names=cached.active_ops(program,degree,len(query))
            count=sum(op=="R" for op in names)
            r_counts[count]+=1
            invocations=[]
            def counted(source,op):
                invocations.append(op)
                return native.unary(source,op)
            actual=cached.execute_cached(program,query,degree,unary_impl=counted)
            original_values=native.execute_authorized_program(program,query,degree)
            self.assertEqual(actual,original_values)
            self.assertEqual(actual,target)
            self.assertEqual(len(invocations),len(set(names)))
            self.assertEqual(set(invocations),set(names))
            self.assertTrue(all(n==1 for n in Counter(invocations).values()))
            a=cached.exact_output_with_root(program,query,degree)
            self.assertEqual(a["output"],target)
            self.assertEqual(a["merkle_root"],native.independently_computed_root(target))
            self.assertEqual(a["active_op_count"],len(names))
            self.assertEqual(a["distinct_op_count"],len(set(names)))
        self.assertEqual(r_counts,{0:4,1:5,2:6,3:1})
        self.assertIs(nor46.exact_convolution_prefix,saved_global)

    def test_inactive_R_does_not_compute_and_invalid_program_rejected(self):
        program,query,degree,trace,target=self.training[0]
        p=dict(program)
        p["hidden_beta"]=0
        p["hidden_transform"]=None
        counted=[]
        def delegate(source,op):
            counted.append(op)
            return native.unary(source,op)
        result=cached.execute_cached(p,query,degree,unary_impl=delegate)
        self.assertEqual(result,native.execute_authorized_program(p,query,degree))
        self.assertEqual(set(counted),set(cached.active_ops(p,degree,len(query))))
        q=dict(p);q["spacing"]=1
        with self.assertRaises((ValueError,TypeError)):
            cached.execute_cached(q,query,degree)
        with self.assertRaises((ValueError,TypeError)):
            cached.execute_cached(p,(2,*query[1:]),degree)
        with self.assertRaises((ValueError,TypeError)):
            cached.execute_cached(p,(1,*[True for _ in query[1:]]),degree)

    def test_explicit_duplicate_R_control(self):
        degree=512
        query=(1,)+tuple(0 for _ in range(3*degree//2))
        program={"base":"R","u":"R","v":"R","delay":64,
                 "spacing":24,"amplitude":1,"hidden_beta":0,
                 "hidden_transform":None}
        calls=Counter()
        def spy(source,op):
            calls[op]+=1
            return native.unary(source,op)
        actual=cached.execute_cached(program,query,degree,spy)
        self.assertEqual(actual,native.execute_authorized_program(program,query,degree))
        self.assertEqual(calls,{"R":1})

if __name__=="__main__":unittest.main()
