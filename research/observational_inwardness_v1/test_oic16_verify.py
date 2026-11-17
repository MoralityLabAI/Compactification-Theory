"""OIC-16 independent receipt and finite theorem tests (stdlib only)."""
import copy
import math
import unittest

from oic16_verify import (
    load_receipt,validate_run,verify,filter_episode,oracle,report_panel,
    word_for,validate_summary,
)
from oic16_scale import neighbor_gap,run as scaling_run
from oic16_projection import projected_sets,oracle_sets


class OIC16IndependentReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta,cls.data=load_receipt()

    def test_manifest_and_full_archive(self):
        self.assertEqual(self.meta['run_count'],20)
        self.assertEqual(self.meta['full_panel_count'],600)
        self.assertEqual(self.meta['relabel_control_count'],100)
        self.assertEqual(self.meta['fault_case_count'],1080)
        self.assertEqual(len(self.data['runs']),20)
        self.assertTrue(validate_summary(self.meta,self.data))

    def test_torch_free_one_seed_full_recompute(self):
        self.assertTrue(validate_run(self.data['runs'][0]))

    def test_mutated_probability_is_detected(self):
        mutated=copy.deepcopy(self.data['runs'][0])
        mutated['panels']['8']['full_active']['probabilities'][0]=0.81
        with self.assertRaises(ValueError):
            validate_run(mutated)

    def test_mutated_kernel_is_detected(self):
        mutated=copy.deepcopy(self.data['runs'][0])
        mutated['weights']['kernel_logits'][1]=[100.,0.,0.,0.,0.]
        with self.assertRaises(ValueError):
            validate_run(mutated)

    def test_phase_reflection_indistinguishable(self):
        for n in range(3,13):
            for word in ('0'*(2*n),'1'*n,'10100'*(n//5+1)):
                word=word[:2*n]
                forward=set()
                backward=set()
                for z in range(n):
                    forward.add(tuple(oracle(n,1,z,word)))
                    # Reflect z and transition sign, with marker=0.
                    x=0; phase=(-z)%n; trace=[0]
                    for u in word:
                        x ^= 1 ^ (int(u) & int(phase==0))
                        phase=(phase-1)%n
                        trace.append(x)
                    backward.add(tuple(trace))
                self.assertEqual(forward,backward)

    def test_fourier_resolution_bound(self):
        for r in self.data['runs']:
            w=r['weights']; b,c=w['emission_coeff'][1:]
            for n in (4,5,8,16,32,64,128,256):
                gap,bound=neighbor_gap(n,w)
                self.assertLessEqual(abs(gap),bound+1e-12)
            self.assertGreater(abs(neighbor_gap(32,w)[0]),abs(neighbor_gap(128,w)[0]))

    def test_reference_posterior_passive_gate(self):
        r=self.data['runs'][0]
        for n in (4,5,6,7,8):
            result=report_panel(n,'0'*(2*n),r['weights'])
            self.assertEqual(result['accuracy'],.5)
            self.assertLess(result['mean_passive_deviation'],1e-5)

    def test_full_thorough_verification(self):
        verified=verify()
        self.assertEqual(verified['run_count'],20)


if __name__=='__main__':
    unittest.main()
