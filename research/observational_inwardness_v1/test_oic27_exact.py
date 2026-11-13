"""Exact n6 robust adaptive sensor certificates and independent control tests."""
import copy
from itertools import product
import json
import unittest

from oic27_reference import (WITNESS_21,check,digest,fixed_schedule_parity_lower_bound,
                             minimum_cross,MANIFEST,ARCHIVE)
from oic27_verify import clean_bit,check_belief_tree,replay_worlds,verify
import base64,gzip


class OIC27ExactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=check()
        cls.certs=json.loads(gzip.decompress(base64.b64decode(ARCHIVE.read_bytes().strip())))

    def test_separately_constructed_winning_policy_trees(self):
        for tag,tree in self.certs.items():
            with self.subTest(tag=tag):
                self.assertEqual(verify(tree,15),self.manifest['positive_certificate_checks'][tag])
                self.assertLessEqual(verify(tree,15)['belief']['max_depth'],15)

    def test_entire_true_world_adversarial_replay(self):
        for tag,tree in self.certs.items():
            receipt=replay_worlds(tree,15)
            self.assertEqual(receipt['worlds'],36)
            self.assertGreater(receipt['paths'],36)

    def test_tampering_with_terminal_shift_is_rejected(self):
        original=self.certs['affine']
        trial=copy.deepcopy(original)
        def corrupt(node):
            if 'shift' in node:
                node['shift']=(node['shift']+1)%6
                return True
            for child in node['children'].values():
                if corrupt(child):return True
            return False
        self.assertTrue(corrupt(trial))
        with self.assertRaises(AssertionError):check_belief_tree(trial,15)

    def test_translation_and_unit_phase_relabel_gauge(self):
        for u,c,d,z,m,t in product((1,5),range(6),range(6),range(6),range(6),range(15)):
            orig=clean_bit(d,z,t,m)
            changed=clean_bit((u*d)%6,(u*z+c)%6,t,(u*m+c)%6)
            self.assertEqual(orig,changed)

    def test_pairwise_future_difference_ceiling(self):
        # Any *single* sensor marker differentiates the two clean bits at
        # step t exactly when the phase positions differ and it matches one
        # of those two positions. Exhaustively verify the maximizing bit.
        for d,z,d2,z2,t in product(range(6),range(6),range(6),range(6),range(9)):
            a=(z+t*d)%6;b=(z2+t*d2)%6
            best=max(clean_bit(d,z,t,m)^clean_bit(d2,z2,t,m) for m in range(6))
            self.assertEqual(best,int(a!=b))

    def test_fixed_analytic_lower_bound(self):
        self.assertEqual(fixed_schedule_parity_lower_bound(6,1),18)
        # There are only 8 even clock steps and 7 odd steps at Q=15.
        # Each time-parity residue needs >=9 addressed measurements for
        # all antipodal phase pairs to be separated by distance >=3.
        self.assertLess((15+1)//2,9)
        self.assertLess(15//2,9)

    def test_fixed_21_witness_for_one_substitution(self):
        self.assertEqual(len(WITNESS_21),21)
        self.assertEqual(minimum_cross(WITNESS_21),3)
        self.assertEqual(self.manifest['fixed_schedule']['verified_q21_min_cross_shift_distance'],3)

    def test_certificates_source_integrity_and_resource_ledger(self):
        x=check()
        self.assertEqual(x['negative_translation_states'],1639144)
        self.assertEqual(x['negative_affine_states'],813921)
        self.assertEqual(x['online_resource']['total_bits_direct'],60)
        self.assertEqual(x['exact_minimum_adaptive_queries'],15)


if __name__=='__main__':unittest.main()
