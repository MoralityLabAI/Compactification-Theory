"""OIC-29 fixed-schedule code witnesses, direct corruption and controls."""
from __future__ import annotations
import unittest
from itertools import product
from oic29_fixed import WITNESSES,all_flip_receipts,distance_spectrum,reference,verify_word,oracle_bit

class TestOIC29Fixed(unittest.TestCase):
    def test_explicit_minimum_distance(self):
        for name,word in WITNESSES.items():
            spectrum=distance_spectrum(word)
            self.assertEqual(min(spectrum),3)
            self.assertEqual(sum(spectrum.values()),1029)
            self.assertEqual(spectrum.get(0,0),0)
            self.assertEqual(spectrum.get(1,0),0)
            self.assertEqual(spectrum.get(2,0),0)

    def test_all_fault_paths_unambiguously_decode(self):
        for name,word in WITNESSES.items():
            stats=all_flip_receipts(word)
            self.assertEqual(stats['true_world_fault_paths'],49*(len(word)+1))
            self.assertGreater(stats['distinct_observed_receipts'],49)

    def test_corrupt_three_bits_can_break_guarantee(self):
        # A distance-3 cross-shift pair exists, so two faults split between
        # two possible sources can create a common received report.
        from oic29_fixed import encoded_report
        for word in WITNESSES.values():
            candidates=[(d,z,encoded_report(word,d,z)) for d,z in product(range(7),repeat=2)]
            a,b=next((x,y) for x in candidates for y in candidates if x[0]!=y[0] and (x[2]^y[2]).bit_count()==3)
            diff=a[2]^b[2]
            one=diff&(-diff)
            received=a[2]^one
            self.assertEqual((received^a[2]).bit_count(),1)
            self.assertEqual((received^b[2]).bit_count(),2)

    def test_resource_account(self):
        a=verify_word('a',WITNESSES['three_markers_q22'])
        b=verify_word('b',WITNESSES['two_markers_q27'])
        self.assertEqual((a['q'],a['distinct_marker_count']), (22,3))
        self.assertEqual((b['q'],b['distinct_marker_count']), (27,2))
        self.assertEqual(a['first_episode_total_with_offline_preprogramming'],75)
        self.assertEqual(b['first_episode_total_with_offline_preprogramming'],60)
        self.assertEqual(a['online_total_bits_with_dynamic_addressing'],88)
        self.assertEqual(b['online_total_bits_with_dynamic_addressing'],108)

    def test_oracle_equivariance_and_type(self):
        for d,z,t,m in product(range(7),range(7),range(0,23),range(7)):
            self.assertEqual(oracle_bit(d,z,t,m),oracle_bit(d,(z+3)%7,t,(m+3)%7))
        with self.assertRaises(ValueError):distance_spectrum((0,1,7))

    def test_reference_determinism(self):
        self.assertEqual(reference(),reference())
        self.assertEqual(len(reference()['schedules']),2)
        self.assertEqual(reference()['solver_status']['q21_integer_solver_20s'],'TIME_LIMIT')

if __name__=='__main__': unittest.main()
