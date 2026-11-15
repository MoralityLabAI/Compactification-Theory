"""COE-06: pure standard-library tests of phase-blind observer capacity."""
from __future__ import annotations
import json
import random
import unittest
from pathlib import Path

try:
    from .gauge_orbit_codes import (WIDTH, WEIGHT, ORBIT_WORDS, certificate,
        decode_under_unknown_phase, necklace, orbit, orbit_distance, rotate, trace)
    from .resource_bound import all_small_bounds, deletion_bound
    from .phase_gauge_replay import evidence, evaluate, load_json
except ImportError:
    from gauge_orbit_codes import (WIDTH, WEIGHT, ORBIT_WORDS, certificate,
        decode_under_unknown_phase, necklace, orbit, orbit_distance, rotate, trace)
    from resource_bound import all_small_bounds, deletion_bound
    from phase_gauge_replay import evidence, evaluate, load_json

HERE=Path(__file__).resolve().parent


class TestGaugeInvariantCoding(unittest.TestCase):
    def test_shared_time_independent_readout(self):
        for word in ORBIT_WORDS:
            self.assertEqual(trace(word),word)
            self.assertEqual(trace(rotate(word)),rotate(word))
            self.assertEqual(rotate(word,WIDTH).bit_count(),WEIGHT)

    def test_cyclic_min_distance_and_21_labels(self):
        self.assertEqual(len(ORBIT_WORDS),21)
        self.assertEqual(len(set(necklace(x) for x in ORBIT_WORDS)),21)
        for i,x in enumerate(ORBIT_WORDS):
            self.assertEqual(x.bit_count(),6)
            for j,y in enumerate(ORBIT_WORDS):
                if i!=j:
                    self.assertGreaterEqual(orbit_distance(x,y),4)

    def test_every_phase_single_flip(self):
        for i,word in enumerate(ORBIT_WORDS):
            for shift in orbit(word):
                self.assertEqual(decode_under_unknown_phase(shift),i)
                for j in range(WIDTH):
                    self.assertEqual(decode_under_unknown_phase(shift ^ (1<<j)),i)

    def test_bad_pair_rejected(self):
        with self.assertRaises(ValueError):
            certificate((0b1111,0b1110),4,0)
        # Two distinct weight-six necklaces at distance two would defeat
        # one-flip recovery under at least one unknown phase.
        with self.assertRaises(ValueError):
            certificate((ORBIT_WORDS[0],ORBIT_WORDS[0] ^ 1 ^ 256), WIDTH, WEIGHT)

    def test_all_code_certificates_match_receipt(self):
        self.assertEqual(json.loads(json.dumps(certificate())),json.loads((HERE/'COE06_RECEIPT.json').read_text()))


class TestExactOrbitBounds(unittest.TestCase):
    def test_complete_integer_enumeration_before_14(self):
        rows=all_small_bounds(13)
        self.assertEqual(len(rows), sum(t//2+1 for t in range(1,14)))
        self.assertEqual(max(r['combined_bound'] for r in rows),17)
        self.assertEqual(max(r['combined_bound'] for r in rows if r['T']==12),14)
        self.assertEqual(max(r['combined_bound'] for r in rows if r['T']==13),17)
        self.assertEqual(json.loads(json.dumps(rows)),json.loads((HERE/'COE06_BOUND_LEDGER.json').read_text()))

    def test_worst_case_resource_count(self):
        worst=deletion_bound(13,6)
        self.assertEqual(worst['total_resources'],99)
        self.assertEqual(worst['usage_counts'],{5:6,6:126})
        self.assertEqual(worst['bound'],17)
        self.assertEqual(deletion_bound(12,6)['bound'],14)

    def test_orbit_deletion_resource_disjointness(self):
        # Independent check on every candidate binary necklace of width 8,
        # weight 4: sharing a deletion necklace forces min orbit distance 2.
        from itertools import combinations
        T,w=8,4
        reps=sorted({necklace(x,T) for x in range(1<<T) if x.bit_count()==w})
        def resources(word):
            return {necklace(word ^ (1<<j),T) for j in range(T) if word>>j&1}
        for a,b in combinations(reps,2):
            if resources(a)&resources(b):
                self.assertEqual(orbit_distance(a,b,T),2)


class TestFrozenModelGaugeDiagnostic(unittest.TestCase):
    def test_unaltered_probe_replay(self):
        bundle=load_json(HERE/'replication_activations.json.gz')
        receipt=load_json(HERE/'new_fit_phase_receipt.json')
        result=evidence(bundle,receipt)
        self.assertEqual(result['pooled'],{
            'ordered':243,'cyclic':233,'count':233,'order_only':10,
            'cyclic_beyond_count':0,'phase_sensitive_only':10,'denominator':336})
        self.assertEqual(result,json.loads((HERE/'COE06_GAUGE_REPLAY.json').read_text()))

    def test_thresholds_unchanged_on_context_holdout(self):
        r=evidence(load_json(HERE/'replication_activations.json.gz'),
                   load_json(HERE/'new_fit_phase_receipt.json'))
        for item in r['cohort_results']:
            self.assertLessEqual(item['count'],item['cyclic'])
            self.assertLessEqual(item['cyclic'],item['ordered'])


if __name__=='__main__':unittest.main()
