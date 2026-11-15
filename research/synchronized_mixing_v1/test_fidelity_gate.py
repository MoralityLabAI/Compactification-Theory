"""Frozen SAE and adversarial one-swap fidelity tests; standard library only."""
from itertools import combinations
from fractions import Fraction
from pathlib import Path
import json
import unittest

from approx_geometry import bounded_cover, recover_with_one_swap
from fidelity_gate import audit_rows, frozen_evidence_audit
from transport_lift import all_matchings


class FidelityGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads(Path(__file__).with_name('sae_activation_evidence.json').read_text())

    def test_seed101_certified_train_but_heldout_off_manifold(self):
        v = frozen_evidence_audit(Path(__file__).with_name('sae_activation_evidence.json'))['101']
        self.assertEqual(v['geometry_status'], 'certified_on_training')
        self.assertFalse(v['heldout_only']['all_rows_on_manifold'])
        self.assertEqual(v['heldout_only']['matched'], 0)
        self.assertTrue(all(Fraction(row['unconstrained_to_matching_score_penalty']) > 0
                            for row in v['heldout_only']['rows']))

    def test_seed103_heldout_passes(self):
        v = frozen_evidence_audit(Path(__file__).with_name('sae_activation_evidence.json'))['103']
        self.assertTrue(v['heldout_only']['all_rows_on_manifold'])
        self.assertEqual(v['heldout_only']['matched'], 3)
        self.assertTrue(all(row['unconstrained_to_matching_score_penalty'] == '0'
                            for row in v['heldout_only']['rows']))

    def test_bad_sae_abstains(self):
        v = frozen_evidence_audit(Path(__file__).with_name('sae_activation_evidence.json'))['105']
        self.assertEqual(v['geometry_status'], 'abstain')

    def test_high_confidence_alias_requires_independent_holdout(self):
        edges = tuple(combinations(range(6),2))
        ms = tuple(all_matchings(tuple(range(6))))
        states = tuple(tuple(int(e in m) for e in edges) for m in ms)
        held = (0,1,5)
        keep = [i for i in range(15) if i not in held]
        bad = list(states[i] for i in keep)
        loc = keep.index(12)
        row = list(bad[loc]); row[12] = 0; row[0] = 1
        bad[loc] = tuple(row)
        wrong = bounded_cover(bad)
        good = bounded_cover([states[i] for i in keep])
        self.assertNotEqual(set(map(frozenset,good['vertex_stars'])),
                            set(map(frozenset,wrong['vertex_stars'])))
        scores = [tuple(1000 * b for b in row) for row in bad]
        self.assertTrue(audit_rows(wrong,scores)['all_rows_on_manifold'])
        self.assertEqual({r['top3_gap'] for r in audit_rows(wrong,scores)['rows']},{'1000'})
        withheld_scores = [tuple(1000 * b for b in states[i]) for i in held]
        heldout = audit_rows(wrong,withheld_scores)
        self.assertEqual(heldout['matched'],0)
        self.assertTrue(all(r['unconstrained_to_matching_score_penalty'] == '1000'
                            for r in heldout['rows']))

    def test_feature_index_relabel_is_invariant(self):
        acts = self.evidence['activation_scores_15_by_15']['103']
        ref = recover_with_one_swap(acts[:12])
        a = audit_rows(ref['geometry'],acts[12:])
        perm = (2,0,14,4,5,3,1,6,7,8,9,10,11,12,13)
        shifted = [[row[p] for p in perm] for row in acts]
        bfit = recover_with_one_swap(shifted[:12])
        b = audit_rows(bfit['geometry'],shifted[12:])
        self.assertEqual(a['matched'],b['matched'])
        self.assertEqual([r['unconstrained_to_matching_score_penalty'] for r in a['rows']],
                         [r['unconstrained_to_matching_score_penalty'] for r in b['rows']])


if __name__ == '__main__':
    unittest.main()
