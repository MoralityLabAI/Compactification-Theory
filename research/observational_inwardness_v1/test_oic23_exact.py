"""OIC-23 exact finite fault-adaptive query and certificate checks."""
from __future__ import annotations
import base64
import gzip
import json
import unittest
from pathlib import Path

from oic23_exact import ExactFaultGame
from oic23_reference import (
    CASE_Q, CERTIFICATE, MANIFEST, best_fixed_schedule,
    min_interfamily_distance, replay_true_scenarios, replay_stored,
)


class OIC23Tests(unittest.TestCase):
    def test_all_minimal_horizon_certificates(self):
        cert=json.loads(gzip.decompress(base64.b64decode(CERTIFICATE.read_bytes().strip())))
        for case in cert['cases']:
            with self.subTest(n=case['n'],e=case['substitutions'],r=case['marked_erasures']):
                n,e,r,Q=case['n'],case['substitutions'],case['marked_erasures'],case['Q']
                self.assertEqual(Q,CASE_Q[n][e,r])
                short=ExactFaultGame(n,Q-1,e,r)
                self.assertFalse(short.solve())
                full=ExactFaultGame(n,Q,e,r)
                self.assertTrue(full.solve())
                self.assertEqual(full.certificate(),case['policy'])
                self.assertGreater(full.verify_tree(case['policy']),0)
                self.assertGreater(replay_true_scenarios(n,Q,e,r,case['policy']),0)

    def test_fixed_schedule_strict_adaptive_advantage(self):
        d,word,ct=best_fixed_schedule(4,10)
        self.assertEqual(ct,4**9)
        self.assertEqual(d,2)
        self.assertEqual(min_interfamily_distance(4,word),d)
        self.assertTrue(ExactFaultGame(4,10,1,0).solve())

    def test_fixed_schedules_work_for_two_and_three_phases(self):
        for n in (2,3):
            Q=CASE_Q[n][1,0]
            d,word,ct=best_fixed_schedule(n,Q)
            self.assertEqual(ct,n**(Q-1))
            self.assertGreaterEqual(d,3)
            self.assertEqual(min_interfamily_distance(n,word),d)

    def test_false_policy_leaf_is_rejected(self):
        cert=json.loads(gzip.decompress(base64.b64decode(CERTIFICATE.read_bytes().strip())))
        entry=next(c for c in cert['cases'] if c['n']==3 and c['substitutions']==1 and c['marked_erasures']==0)
        tree=json.loads(json.dumps(entry['policy']))
        def mutate_first_leaf(node):
            if 'shift' in node:
                node['shift']=(node['shift']+1)%3
                return True
            return any(mutate_first_leaf(child) for child in node['children'].values())
        self.assertTrue(mutate_first_leaf(tree))
        with self.assertRaises(AssertionError):
            replay_true_scenarios(3,8,1,0,tree)

    def test_public_history_determines_adaptive_addresses(self):
        # Raw external reports determine all future chosen markers. A flipped bit
        # can change which child policy is selected; the verifier traverses both.
        cert=json.loads(gzip.decompress(base64.b64decode(CERTIFICATE.read_bytes().strip())))
        entry=next(c for c in cert['cases'] if c['n']==4 and c['substitutions']==1 and c['marked_erasures']==0)
        def paths(node):
            if 'shift' in node:return [()]
            out=[]
            for y,child in node['children'].items():
                out.extend(((node['query'],y),)+tail for tail in paths(child))
            return out
        receipts=paths(entry['policy'])
        self.assertTrue(receipts)
        self.assertTrue(all(len(p)<=10 for p in receipts))
        # Every generated policy response path has a valid indexed query action.
        self.assertTrue(all(0<=marker<4 and response in ('0','1') for p in receipts for marker,response in p))

    def test_pinned_manifest_and_compressed_policy_evidence(self):
        manifest=replay_stored()
        self.assertEqual(manifest['certificate_cases'],9)
        self.assertEqual(len(manifest['fixed_schedule_comparison']),3)
        self.assertEqual(sum(case['true_world_fault_paths_checked'] for case in manifest['threshold_cases']),2481)

    def test_validation(self):
        for args in ((1,2,1,0),(4,-1,1,0),(4,4,-1,0),(4,4,1,-1)):
            with self.assertRaises(ValueError):
                ExactFaultGame(*args)


if __name__=='__main__':unittest.main()
