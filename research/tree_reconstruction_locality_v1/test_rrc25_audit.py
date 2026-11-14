"""KSR-01C frozen native recurrent adapter regression suite."""
import itertools
import random
import unittest
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from audit_rrc25 import (Q, BASE_SEED, STATUS, analyze_cohort, digest,
    gru_step, logits, make_bank, make_cohorts, native_root, row_law,
    run, state_label, tree_laws)
from research.rrc_compactification import bench25, compound25 as source


class FrozenNativeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models, hashes=bench25.checkpoints()
        cls.checkpoint=models['recurrent']
        cls.hash=hashes['recurrent']
        cls.cases=make_cohorts(cal_per=1,test_per=1)
        cls.cal=[native_root(c,cls.checkpoint) for c in cls.cases['calibration']]
        cls.bank,cls.bank_meta=make_bank(cls.cal,2)

    def test_original_frozen_checkpoint_and_native_logit_replay(self):
        self.assertEqual(self.checkpoint['parameters'],1000)
        self.assertEqual(self.hash,'54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b')
        for c in self.cases['heldout']:
            r=native_root(c,self.checkpoint)
            self.assertEqual(r['native_logit_max_abs_error'],0.0)
            self.assertEqual(len(r['hidden']),12)
            self.assertIn(r['label'],range(4))

    def test_no_future_or_hidden_program_in_root_state(self):
        c=self.cases['heldout'][0]
        altered=replace(c,future=tuple(9999999 for _ in c.future),program=None)
        self.assertEqual(native_root(c,self.checkpoint),native_root(altered,self.checkpoint))

    def test_calibration_bank_and_disjoint_visible_roots(self):
        cal={c.visible for c in self.cases['calibration']}
        self.assertFalse(cal & {c.visible for c in self.cases['heldout']})
        bank,meta=make_bank(self.cal,2)
        self.assertEqual(bank,self.bank)
        self.assertEqual(digest(meta),digest(self.bank_meta))
        self.assertTrue(all(x['case_id'].startswith('ksr01c-calibration') for x in meta))

    def test_full_depth_two_tree_matches_independent_edge_bruteforce(self):
        h=self.cal[0]['hidden']; bank=self.bank; K=len(bank)
        reference=tree_laws(h,bank,self.checkpoint)
        brute=defaultdict(F)
        for choices in itertools.product(range(K),repeat=6):
            left=gru_step(h,bank[choices[0]],self.checkpoint)
            right=gru_step(h,bank[choices[1]],self.checkpoint)
            leaves=[gru_step(parent,bank[index],self.checkpoint)
                    for parent,index in zip((left,left,right,right),choices[2:])]
            labels=[state_label(x,self.checkpoint) for x in leaves]
            histogram=tuple(labels.count(j) for j in range(Q))
            brute[histogram]+=F(1,K**6)
        self.assertEqual(dict(brute),reference['counts'][2])
        self.assertEqual(sum(reference['counts'][1].values()),1)
        self.assertEqual(sum(reference['counts'][2].values()),1)

    def test_native_coarse_state_audit_has_finite_law_scope(self):
        result=analyze_cohort(self.cal,self.bank,self.checkpoint)
        self.assertEqual(result['root_count'],3)
        self.assertFalse(result['original_uniform_symmetric_KSR_floor_applied'])
        self.assertFalse(result['positive_asymptotic_neural_transfer_claim'])
        for s in result['population_count_decoding']:
            self.assertLessEqual(F(s['majority_prior']),F(s['covariance_floor']))
            self.assertLessEqual(F(s['covariance_floor']),F(s['count_MAP']))
            self.assertLessEqual(F(s['count_MAP']),1)
            self.assertTrue(s['decoder_fit_to_same_finite_prior'])
        self.assertFalse(result['stationary_markov_composition_check']['available'])

    def test_small_complete_report_and_source_hashes(self):
        r=run(cal_per=1,test_per=1,bank_size=2)
        self.assertEqual(r['status'],STATUS)
        self.assertEqual(r['checkpoint_sha256'],self.hash)
        self.assertEqual(r['calibration']['root_count'],3)
        self.assertEqual(r['heldout']['root_count'],6)
        self.assertTrue(r['public_visible_cohorts_disjoint'])
        self.assertFalse(r['input_bank_fit_uses_heldout_roots'])
        self.assertFalse(r['model_weights_modified'])
        self.assertFalse(r['original_Q5_weights_evaluated'])
        self.assertEqual(len(r['input_bank_sha256']),64)
        self.assertEqual(len(r['experiment_sha256']),64)
        self.assertEqual(r['heldout']['native_gru_steps_in_branch_enumeration'],36)

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError): make_cohorts(0,1)
        with self.assertRaises(ValueError): make_bank(self.cal,0)
        with self.assertRaises(ValueError): gru_step([0.0]*11,[0.0]*8,self.checkpoint)
        with self.assertRaises(ValueError): tree_laws(self.cal[0]['hidden'],[],self.checkpoint)


if __name__=='__main__': unittest.main()
