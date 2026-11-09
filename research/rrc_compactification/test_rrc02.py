"""RRC-02: deterministic split, information limit, exact verifier and model tests."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

try:
    from .rrc import ResidueRule, euler_exponents, first_mismatch, infer_periodic_rule, product_coefficients
    from .rrc02 import (Case, TRAIN_DEGREE, EVAL_DEGREE, MODULUS_CAP, HERE, DEPTHS,
                        candidate_rows, certificate, coefficients_from_exponents,
                        distinct_coefficients, exponent_mask, infer_student,
                        make_cases, receipt, training_rows)
    from .student02 import make_features, ResidueStudent
except ImportError:
    from rrc import ResidueRule, euler_exponents, first_mismatch, infer_periodic_rule, product_coefficients
    from rrc02 import (Case, TRAIN_DEGREE, EVAL_DEGREE, MODULUS_CAP, HERE, DEPTHS,
                       candidate_rows, certificate, coefficients_from_exponents,
                       distinct_coefficients, exponent_mask, infer_student,
                       make_cases, receipt, training_rows)
    from student02 import make_features, ResidueStudent


class RRC02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = make_cases()
        cls.by_id = {c.case_id:c for c in cls.cases}

    def test_roster_is_deterministic_family_disjoint(self):
        self.assertEqual((sum(c.split == 'train' for c in self.cases),
                          sum(c.split == 'dev' for c in self.cases),
                          sum(c.split == 'test' for c in self.cases)), (224,8,44))
        self.assertEqual(len(self.by_id),len(self.cases))
        self.assertEqual(self.cases, make_cases())
        train_families = {c.family for c in self.cases if c.split == 'train'}
        self.assertEqual(train_families,
                         {'periodic_product','finite_defect_product','colored_product'})
        self.assertFalse(any(c.family.startswith('local_') for c in self.cases if c.split != 'test'))
        self.assertFalse(any('unseen_modulus' in c.family for c in self.cases if c.split == 'train'))
        self.assertEqual({c.family for c in self.cases if c.split=='dev'}, {'distinct_parts'})

    def test_train_rows_exact_split_and_recovery_targets(self):
        features, labels, membership = training_rows(self.cases)
        self.assertEqual(len(features), 17472)
        self.assertEqual(len(labels),len(features))
        self.assertTrue(set(labels) == {0,1,2})
        self.assertTrue(set(membership) <= {c.case_id for c in self.cases if c.split=='train'})
        self.assertTrue(all(len(x)==6 for x in features))
        self.assertTrue(all(x[4] > 0 for x in features))

    def test_independent_euler_reconstruction(self):
        for modulus,residues in ((2,(1,)),(5,(1,4)),(7,(0,3,6)),(11,(2,7))):
            a=exponent_mask(modulus,residues)
            f=coefficients_from_exponents(a)
            self.assertEqual(f,product_coefficients(EVAL_DEGREE,ResidueRule(modulus,residues)))
            self.assertEqual(euler_exponents(f),a)

    def test_distinct_parts_independent_dp(self):
        self.assertEqual(distinct_coefficients(8),[1,1,1,2,2,3,4,5,6])
        self.assertEqual(infer_periodic_rule(distinct_coefficients(64),modulus_cap=12),
                         ResidueRule(2,(1,)))

    def test_two_local_rogers_ramanujan_rules_are_held_out(self):
        self.assertEqual(infer_periodic_rule(self.by_id['test_gap2_s1'].prefix),
                         ResidueRule(5,(1,4)))
        self.assertEqual(infer_periodic_rule(self.by_id['test_gap2_s2'].prefix),
                         ResidueRule(5,(2,3)))
        for suffix in ('test_gap3_s1','test_gap4_s1'):
            self.assertIsNone(infer_periodic_rule(self.by_id[suffix].prefix))

    def test_finite_prefix_collisions_are_unidentifiable(self):
        for i,k in enumerate((65,73,96,112)):
            base=self.by_id[f'test_collision_base_{i}']
            mutant=self.by_id[f'test_collision_mutant_{i}']
            self.assertEqual(base.prefix,mutant.prefix)
            self.assertEqual(first_mismatch(base.coefficients,mutant.coefficients),k)
            self.assertEqual(mutant.coefficients[k]-base.coefficients[k],1)
            self.assertEqual(candidate_rows(base.prefix),candidate_rows(mutant.prefix))
            self.assertEqual(infer_periodic_rule(base.prefix),infer_periodic_rule(mutant.prefix))

    def test_exact_certificate_rejects_wrong_program_and_mutation(self):
        prefix=self.by_id['test_gap2_s1'].prefix
        self.assertTrue(certificate(prefix,ResidueRule(5,(1,4)))['valid'])
        self.assertFalse(certificate(prefix,ResidueRule(5,(0,2)))['valid'])
        self.assertFalse(certificate(prefix,ResidueRule(13,(1,)))['valid'])
        self.assertFalse(certificate(prefix,ResidueRule(5,(4,4)))['valid'])
        self.assertFalse(certificate(prefix,None)['valid'])
        mutated=list(prefix);mutated[61]+=1
        self.assertFalse(certificate(tuple(mutated),ResidueRule(5,(1,4)))['valid'])
        with self.assertRaises(ValueError):
            candidate_rows(tuple(mutated[:-1]))
        with self.assertRaises(ValueError):
            make_features([1])

    def test_portable_models_parameter_counts_and_exact_rules(self):
        recurrent=ResidueStudent.load(HERE/'results/rrc02_recurrent.json')
        feedforward=ResidueStudent.load(HERE/'results/rrc02_feedforward.json')
        shuffled=ResidueStudent.load(HERE/'results/rrc02_shuffled.json')
        self.assertEqual((recurrent.parameter_count, feedforward.parameter_count,
                          shuffled.parameter_count),(363,371,363))
        prefix=self.by_id['test_gap2_s1'].prefix
        for depth in DEPTHS:
            found=infer_student(prefix,recurrent,depth)
            self.assertEqual(found['rule'],ResidueRule(5,(1,4)))
            self.assertTrue(certificate(prefix,found['rule'])['valid'])
        self.assertEqual(infer_student(prefix,feedforward,0)['rule'],ResidueRule(5,(1,4)))
        self.assertIsNone(infer_student(prefix,shuffled,0)['rule'])
        with self.assertRaises(ValueError):
            feedforward.logits((0.,)*6,depth=1)
        with self.assertRaises(ValueError):
            recurrent.logits((0.,)*6,depth=3)

    def test_student_inference_cannot_read_evaluation_only_coefficients(self):
        recurrent=ResidueStudent.load(HERE/'results/rrc02_recurrent.json')
        case=self.by_id['test_gap2_s2']
        corrupt=list(case.coefficients)
        corrupt[TRAIN_DEGREE+1] += 1234567
        changed=Case('same-prefix', 'test', 'mutated_eval', tuple(corrupt))
        self.assertEqual(infer_student(case.prefix,recurrent,8),
                         infer_student(changed.prefix,recurrent,8))

    def test_receipt_source_bound_and_exact_metrics(self):
        checkpoints={k: HERE/f'results/rrc02_{k}.json' for k in ('recurrent','feedforward','shuffled')}
        measured=receipt(checkpoints)
        canonical=json.loads((HERE/'results/rrc02_pilot.json').read_text())
        self.assertEqual(measured, canonical)
        self.assertEqual(measured['evidence']['counts'],{'train':224,'dev':8,'test':44})
        entries=measured['evidence']['cases']
        main=[r for r in entries if r['split']=='test' and r['model']=='recurrent' and r['depth']==8]
        self.assertEqual(len(main),44)
        self.assertEqual(sum(r['same_as_exact_baseline'] for r in main),44)
        self.assertEqual(sum(r['heldout_exact'] for r in main),30)
        self.assertEqual(sum(r['accepted'] and not r['heldout_exact'] for r in main),4)
        for r in main:
            self.assertEqual(bool(r['collision_of']), r['accepted'] and not r['heldout_exact'])


if __name__ == '__main__':
    unittest.main()
