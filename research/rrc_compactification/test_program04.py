"""RRC-04 mathematical oracles, controls, split isolation and frozen replay."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import math
import random
import unittest

try:
    from .program04 import (Program,PREFIX,HORIZON,MAX_PERIOD,CAPS,SEED,
                            TRAIN_PERIODS,TEST_PERIODS,coefficients,
                            arbitrary_part_coefficients,late_defect,
                            recover_allowed,fits_period,synthesize,synthesize_all,
                            complete_skeleton,roster,brute_single,verify_prefix,
                            simple_features,roster_digest)
    from .student04 import ProgramStudent,RECUR_PARAMS,FF_PARAMS
    from .rrc import gap_coefficients,euler_exponents
except ImportError:
    from program04 import (Program,PREFIX,HORIZON,MAX_PERIOD,CAPS,SEED,
                           TRAIN_PERIODS,TEST_PERIODS,coefficients,
                           arbitrary_part_coefficients,late_defect,
                           recover_allowed,fits_period,synthesize,synthesize_all,
                           complete_skeleton,roster,brute_single,verify_prefix,
                           simple_features,roster_digest)
    from student04 import ProgramStudent,RECUR_PARAMS,FF_PARAMS
    from rrc import gap_coefficients,euler_exponents

HERE=Path(__file__).resolve().parent

class Program04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roster=roster(train_size=120,dev_size=40,unseen_size=30)

    def test_direct_product_vs_recursive_enumeration(self):
        for cap in CAPS:
            for period in (1,2,4,7):
                p=Program(cap,period,tuple(r for r in range(period) if r%3!=0) or (0,))
                result=coefficients(17,p)
                allowed=[n for n in range(1,18) if n%period in p.residues]
                for n in range(18):
                    self.assertEqual(result[n],brute_single(n,allowed,cap),(p,n))
                    self.assertEqual(result[n],arbitrary_part_coefficients(17,allowed,cap)[n])

    def test_unique_euler_recovery_for_each_cap(self):
        for cap in CAPS:
            for period in (1,3,5,7,11,12):
                mask=tuple(r for r in range(period) if r%4<2)
                p=Program(cap,period,mask)
                f=coefficients(PREFIX,p)
                allowed=recover_allowed(euler_exponents(f),cap)
                self.assertEqual(allowed[1:],[int(n%period in mask)
                                                    for n in range(1,PREFIX+1)])
                self.assertIn(p,synthesize_all(f))
                self.assertTrue(verify_prefix(f,p))

    def test_infinitely_equivalent_distinct_and_odd(self):
        distinct=Program(1,1,(0,))
        odd=Program(0,2,(1,))
        self.assertEqual(coefficients(HORIZON,distinct),coefficients(HORIZON,odd))
        all_candidates=synthesize_all(coefficients(PREFIX,distinct))
        self.assertIn(distinct,all_candidates)
        self.assertIn(odd,all_candidates)
        self.assertEqual(synthesize(coefficients(PREFIX,distinct)),distinct)

    def test_unseen_local_gap_rr_positive_controls(self):
        for minpart,mask in ((1,(1,4)),(2,(2,3))):
            f=gap_coefficients(HORIZON,gap=2,minimum_part=minpart)
            p=Program(0,5,mask)
            self.assertEqual(coefficients(HORIZON,p),f)
            self.assertIn(p,synthesize_all(f[:PREFIX+1]))
        for minpart in range(1,5):
            self.assertIsNone(synthesize(gap_coefficients(PREFIX,gap=3,
                                                          minimum_part=minpart)))

    def test_nonperiodic_rejection_with_registered_cap(self):
        for case in self.roster['nonperiodic']:
            self.assertIsNone(synthesize(case.prefix),case.name)
        for case in self.roster['local_gap']:
            if case.name in ('gap1-min1','gap2-min1','gap2-min2'):
                self.assertIsNotNone(synthesize(case.prefix))
            else:
                self.assertIsNone(synthesize(case.prefix),case.name)

    def test_collision_certificate_impossibility(self):
        for i in range(8):
            base,mutant=self.roster['collision'][2*i:2*i+2]
            self.assertEqual(base.prefix,mutant.prefix)
            self.assertNotEqual(base.full,mutant.full)
            self.assertEqual(synthesize(base.prefix),synthesize(mutant.prefix))
            fitted=synthesize(base.prefix)
            self.assertIsNotNone(fitted)
            p=coefficients(HORIZON,fitted)
            self.assertEqual(p,list(base.full))
            self.assertNotEqual(p,list(mutant.full))
            self.assertEqual(base.full[:PREFIX+1],mutant.full[:PREFIX+1])
        self.assertEqual(len({c.signature() for c in self.roster['collision']}),8)

    def test_heldout_beyond_train_horizon(self):
        q=Program(0,5,(1,4))
        true=coefficients(HORIZON,q)
        corrupted=late_defect(true,73)
        self.assertEqual(true[:PREFIX+1],corrupted[:PREFIX+1])
        self.assertNotEqual(true[73],corrupted[73])
        self.assertEqual(synthesize(true[:PREFIX+1]),synthesize(corrupted[:PREFIX+1]))
        self.assertEqual(true[73]+1,corrupted[73])
        self.assertIsNone(synthesize([1]+[5]*PREFIX))

    def test_hybrid_mask_completer(self):
        f=coefficients(PREFIX,Program(2,7,(1,3,5)))
        self.assertEqual(complete_skeleton(f,2,7),Program(2,7,(1,3,5)))
        self.assertIsNone(complete_skeleton(f,0,7))
        self.assertIsNone(complete_skeleton(f,2,2))
        self.assertTrue(len(synthesize_all(f))>=1)

    def test_invalid_and_short_prefix_rejected(self):
        with self.assertRaises(ValueError):Program(3,4,(1,))
        with self.assertRaises(ValueError):Program(0,0,())
        with self.assertRaises(ValueError):Program(0,2,(2,))
        with self.assertRaises(ValueError):Program(0,2,(1,1))
        with self.assertRaises(ValueError):coefficients(-1,Program(0,2,(1,)))
        with self.assertRaises(ValueError):synthesize_all([1]+[0]*12)
        with self.assertRaises(ValueError):simple_features([1]*12)
        with self.assertRaises(ValueError):roster(train_size=0)

    def test_generator_disjoint_and_modulus_disjoint(self):
        self.assertEqual(set(TRAIN_PERIODS).intersection(TEST_PERIODS),set())
        self.assertEqual({c.family for c in self.roster['train']},
                         {'product_cap0','product_cap1','product_cap2'})
        self.assertTrue(all(c.family=='local_gap' for c in self.roster['local_gap']))
        train_signatures={c.signature() for c in self.roster['train']}
        for k in ('dev','unseen_modulus'):
            self.assertFalse(train_signatures.intersection(c.signature() for c in self.roster[k]))
        self.assertEqual(len(self.roster['local_gap']),32)
        self.assertEqual(len(self.roster['nonperiodic']),15)
        self.assertEqual(len(self.roster['collision']),16)
        self.assertEqual(len(roster(train_size=120,dev_size=40,unseen_size=30,
                                   include_test=False)['local_gap']),0)

    def test_student_checkpoint_and_inputs(self):
        self.assertLess(abs(RECUR_PARAMS-FF_PARAMS)/FF_PARAMS,.01)
        data=roster(train_size=10,dev_size=5,unseen_size=5)
        case=data['dev'][0]
        inputs=simple_features(case.prefix)
        self.assertEqual(len(inputs),65)
        self.assertEqual(len(inputs[0]),3)
        # Checkpoint matrices contain only declared weights and typed shapes.
        for kind in ('recurrent','feedforward','shuffled_recurrent'):
            model=ProgramStudent.load(HERE/'results'/f'rrc04_{kind}.json')
            self.assertEqual(model.parameters,FF_PARAMS if kind=='feedforward' else RECUR_PARAMS)
            self.assertEqual(len(model.logits(case.prefix,2)),27)
            self.assertTrue(verify_prefix(case.prefix,synthesize(case.prefix)))
            prediction=model.predict(case.prefix)
            self.assertIsInstance(prediction,Program)

    def test_source_bound_replay_and_collision_gate(self):
        from bench04 import sha_file
        path=HERE/'results'/'rrc04_pilot.json'
        self.assertTrue(path.is_file())
        saved=json.loads(path.read_text())
        self.assertEqual(saved['schema'],'rrc04.pilot.v1')
        for key,signature in saved['source_sha256'].items():
            self.assertEqual(sha_file(HERE/key),signature,key)
        for name,signature in saved['checkpoints_sha256'].items():
            self.assertEqual(sha_file(HERE/'results'/f'rrc04_{name}.json'),signature,name)
        self.assertEqual(saved['collision_bound']['pairs'],8)
        self.assertEqual(saved['sizes']['train'],1500)
        self.assertEqual(saved['sizes']['unseen_modulus'],108)

if __name__=='__main__':unittest.main()
