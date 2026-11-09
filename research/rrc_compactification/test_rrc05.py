"""Exact-algebra, source-family and anti-overclaiming gates for RRC-05."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import unittest

try:
    from .rrc05 import (PREFIX,HORIZON,PERIOD_CAP,Program05,product_coefficients,
        log_derivative_coefficients,proper_divisors,divisor_residual,
        coefficient_residual,jacobi_exact,learned_iteration,feedforward_estimate,
        quantize,proposed_programs_from_euler,exact_program,candidate_from_estimate,
        roster,roster_digest,brute_single)
    from .rrc import euler_exponents
    from .bench05 import evaluate
except ImportError:
    from rrc05 import (PREFIX,HORIZON,PERIOD_CAP,Program05,product_coefficients,
        log_derivative_coefficients,proper_divisors,divisor_residual,
        coefficient_residual,jacobi_exact,learned_iteration,feedforward_estimate,
        quantize,proposed_programs_from_euler,exact_program,candidate_from_estimate,
        roster,roster_digest,brute_single)
    from rrc import euler_exponents
    from bench05 import evaluate

HERE=Path(__file__).resolve().parent


class FormalEulerTests(unittest.TestCase):
    def test_independent_small_enumeration_caps(self):
        for cap in (0,1,2):
            for period,residues in ((3,(0,1)),(5,(2,3)),(7,(1,4))):
                p=Program05(cap,period,residues)
                actual=product_coefficients(16,p)
                allowed=[k for k in range(1,17) if k%period in residues]
                self.assertEqual(actual,[brute_single(n,allowed,cap) for n in range(17)])

    def test_log_derivative_equals_euler_divisor_transform(self):
        for p in (Program05(0,13,(1,5,9)),Program05(1,7,(1,2,4)),
                  Program05(2,17,(0,3,12))):
            f=product_coefficients(PREFIX,p)
            b=log_derivative_coefficients(f)
            a=euler_exponents(f)
            self.assertEqual(divisor_residual(b,a),[0]*(PREFIX+1))
            self.assertEqual(coefficient_residual(f,a),[0]*(PREFIX+1))

    def test_finite_depth_nilpotency_and_sharp_witness(self):
        f=product_coefficients(PREFIX,Program05(0,1,(0,)))
        b=log_derivative_coefficients(f)
        a=euler_exponents(f)
        at6=jacobi_exact(b,6)
        at7=jacobi_exact(b,7)
        self.assertNotEqual(at6[64],Fraction(a[64]))
        self.assertEqual(at7,[Fraction(v) for v in a])
        for t in (7,8,9):
            self.assertEqual(jacobi_exact(b,t),at7)

    def test_algebraic_error_is_a_certificate_with_first_fault(self):
        p=Program05(1,13,(1,4,11))
        f=product_coefficients(PREFIX,p)
        b=log_derivative_coefficients(f)
        a=euler_exponents(f)
        changed=a.copy();changed[31]+=1
        residue=divisor_residual(b,changed)
        series=coefficient_residual(f,changed)
        self.assertEqual(next(i for i in range(1,65) if residue[i]),31)
        self.assertEqual(next(i for i in range(1,65) if series[i]),31)
        self.assertEqual(residue[:31],[0]*31)
        self.assertEqual(series[:31],[0]*31)

    def test_known_rr_cases_reconstruct_from_proposed_exponents(self):
        try:
            from .rrc import gap_coefficients
        except ImportError:
            from rrc import gap_coefficients
        for minpart,residues in ((1,(1,4)),(2,(2,3))):
            prefix=gap_coefficients(PREFIX,minimum_part=minpart)
            b=log_derivative_coefficients(prefix)
            cert=candidate_from_estimate(prefix,b,jacobi_exact(b,7))
            self.assertTrue(cert['exact_intermediate'])
            self.assertIsNone(cert['first_series_residual'])
            self.assertEqual(cert['program'],Program05(0,5,residues))

    def test_no_spurious_acceptance_from_corrupted_intermediate(self):
        p=Program05(0,13,(1,5,9))
        prefix=product_coefficients(PREFIX,p)
        b=log_derivative_coefficients(prefix)
        exact=euler_exponents(prefix)
        wrong=exact.copy();wrong[43]+=1
        cert=candidate_from_estimate(prefix,b,wrong)
        self.assertEqual(cert['first_divisor_residual'],43)
        self.assertEqual(cert['first_series_residual'],43)
        self.assertIsNone(cert['program'])

    def test_product_identity_alias_grammar_validation(self):
        # All distinct positive integers also count unrestricted odd parts.
        prefix=product_coefficients(PREFIX,Program05(1,1,(0,)))
        aliases=proposed_programs_from_euler(euler_exponents(prefix),prefix)
        self.assertIn(Program05(1,1,(0,)),aliases)
        self.assertIn(Program05(0,2,(1,)),aliases)
        self.assertTrue(all(product_coefficients(PREFIX,p)==prefix for p in aliases))

    def test_rounding_vs_exact_checker(self):
        self.assertEqual(quantize([0.,.49,-.51]),[0,0,-1])
        self.assertIsNone(quantize([0.,.49],tolerance=.1))
        self.assertIsNone(quantize([0.,float('inf')]))
        with self.assertRaises(ValueError):
            log_derivative_coefficients([2,3])
        with self.assertRaises(ValueError):
            jacobi_exact([0,1],-1)
        with self.assertRaises(ValueError):
            Program05(0,PERIOD_CAP+1,(1,))
        with self.assertRaises(ValueError):
            proposed_programs_from_euler([0,1],[1,1])

    def test_proper_divisor_graph_depth(self):
        P=proper_divisors(PREFIX)
        self.assertEqual(P[1],())
        self.assertEqual(P[64],(1,2,4,8,16,32))
        self.assertGreater(sum(map(len,P)),0)


class FamilyIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits=roster()

    def test_sizes_and_prefix_no_leakage(self):
        sizes={k:len(v) for k,v in self.splits.items()}
        self.assertEqual(sizes,dict(train=384,dev=96,unseen_modulus=120,
                                    local_gap=40,nonperiodic=15,
                                    out_of_grammar=18,collision=12))
        train={c.prefix_hash() for c in self.splits['train']}
        self.assertFalse(train.intersection(c.prefix_hash() for c in self.splits['dev']))
        self.assertFalse(train.intersection(c.prefix_hash() for c in self.splits['unseen_modulus']))
        self.assertEqual(len(train),len(self.splits['train']))
        self.assertEqual(len(roster(include_test=False)['local_gap']),0)

    def test_collisions_are_identical_inputs_and_future_disagree(self):
        pairs=self.splits['collision']
        for i in range(0,len(pairs),2):
            a,b=pairs[i:i+2]
            self.assertEqual(a.prefix,b.prefix)
            self.assertNotEqual(a.full,b.full)
            k=next(n for n in range(PREFIX+1,HORIZON+1) if a.full[n]!=b.full[n])
            self.assertEqual(k,(65,77,89,105,116,127)[i//2])
            self.assertEqual(a.collision_group,b.collision_group)
            self.assertNotEqual(a.name,b.name)

    def test_unseen_modulus_programs_are_exact_in_grammar(self):
        for c in self.splits['unseen_modulus']:
            p=exact_program(c.prefix)
            self.assertIsNotNone(p)
            self.assertEqual(product_coefficients(HORIZON,p),list(c.full))

    def test_full_receipt_replay_and_negative_controls(self):
        frozen=json.loads((HERE/'results'/'rrc05_pilot.json').read_text())
        result=evaluate()
        self.assertEqual(result,frozen)
        by=result['metrics']
        self.assertEqual(by['structural_jacobi']['7']['unseen_modulus']['intermediate_exact'],120)
        self.assertEqual(by['learned_recurrent']['7']['unseen_modulus']['correct_future'],120)
        self.assertEqual(by['exact_symbolic']['collision']['false_accepts'],6)
        self.assertEqual(by['learned_recurrent']['7']['collision']['false_accepts'],6)
        self.assertEqual(by['shuffled_labels']['unseen_modulus']['prefix_verified'],0)
        self.assertEqual(by['divisor_erased']['unseen_modulus']['prefix_verified'],1)
        self.assertEqual(by['one_pass_feedforward']['unseen_modulus']['prefix_verified'],5)
        self.assertEqual(by['learned_recurrent']['7']['out_of_grammar']['prefix_verified'],0)


if __name__=='__main__':
    unittest.main()
