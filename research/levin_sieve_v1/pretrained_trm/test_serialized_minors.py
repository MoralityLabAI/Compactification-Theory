"""Pure-standard-library independent checks for serialized proof witnesses."""
import unittest
from fractions import Fraction
from verify_serialized_minors import parse_ball,independent_det,verify_record,verify_receipt


class SerialProofTests(unittest.TestCase):
    def witness(self):
        entries=[['1' if i==j else '0' for j in range(4)] for i in range(4)]
        return {'case_index':0,'columns':[0,1,2,3],
                'verified_jacobian_entry_enclosures':entries,
                'observed_autograd_binary64_hex':[[float(int(i==j)).hex() for j in range(4)] for i in range(4)],
                'previous_required_entry_error_lt':0.1}

    def test_decimal_and_ball_parsing(self):
        self.assertEqual(parse_ball('[2.5 +/- 0.1]'),(Fraction(12,5),Fraction(13,5)))
        self.assertEqual(parse_ball('[+/- 1e-4]'),(-Fraction(1,10000),Fraction(1,10000)))
        self.assertEqual(parse_ball('-0.125'),(-Fraction(1,8),-Fraction(1,8)))
        for bad in ('[1 +/- -1]','foo','[nan +/- 1]',123):
            with self.assertRaises(ValueError):parse_ball(bad)

    def test_independent_exact_rational_determinant(self):
        ident=[[(Fraction(int(i==j)),Fraction(int(i==j))) for j in range(4)] for i in range(4)]
        self.assertEqual(independent_det(ident),(Fraction(1),Fraction(1)))

    def test_rank_and_autograd_budget_pass(self):
        r=verify_record(self.witness())
        self.assertTrue(r['rank_four_proved'])
        self.assertTrue(r['previous_lsv12_autograd_error_budget_proved'])
        rec={'schema':'LSV-13/arb-validated-goal-minors-v1','cases_checked':1,
             'cases':[self.witness()]}
        self.assertEqual(verify_receipt(rec)['independent_autograd_budget_count'],1)

    def test_tampered_det_fails(self):
        c=self.witness();c['verified_jacobian_entry_enclosures'][0][0]='0'
        with self.assertRaises(AssertionError):verify_record(c)

    def test_tampered_observed_derivative_fails(self):
        c=self.witness();c['observed_autograd_binary64_hex'][2][2]=float(2).hex()
        with self.assertRaises(AssertionError):verify_record(c)

    def test_missing_observed_witness_rejected(self):
        c=self.witness();del c['observed_autograd_binary64_hex']
        with self.assertRaises(AssertionError):verify_record(c)
        self.assertTrue(verify_record(c,require_autograd=False)['rank_four_proved'])

if __name__=='__main__':unittest.main()
