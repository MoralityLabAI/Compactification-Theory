"""OIC-19 receipt consistency and deliberate mutation detection."""
import copy
import unittest
from fractions import Fraction
from oic19_verify import load_records,check_record,fault_audit,verify,require
from oic19_identifiability import first_n_pulse_probability


class OIC19VerifierTests(unittest.TestCase):
    def test_complete_40_fit_receipt(self):
        m,records=load_records()
        self.assertEqual(len(records),40)
        self.assertEqual(len(m['aggregates']),8)
        self.assertEqual(m['lambda_validation']['chosen_lambda'],.02)

    def test_every_record_has_valid_budget(self):
        _,records=load_records()
        for r in records:self.assertTrue(check_record(r))

    def test_fault_panel_exact_denominators(self):
        self.assertEqual(fault_audit(),10*2*sum(2*n*(n+3) for n in (8,24,64)))

    def test_mutated_family_accuracy_detected(self):
        _,records=load_records()
        bad=copy.deepcopy(records[0]);bad['panels']['8']['full_active']['accuracy']=0.123
        with self.assertRaises(ValueError):check_record(bad)

    def test_mutated_kernel_mass_detected(self):
        _,records=load_records()
        bad=copy.deepcopy(records[0]);bad['kernels']['rotating_step_mass']=.123
        with self.assertRaises(ValueError):check_record(bad)

    def test_testcase_gcd_invariance(self):
        self.assertEqual(first_n_pulse_probability(12,4),Fraction(1,4))
        self.assertEqual(first_n_pulse_probability(12,5),1)


if __name__=='__main__':unittest.main()
