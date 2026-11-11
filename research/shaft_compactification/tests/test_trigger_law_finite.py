import unittest
from research.shaft_compactification.trigger_law_finite import (
    FAMILIES,PERMS,compatible_rules,identify_from_exact,
    one_signature,reference,rename_rule,two_signature
)

class TriggerLawFiniteTests(unittest.TestCase):
    def test_exact_recovery(self):
        for rule in FAMILIES.values():
            for b in range(64):
                for p in range(6):
                    bits=identify_from_exact(b,p,two_signature(b,p,rule))
                    self.assertEqual(bits.index(1),rule[p])
                    self.assertEqual(sum(bits),1)
    def test_visible_previous_identifies_law(self):
        for rule in FAMILIES.values():
            self.assertEqual(compatible_rules(dict(enumerate(rule))),[rule])
    def test_two_missing_labels_leave_two_laws(self):
        for rule in FAMILIES.values():
            self.assertEqual(len(compatible_rules(dict(enumerate(rule[:4])))),2)
    def test_one_missing_label_forced_by_matching(self):
        for rule in FAMILIES.values():
            self.assertEqual(len(compatible_rules(dict(enumerate(rule[:5])))),1)
    def test_unlabeled_gauge_720(self):
        for rule in FAMILIES.values():
            self.assertEqual(len({rename_rule(rule,p) for p in PERMS}),720)
    def test_invalid_signature(self):
        with self.assertRaises(ValueError):
            two_signature(64,0,FAMILIES["repeat_toggle"])
    def test_full_reference(self):
        x=reference()
        self.assertEqual(x["xor_positions_checked"],27648)
        self.assertEqual(x["permutation_rules"],720)
        self.assertEqual(x["visible_previous_only_four_rule_candidates"],2)

if __name__=="__main__":
    unittest.main()
