import unittest
from research.shaft_compactification.comp23_audit import audit

class Comp23AuditTests(unittest.TestCase):
    def test_frozen_result(self):
        r=audit()
        self.assertEqual(r["status"],"verified")
        self.assertEqual(r["heldout_model_count"],4)
    def test_registered_primary_gain(self):
        self.assertAlmostEqual(audit()["primary_difference"],.376418)
    def test_direct_exact_control(self):
        self.assertEqual(audit()["heldout_depth40"]["direct36"][1],0)
    def test_factorized_exact(self):
        self.assertAlmostEqual(audit()["heldout_depth40"]["factorized900"][1],.414551)
    def test_information_matched_parameter_count(self):
        from research.shaft_compactification.comp23_audit import DEFAULT
        import json
        r=json.load(open(DEFAULT))
        self.assertEqual(r["decoders"]["factorized900"]["parameters"],r["decoders"]["direct36"]["parameters"])

if __name__=="__main__":
    unittest.main()
