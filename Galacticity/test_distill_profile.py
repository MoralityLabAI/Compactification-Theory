import unittest
from fractions import Fraction
from distill_profile import profile

def row(cap, candidate, utility="3/1", risk="1/10"):
    literal={"work":"10/1","utility":"3/1","false_recommendations":"1/10"}
    other={"work":candidate,"utility":utility,"false_recommendations":risk}
    return {"root_id":"test","cap":cap,"split":"ordinary","policies":{
        "literal_tail":literal,"literal_broad":literal,
        "flags_tail":other,"flags_broad":other,
        "marginal_tail":other,"marginal_broad":other}}

class Tests(unittest.TestCase):
    def test_first_hit_not_tail(self):
        rows=profile([row(8,"5/1"),row(16,"6/1")],Fraction(2))
        self.assertEqual(len(rows),4)
        for r in rows:
            self.assertEqual(r["observed_first_hit"],8)
            self.assertFalse(r["stable_tail_certified"])
    def test_utility_risk_gate(self):
        for a in [row(8,"1/1",utility="2/1"),row(8,"1/1",risk="2/10")]:
            self.assertTrue(all(r["no_hit_on_grid"] for r in profile([a],Fraction(2))))
    def test_duplicate(self):
        with self.assertRaises(ValueError):
            profile([row(8,"5/1"),row(8,"5/1")],Fraction(2))
    def test_invalid_q(self):
        with self.assertRaises(ValueError):
            profile([],Fraction(1))

if __name__=="__main__":
    unittest.main()
