"""Run: python -m unittest discover -s research/dynamical_closure -v"""
import unittest
from fractions import Fraction
from itertools import product
from closure import closure_witness, induced_dynamics, closure_risk, partitions

class FiniteClosureTests(unittest.TestCase):
    def test_factorization_theorem_exhaustive(self):
        for n in range(1,5):
            for f in product(range(n),repeat=n):
                for p in partitions(n):
                    labels = sorted(set(p))
                    factorizes = any(
                        all(g[p[x]] == p[f[x]] for x in range(n))
                        for outputs in product(labels,repeat=len(labels))
                        for g in [dict(zip(labels,outputs))]
                    )
                    witness = closure_witness(f,p)
                    self.assertEqual(witness is None,factorizes)
                    if witness is None:
                        g = induced_dynamics(f,p)
                        self.assertTrue(all(g[p[x]] == p[f[x]] for x in range(n)))
                    else:
                        x,y = witness
                        self.assertEqual(p[x],p[y])
                        self.assertNotEqual(p[f[x]],p[f[y]])

    def test_nonmonotonic_closure(self):
        f = (2,0,2)
        fine,mid,coarse = (0,1,2),(0,0,1),(0,0,0)
        self.assertIsNone(closure_witness(f,fine))
        self.assertIsNotNone(closure_witness(f,mid))
        self.assertIsNone(closure_witness(f,coarse))
        self.assertEqual(closure_risk(f,mid),Fraction(1,6))
        self.assertEqual(closure_risk(f,coarse),0)

    def test_zero_risk_equivalence(self):
        for n in range(1,5):
            for f in product(range(n),repeat=n):
                for p in partitions(n):
                    self.assertEqual(closure_risk(f,p)==0,closure_witness(f,p) is None)

    def test_empirical_support_matters(self):
        f,p = (2,0,2),(0,0,1)
        self.assertEqual(closure_risk(f,p,[1,0,1]),0)
        self.assertEqual(closure_risk(f,p,[1,1,1]),Fraction(1,6))

if __name__ == "__main__":
    unittest.main()
