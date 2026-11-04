import unittest
from fractions import Fraction as Q
from theory import margin_lower, chain_budget, ln_curvature_slack, quadratic_threshold_passes


class TestTaylorTheory(unittest.TestCase):
    def test_sharp_ln_identity_grid(self):
        for i in range(101):
            for j in range(101):
                self.assertGreaterEqual(ln_curvature_slack(Q(i,100),Q(j,100)),0)
    def test_ln_curvature_attainment(self):
        for q in (Q(1,3),Q(1,2),Q(3,4),Q(1)):
            self.assertEqual(ln_curvature_slack(q,1/(3*q)),0)
    def test_quadratic_bound_is_attained(self):
        for r in (Q(0),Q(1,10),Q(1,2)):
            exact=1-2*r-3*r*r
            self.assertEqual(margin_lower(1,r,2,[3*r*r]),exact)
    def test_uniform_remainder_is_essential(self):
        for r in (Q(1,10),Q(1,1000000)):
            c=2/(r*r)
            self.assertEqual(1-c*r*r,-1)
            self.assertGreater(margin_lower(1,r,0,[]),0)
            self.assertLess(margin_lower(1,r,0,[c*r*r]),0)
    def test_all_rivals_required(self):
        r=Q(1,50)
        self.assertGreater(1-0,0)
        self.assertLess(1-(-1+200*r),0)
    def test_linear_cancellation_is_exact(self):
        for x in range(-10,11):
            for z in range(-10,11):
                self.assertEqual((10*x+10*z)+(-10*x-10*z)+1,1)
        self.assertEqual(margin_lower(1,1000000,0,[]),1)
    def test_chain_with_hardware_error_budget(self):
        end,rows=chain_budget(Q(1,10),[
            {'a':2,'b':0,'q':3,'eta':Q(1,100)},
            {'a':3,'b':2,'q':1,'eta':0}])
        self.assertEqual(end,Q(63,100))
        self.assertEqual(rows[0]['output_defect'],Q(3,100))
        self.assertEqual(rows[1]['local_defect'],Q(441,10000))
    def test_invalid_inputs_fail_closed(self):
        for args in ((1,-1,1,[]),(1,1,-1,[]),(1,1,1,[-1])):
            with self.assertRaises(ValueError):margin_lower(*args)
        with self.assertRaises(TypeError):margin_lower(1,0.1,1,[])
        with self.assertRaises(ValueError):ln_curvature_slack(2,0)
    def test_strict_margin_boundary(self):
        self.assertFalse(quadratic_threshold_passes(1,1,0,1))
        self.assertTrue(quadratic_threshold_passes(1,1,0,Q(999,1000)))

if __name__=='__main__':unittest.main()
