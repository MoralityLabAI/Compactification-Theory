import random,unittest
from fractions import Fraction as F
from critic import values,bellman,norm,residual_bound,transport,sweeps,solve
class CriticTests(unittest.TestCase):
    def setUp(self):
        self.p=[[F(4,5),F(1,10)],[F(1,5),F(3,5)]];self.r=[F(1,10),F(2,5)];self.g=F(9,10)
    def test_exact_fixed_point(self):
        v=values(self.p,self.r,self.g);self.assertEqual(v,bellman(self.p,self.r,self.g,v))
    def test_resolvent_96_cases(self):
        rng=random.Random(904)
        def kernel():
            rows=[]
            for _ in range(3):
                xs=[rng.randrange(0,10) for _ in range(4)];den=sum(xs) or 1
                rows.append([F(x,den) for x in xs[:3]])
            return rows
        for _ in range(96):
            p=kernel();q=kernel();r=[F(rng.randrange(10),10) for _ in range(3)]
            s=[F(rng.randrange(10),10) for _ in range(3)]
            observed,predicted=transport(p,r,q,s,self.g);self.assertEqual(observed,predicted)
            vo=values(p,r,self.g);vn=values(q,s,self.g)
            self.assertLessEqual(norm([a-b for a,b in zip(vo,vn)]),residual_bound(q,s,self.g,vo))
    def test_exact_critic_zero_residual(self):
        self.assertEqual(residual_bound(self.p,self.r,self.g,values(self.p,self.r,self.g)),0)
    def test_warmup_contraction(self):
        v=values(self.p,self.r,self.g);initial=[F(0),F(0)]
        after=sweeps(self.p,self.r,self.g,initial,4)
        self.assertLessEqual(norm([a-b for a,b in zip(after,v)]),self.g**4*norm(v))
    def test_inheritance_can_be_worse_than_reset(self):
        p=[[F(1)]];vo=values(p,[F(1)],self.g);vn=values(p,[F(0)],self.g)
        self.assertGreater(norm([vo[0]-vn[0]]),norm(vn))
    def test_affine_reward_shift_for_stochastic_chain(self):
        p=[[F(3,4),F(1,4)],[F(1,3),F(2,3)]];r=[F(1,10),F(2,5)]
        a=values(p,r,self.g);b=values(p,[2*x+F(1,5) for x in r],self.g)
        self.assertEqual(b,[2*x+F(1,5)/(1-self.g) for x in a])
    def test_invalid_kernel_and_gamma(self):
        with self.assertRaises(ValueError):values([[F(2)]],[F(1)],self.g)
        with self.assertRaises(ValueError):values([[F(1)]],[F(1)],F(1))
        with self.assertRaises(ValueError):solve([[0]],[1])
class HistogramTests(unittest.TestCase):
    def test_histogram_beats_mean_substitution(self):
        from critic import expected_terminal_spr
        value=expected_terminal_spr(0,{0:F(1,2),100:F(1,2)},100)
        self.assertEqual(value,F(1,4));self.assertNotEqual(value,F(50,150))
    def test_divisor_reweighting(self):
        from critic import expected_terminal_spr
        pmf={0:F(1,2),100:F(1,2)}
        self.assertEqual(expected_terminal_spr(F(1,2),pmf,100),F(3,4))
        self.assertEqual(expected_terminal_spr(F(1,2),pmf,50),F(5,6))
    def test_invalid_histogram(self):
        from critic import expected_terminal_spr
        with self.assertRaises(ValueError):expected_terminal_spr(0,{0:F(1,2)},100)
if __name__=='__main__':unittest.main()

