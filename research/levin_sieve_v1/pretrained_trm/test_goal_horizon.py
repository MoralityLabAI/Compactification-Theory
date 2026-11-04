"""Exact-rational LSV-10 tests, including switched interventions."""
import random
import unittest
from fractions import Fraction as Q
from goal_horizon import (exact_goal_quotient,horizon_bases,coefficient_row,
    row_times,project,step,defect_transfer_bound,margin_preserved,system,rref)


class GoalHorizonTests(unittest.TestCase):
    def test_shift_register_cognitive_horizon(self):
        a=(((0,1,0,0),(0,0,1,0),(0,0,0,1),(0,0,0,0)),)
        c=((1,0,0,0),)
        self.assertEqual([len(b) for b in horizon_bases(a,c,5)],[1,2,3,4,4,4])
        q=exact_goal_quotient(a,c)
        self.assertEqual(q['minimal_dim'],4)

    def test_cancellation_with_arbitrarily_large_hidden_gain(self):
        a=(((10,10),(-10,-10)),)
        c=((1,1),)
        q=exact_goal_quotient(a,c)
        self.assertEqual(q['minimal_dim'],1)
        self.assertEqual(q['actions'],(((Q(0),),),))
        for x in ((-3,5),(10000,-9999),(0,0)):
            self.assertEqual(project(q['projection'],step(a[0],x)), (Q(0),))
        self.assertEqual(len(horizon_bases(a,c,8)[-1]),1)

    def test_fully_unobserved_unstable_dimensions(self):
        a=(((Q(1,2),0,0),(0,100,0),(0,0,-7)),)
        c=((1,0,0),)
        q=exact_goal_quotient(a,c)
        self.assertEqual(q['minimal_dim'],1)
        self.assertEqual(q['actions'],(((Q(1,2),),),))
        self.assertEqual([len(b) for b in horizon_bases(a,c,4)],[1]*5)

    def test_intervention_increases_relevant_state(self):
        a0=((Q(1,2),0),(0,100))
        a1=((0,1),(0,0))
        c=((1,0),)
        self.assertEqual(len(horizon_bases((a0,),c,2)[-1]),1)
        self.assertEqual([len(b) for b in horizon_bases((a0,a1),c,2)],[1,2,2])
        q=exact_goal_quotient((a0,a1),c)
        self.assertEqual(q['minimal_dim'],2)

    def test_affine_dynamics_commute_in_macrospace(self):
        modes=(((Q(1,2),0,0),(0,8,2),(0,1,7)),
               ((Q(1,3),0,0),(0,0,9),(0,-4,0)))
        c=((1,0,0),)
        q=exact_goal_quotient(modes,c)
        self.assertEqual(q['minimal_dim'],1)
        x=(Q(1,5),Q(-11),Q(22))
        for a,b,bias in zip(modes,q['actions'],((1,2,3),(-1,4,7))):
            left=project(q['projection'],step(a,x,bias))
            right=step(b,project(q['projection'],x),project(q['projection'],bias))
            self.assertEqual(left,right)

    def test_exact_randomized_companion_equations(self):
        rng=random.Random(20261104)
        for trial in range(80):
            n=2+trial%3
            modes=tuple(tuple(tuple(rng.randrange(-2,3) for _ in range(n)) for _ in range(n))
                        for _ in range(1+trial%2))
            C=tuple(tuple(rng.randrange(-2,3) for _ in range(n)) for _ in range(1+trial%3))
            q=exact_goal_quotient(modes,C)
            P=q['projection']
            for mode,B in zip(modes,q['actions']):
                for i,row in enumerate(P):
                    self.assertEqual(row_times(row,mode),
                        tuple(sum((B[i][j]*P[j][k] for j in range(len(P))),Q(0)) for k in range(n)))
            for i,row in enumerate(C):
                self.assertEqual(tuple(map(Q,row)),
                    tuple(sum((q['readout'][i][j]*P[j][k] for j in range(len(P))),Q(0)) for k in range(n)))
            self.assertEqual(len(P),len(horizon_bases(modes,C,n+2)[-1]))

    def test_minimality_on_finite_horizon(self):
        a=(((0,1,0),(0,0,1),(0,0,0)),)
        c=((1,0,0),)
        b=horizon_bases(a,c,2)
        # Any rank-1 projection that recovers both immediate and one-step goal
        # would have to contain independent rows e_1 and e_2.
        self.assertEqual(len(b[1]),2)
        with self.assertRaises(ValueError):coefficient_row((0,1,0),b[0])

    def test_zero_goal_readout(self):
        q=exact_goal_quotient((((1,0),(0,1)),),((0,0),))
        self.assertEqual(q['minimal_dim'],0)
        self.assertEqual(q['actions'],((),))
        self.assertEqual(q['readout'],((),))

    def test_exact_residual_transfer_tight_affine(self):
        rho=Q(3,5);epsilon=Q(1,20)
        x=Q(1,2)
        for k in range(8):
            self.assertEqual(x, defect_transfer_bound(Q(1,2),rho,epsilon,k))
            x=rho*x+epsilon
        self.assertTrue(margin_preserved(Q(1,2),2,Q(1,5)))
        self.assertFalse(margin_preserved(Q(2,5),2,Q(1,5)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):horizon_bases(((((0.1,),),),),((1,),),2)
        with self.assertRaises(ValueError):horizon_bases((),((1,),),1)
        with self.assertRaises(ValueError):horizon_bases(((((1,1),),),),((1,),),1)
        with self.assertRaises(ValueError):defect_transfer_bound(0,1,0,2)
        with self.assertRaises(ValueError):defect_transfer_bound(0,Q(1,2),-1,2)
        with self.assertRaises(ValueError):margin_preserved(1,-1,1)


# The nonlinear example uses an exactly preserved goal sum while its
# transverse cubic mode can expand without bound. These are finite rational
# checks of the algebraic identity, not model-checking of arbitrary programs.
class NonlinearGoalProjectionTests(unittest.TestCase):
    def test_cubic_transverse_explosion_with_exact_goal_recovery(self):
        def F(x):
            s=x[0]+x[1]
            t=Q(100)*(x[0]-x[1])**3
            return (s/4+t,s/4-t)
        for start in ((Q(2),Q(-1)),(Q(7,3),Q(1,5)),(Q(-10),Q(20))):
            x=start
            for k in range(4):
                self.assertEqual(sum(x),sum(start)/Q(2)**k)
                x=F(x)
        self.assertGreater(abs(F((Q(2),Q(-1)))[0]),1000)

    def test_transverse_leakage_breaks_global_macro_defect(self):
        def F_leak(x):
            s=x[0]+x[1]
            t=Q(100)*(x[0]-x[1])**3
            return (s/4+t+Q(1,100)*(x[0]-x[1]),s/4-t)
        # Projection pi(x)=x0+x1, macro G(z)=z/2.
        # For x=(T,-T), defect = 2T/100, unbounded as T grows.
        for T in (Q(1),Q(10),Q(1000)):
            x=(T,-T)
            self.assertEqual(sum(F_leak(x)),T/50)

if __name__=='__main__':unittest.main()
