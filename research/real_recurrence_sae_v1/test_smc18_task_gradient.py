"""SMC-18 isolated CPU tests independent of original trained checkpoints."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
import smc18_task_gradient as t


class LinearTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Identity()
        self.head=nn.Linear(48,10,dtype=torch.float64,bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[1,0]=2.


class QuadraticTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Identity()
    def head(self,h):
        out=torch.zeros((*h.shape[:2],10),dtype=h.dtype)
        # Use a non-inplace concatenation that retains autograd.
        a=torch.zeros((*h.shape[:-1],1),dtype=h.dtype)
        b=h[...,0:1]**2
        c=torch.zeros((*h.shape[:-1],8),dtype=h.dtype)
        return torch.cat((a,b,c),dim=-1)


class SMC18TaskGradientTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_goal_diagonal_task_gradient_agrees_with_linear_teacher(self):
        model=LinearTeacher().eval()
        stage=torch.zeros(81,48,dtype=torch.float32)
        stage[:,0]=1.
        clean=model.head(stage.double())
        pos=torch.arange(8)
        truth=torch.ones(8,dtype=torch.long)
        mask=torch.tensor([True,False,True,False,True,False,False,True])
        g=t.task_gradients(model,stage,clean,truth,pos,mask)
        self.assertEqual(tuple(g.shape),(8,48))
        self.assertTrue(bool((g[~mask]==0).all()))
        self.assertTrue(bool(torch.allclose(g[mask,0],torch.full((4,),2.,dtype=torch.float64))))
        self.assertTrue(bool((g[mask,1:]==0).all()))

    def test_quadratic_directional_curvature_and_local_gradient(self):
        model=QuadraticTeacher().eval()
        stage=torch.zeros(81,48)
        stage[:,0]=1.
        clean=model.head(stage.double())
        pos=torch.arange(8)
        truth=torch.ones(8,dtype=torch.long)
        mask=torch.tensor([True,False,True,False,False,False,False,False])
        k=torch.zeros(8,48,dtype=torch.float64)
        s=torch.zeros_like(k);k[:,0]=1.;s[:,1]=1.
        g=t.task_gradients(model,stage,clean,truth,pos,mask)
        self.assertAlmostEqual(float(g[0,0]),2,places=11)
        qk,qs=t.curvature_probe(model,stage,clean,truth,pos,mask,k,s)
        self.assertAlmostEqual(float(qk[0]),2,places=9)
        self.assertAlmostEqual(float(qs[0]),0,places=9)
        self.assertTrue(bool((qk[~mask]==0).all()))

    def test_full_rank_inactive_dual_reconstructs_task_gradient(self):
        cov=torch.zeros(96,48,dtype=torch.float64)
        for col in range(48):cov[4+col,col]=1.
        g=torch.zeros(48,dtype=torch.float64);g[0]=2.;g[1]=-3.
        u=torch.zeros(48,dtype=torch.float64);u[0]=4.;u[1]=1.
        out=t.inactive_gradient_dual(cov,[0,1,2,3],g,u)
        self.assertAlmostEqual(out['task_gradient_dual_dot'],5.)
        self.assertAlmostEqual(out['task_gradient_inactive_cs_ceiling'],float(g.norm()*u.norm()))
        self.assertAlmostEqual(out['minimum_inactive_singular_value'],1.)
        self.assertAlmostEqual(out['reconstruction_l2_error'],0.)

    def test_inactive_rank_defect_is_rejected(self):
        cov=torch.zeros(96,48,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'full column rank'):
            t.inactive_gradient_dual(cov,[0,1,2,3],torch.zeros(48,dtype=torch.float64),
                                     torch.zeros(48,dtype=torch.float64))

    def example_rows(self):
        rows=[]
        for p in range(64):
            eligible=[p<24]+[False]*7
            item={'finite_kill_minus_sham':2.,'clean_linear_prediction':1.8}
            rows.append({'eligible_mask':eligible,
                         'metrics':[dict(item) if p<24 else None]+[None]*7})
        return rows

    def test_registered_coverage_gradient_agreement_and_ratio_gates(self):
        rows=self.example_rows()
        out=t.score_family(rows,seed=202618130)
        self.assertTrue(out['gate'])
        self.assertEqual(out['eligible_goals'],24)
        self.assertEqual(out['eligible_puzzles'],24)
        self.assertEqual(out['comparable_goals'],24)
        self.assertAlmostEqual(out['relative_absolute_remainder'],.1)
        self.assertEqual(out['puzzle_bootstrap95_sign_agreement_minus_half'],[.5,.5])
        self.assertEqual(out['puzzle_bootstrap95_finite'],[2.,2.])

    def test_retains_negative_finite_and_linear_prediction_gates(self):
        rows=self.example_rows()
        for row in rows:
            if row['metrics'][0] is not None:
                row['metrics'][0]['clean_linear_prediction']=-1.
        out=t.score_family(rows,seed=202618130)
        self.assertFalse(out['registered_gradient_sign_gate'])
        self.assertFalse(out['registered_linear_remainder_gate'])
        self.assertFalse(out['gate'])
        for row in rows:
            if row['metrics'][0] is not None:
                row['metrics'][0]['finite_kill_minus_sham']=-2.
        self.assertFalse(t.score_family(rows,seed=202618130)['registered_coverage_and_finite_gate'])

    def test_invalid_geometry_is_guarded(self):
        with self.assertRaises(ValueError):t.fresh_rosters({},limit=5)
        model=LinearTeacher()
        with self.assertRaises(ValueError):
            t.task_gradients(model,torch.zeros(81,48),torch.zeros(81,10),
                             torch.ones(5,dtype=torch.long),torch.arange(8),
                             torch.ones(8,dtype=torch.bool))
        with self.assertRaises(ValueError):
            t.competitors(torch.zeros(8,9),torch.ones(8,dtype=torch.long))


if __name__=="__main__":unittest.main()
