"""CGC-02 source/engineering checks, excluded seed only."""
from __future__ import annotations

import unittest
import torch

from research.causal_gauge_compactification.cgc02.experiment import (
    CAL_SIZE, D, FRESH_ROOTS, GRAM_WEIGHTS, KIDX, RANK,
    RecurrentTeacher, assay_seed, fit_factors, matched_source_pairs,
    physical_weight_basis, protected_coordinates, protected_project,
    run, score_edit, source_batch, subspaces, tangent_matched_sham,
    task_jacobian, top_task_basis, train_teacher,
)


class CGC02SourceEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_protected_full_softmax_exact_for_dark_edits(self):
        state=torch.randn((64,D),generator=torch.Generator().manual_seed(100))
        raw=torch.randn((64,D),generator=torch.Generator().manual_seed(101))*.5
        dark=protected_project(raw)
        left=RecurrentTeacher.protected(state)
        right=RecurrentTeacher.protected(state+dark)
        self.assertTrue(torch.equal(left,right))
        self.assertEqual(len(KIDX),D-2)
        self.assertLess(float((protected_coordinates().T@protected_coordinates()-
                               torch.eye(len(KIDX))).abs().max()),1e-7)

    def test_matched_pair_changes_just_queried_private_value(self):
        h,x,donor,target,dt=matched_source_pairs(3519,256)
        self.assertTrue(torch.equal(x[:,2:],donor[:,2:]))
        self.assertTrue((((x-donor).abs()>0).sum(1)==1).all())
        self.assertTrue(torch.equal(RecurrentTeacher.protected(h),
                                    RecurrentTeacher.protected(h.clone())))
        self.assertTrue(((target-dt).abs().amax(1)>0).all())

    def test_teacher_identical_shared_step_all_depths(self):
        teacher=RecurrentTeacher(901)
        h,x,_=source_batch(1307,32)
        state=teacher.step(h,x)
        zero=torch.zeros_like(x)
        explicit=state
        for _ in range(4):
            explicit=teacher.step(explicit,zero)
        implicit=teacher.no_op(state,4)
        self.assertTrue(torch.allclose(explicit,implicit,atol=1e-7))
        self.assertEqual(tuple(teacher.task(implicit).shape),(32,2))
        self.assertTrue(torch.isfinite(implicit).all())

    def test_ood_correlations_and_different_family(self):
        _,x,_=source_batch(4081,1024,"train")
        _,y,_=source_batch(4081,1024,"heldout_anticorrelated")
        self.assertFalse(torch.equal(x,y))
        corr=torch.corrcoef(y[:,:2].T)[0,1]
        self.assertLess(float(corr),-.85)

    def test_frozen_calibration_contains_no_unseen_depth(self):
        self.assertEqual(tuple(GRAM_WEIGHTS.keys()),(2,4,6))
        self.assertEqual(CAL_SIZE,256)
        self.assertEqual(RANK,2)
        self.assertEqual(FRESH_ROOTS,(2601,2607,2617,2621))

    def test_task_jacobian_is_true_recurrence_derivative(self):
        teacher=RecurrentTeacher(901)
        h,x,_=source_batch(1237,8)
        s=teacher.step(h,x).detach()
        j=task_jacobian(teacher,s,4)
        v=torch.randn((8,len(KIDX)),generator=torch.Generator().manual_seed(501))
        d=torch.zeros_like(s)
        d[:,list(KIDX)]=v
        eps=1e-3
        with torch.no_grad():
            diff=(teacher.task(teacher.no_op(s+eps*d,4))-
                  teacher.task(teacher.no_op(s-eps*d,4)))/(2*eps)
        predicted=(j@v.unsqueeze(-1)).squeeze(-1)
        self.assertLess(float((diff-predicted).abs().max()),2e-3)

    def test_task_basis_global_spectrum_is_orthogonal(self):
        g=torch.diag(torch.tensor([1.,2.,3.,9.,4.,.1,.1,.1,.1,.1]))
        basis,gap,values=top_task_basis(g)
        self.assertTrue(torch.allclose(basis.T@basis,torch.eye(2),atol=1e-6))
        self.assertGreater(gap,0)
        self.assertTrue(torch.equal(basis[[0,4]],torch.zeros((2,2))))
        self.assertAlmostEqual(values[0],9.)

    def test_physical_svd_is_gauge_invariant(self):
        generator=torch.Generator().manual_seed(90100)
        V=torch.randn((32,D),generator=generator)
        U=torch.randn((D,D),generator=generator)
        R=torch.eye(D);R[0,1]=.7
        left=V@U;right=(V@R)@torch.linalg.solve(R,U)
        a,_=physical_weight_basis(left)
        b,_=physical_weight_basis(right)
        self.assertLess(float((a@a.T-b@b.T).abs().max()),1e-4)

    def test_signed_full_task_jacobian_and_norm_sham(self):
        g=torch.Generator().manual_seed(1343)
        j=torch.randn((100,2,len(KIDX)),generator=g)
        delta=protected_project(torch.randn((100,D),generator=g))
        sham,de,ne=tangent_matched_sham(j,delta,g)
        self.assertLess(de,1e-5)
        self.assertLess(ne,1e-5)
        self.assertTrue(torch.equal(sham[:,[0,4]],torch.zeros((100,2))))
        self.assertGreater(float((sham-delta).norm()),.1)

    def test_score_edit_enforces_full_protected_readout(self):
        teacher=RecurrentTeacher(901)
        h,x,_=source_batch(90200,15)
        rec=teacher.step(h,x)
        delta=torch.zeros_like(rec)
        delta[:,0]=.05
        with self.assertRaises(AssertionError):
            score_edit(teacher,rec,rec,delta,torch.ones(15,dtype=torch.bool),2)

    def test_source_only_small_excluded_training_and_assay(self):
        teacher=train_teacher(901,updates=8)
        factors1=fit_factors(teacher,901,0,updates=20)
        factors2=fit_factors(teacher,901,1,updates=20)
        result=assay_seed(901,teacher,factors1,factors2,n=30,null_arms=2,tangent_arms=2)
        self.assertEqual(result["seed"],901)
        self.assertEqual(result["generated_pairs"],30)
        self.assertEqual(set(result["causal_by_depth"]),{"2","4","6","8"})
        self.assertEqual(set(result["privileged_signed_full_task_jacobian_controls"]),{"4","8"})
        self.assertEqual(result["same_dose_random_K_controls_depth8"]["task_joint"]["arms"],2)

    def test_phase_guard(self):
        with self.assertRaises(ValueError):
            run("forbidden",teacher_steps=1,factor_steps=1,n=4,null_arms=1,tangent_arms=1)


if __name__=="__main__":
    unittest.main()
