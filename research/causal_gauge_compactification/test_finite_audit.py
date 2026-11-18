"""CGC-01 exact/numerical source-only reference tests.

All inputs are toy finite matrices; there are no trained checkpoints or
protected test-set outcomes. The independent rational witness uses Fraction.
"""
import unittest

import numpy as np

from finite_audit import (
    affine_observability_matrix, exact_affine_krylov_rank,
    exact_rational_rank, finite_newton_remainder_bound,
    least_norm_public_repair, local_finite_compactification_bound,
    minimal_real_affine_quotient, minimum_code_bits, null_basis,
    observability_gramian, physical_svd_write,
    protected_task_spectrum, synthetic_receipt, task_optimal_block,
)


class CompactificationMathematics(unittest.TestCase):
    def test_protected_intersection_rank_and_task_singulars(self):
        p=np.array([[1.,0,0,0]])
        task=np.array([[0.,3,0,0],[0.,0,1,0]])
        s=protected_task_spectrum(p,task)
        self.assertEqual(s.available_dimension,3)
        self.assertEqual(s.task_rank,2)
        self.assertTrue(np.allclose(s.task_singular_values[:2],[3,1],atol=1e-10))
        self.assertTrue(np.allclose(s.task_gramian.T,s.task_gramian))

    def test_exact_truncated_spectrum_frobenius_and_operator(self):
        p=np.array([[1.,0,0,0]])
        task=np.array([[0.,3,0,0],[0.,0,1,0]])
        e, op, sq=task_optimal_block(p,task,1)
        self.assertAlmostEqual(op,1.0)
        self.assertAlmostEqual(sq,1.0)
        self.assertTrue(np.allclose(e.T@e,np.eye(1)))
        self.assertTrue(np.allclose(p@e,0))
        q=null_basis(p)
        full=task@q
        omitted=full@(np.eye(3)-q.T@e@e.T@q)
        self.assertAlmostEqual(np.linalg.norm(omitted,2),op)
        self.assertAlmostEqual(np.linalg.norm(omitted,'fro')**2,sq)
        e2,op2,sq2=task_optimal_block(p,task,2)
        self.assertAlmostEqual(op2,0.,places=10)
        self.assertAlmostEqual(sq2,0.,places=10)
        self.assertTrue(np.allclose(p@e2,0))

    def test_rank_lower_bound_for_exact_linear_decoder(self):
        p=np.array([[1.,0.,0.,0.]])
        a=np.array([[0.,0.,3.,0.],[0.,2.,0.,0.]])
        rank=protected_task_spectrum(p,a).task_rank
        self.assertEqual(rank,2)
        # An r=1 encoder makes D H Q have rank<=1, so cannot equal A Q.
        q=null_basis(p)
        h=np.array([[1.,0,0,0]])
        self.assertLessEqual(np.linalg.matrix_rank(h@q),1)
        self.assertEqual(np.linalg.matrix_rank(a@q),2)

    def test_weight_svd_can_miss_the_task_direction(self):
        w=np.diag([10.,1.]); features=np.array([[1.,1.]])
        full=features@w
        top,gap=physical_svd_write(features,w,1)
        self.assertTrue(np.allclose(full,[10,1]))
        self.assertTrue(np.allclose(top,[10,0]))
        self.assertAlmostEqual(gap,9)
        task=np.array([[0.,1.]])
        e,tail,_=task_optimal_block(np.zeros((0,2)),task,1)
        self.assertAlmostEqual(tail,0)
        self.assertAlmostEqual(np.linalg.norm(task@e),1)
        self.assertAlmostEqual(float((task@top[0]).item()),0)

    def test_physical_factor_gl_gauge_invariance(self):
        g=np.random.default_rng(61)
        v=g.normal(size=(5,3));u=g.normal(size=(3,4));delta=g.normal(size=(20,5))
        r=np.eye(3);r[0,1]=.7;r[1,2]=-.3
        w=v@u
        gauged=(v@r)@np.linalg.solve(r,u)
        self.assertTrue(np.allclose(w,gauged,atol=1e-13))
        for rank in (1,2,3,4):
            a, gap=physical_svd_write(delta,w,rank)
            b, gap2=physical_svd_write(delta,gauged,rank)
            self.assertTrue(np.allclose(a,b,atol=1e-11))
            self.assertEqual(gap is None,gap2 is None)

    def test_state_orthogonal_frame_equivariance(self):
        p=np.array([[1.,0.,0.,0.]])
        t=np.array([[0.,3.,0.,0.],[0.,0.,2.,0.]])
        q,_=np.linalg.qr(np.random.default_rng(9).normal(size=(4,4)))
        e,tail,_=task_optimal_block(p,t,2)
        rotated_e,rotated_tail,_=task_optimal_block(p@q.T,t@q.T,2)
        self.assertAlmostEqual(tail,rotated_tail,places=12)
        self.assertTrue(np.allclose(rotated_e@rotated_e.T,q@e@e.T@q.T,atol=1e-10))

    def test_nonorthogonal_state_change_alters_euclidean_singular_values(self):
        task=np.array([[1.,0.]])
        scale=np.diag([4.,1.])
        untransformed=protected_task_spectrum(np.zeros((0,2)),task)
        changed=protected_task_spectrum(np.zeros((0,2)),task@np.linalg.inv(scale))
        self.assertAlmostEqual(untransformed.task_singular_values[0],1)
        self.assertAlmostEqual(changed.task_singular_values[0],.25)

    def test_multichannel_public_minimum_norm_repair(self):
        p=np.array([[1.,0,0,0]])
        q=null_basis(p)
        public=np.array([[0.,1.,0.,0.],[0.,0.,1.,0.]])
        residual=np.array([.4,-.2])
        step=least_norm_public_repair(q,public,residual)
        self.assertTrue(np.allclose(p@step,0))
        self.assertTrue(np.allclose(public@step,-residual))
        self.assertTrue(np.allclose(step,np.array([0,-.4,.2,0])))
        random=np.array([0.,0.,0.,2.])
        self.assertGreater(np.linalg.norm(step+random),np.linalg.norm(step))

    def test_undercapacity_public_linear_correction_is_rejected(self):
        q=null_basis(np.array([[1.,0,0]]))
        b=np.array([[0.,1,0],[0.,2,0]])
        with self.assertRaisesRegex(ValueError,"not surjective"):
            least_norm_public_repair(q,b,np.array([1.,1.]))

    def test_nonlinear_tangent_safety_failure_and_newton_remainder(self):
        # At base h=(0,0,0), public G=h0+h1^2, so d=(0,.2,0)
        # is first-order invisible but changes public G by .04.
        d=np.array([0.,.2,0.]); public=lambda z:z[0]+z[1]**2
        self.assertAlmostEqual(public(d)-public(np.zeros(3)),.04)
        p=np.array([[1.,0,0]])
        q=null_basis(p)
        pub_jacobian=np.array([[1.,.4,0.]])
        residual=np.array([public(d)])
        step=least_norm_public_repair(q,pub_jacobian,residual)
        self.assertTrue(np.allclose(step,np.array([0,-.1,0]),atol=1e-12))
        corrected=public(d+step)
        self.assertAlmostEqual(corrected,.01)
        self.assertLessEqual(corrected,finite_newton_remainder_bound(2.,step)+1e-12)

    def test_local_second_order_tail_bound_on_scalar_task(self):
        rho=.1
        # Task g(x1)=x1+.5*x1^2, Hessian uniformly bounded by M=1.
        error=.1+.5*.1**2
        bound=local_finite_compactification_bound(1.,rho,1.)
        self.assertLessEqual(error,bound+1e-12)
        self.assertAlmostEqual(bound,.11)
        with self.assertRaises(ValueError):
            local_finite_compactification_bound(1.,-.1,1.)

    def test_real_affine_minimal_quotient_and_exact_fraction_rank(self):
        a=np.array([[1.,1.,0.],[0.,1.,0.],[0.,0.,1.]])
        c=np.array([[1.,0.,0.]])
        b=np.array([.1,.2,.3])
        quo=minimal_real_affine_quotient([a],c,[b])
        self.assertEqual(quo.dimension,2)
        self.assertEqual(exact_affine_krylov_rank([[1,1,0],[0,1,0],[0,0,1]],[[1,0,0]]),2)
        self.assertTrue(np.allclose(quo.encoder@a,quo.transitions[0]@quo.encoder))
        self.assertTrue(np.allclose(quo.readout@quo.encoder,c))
        for _ in range(10):
            h=np.array([1.1,2.2,3.3]); z=quo.encoder@h
            self.assertTrue(np.allclose(quo.encoder@(a@h+b),quo.transitions[0]@z+quo.input_offsets[0]))
            break
        # Coordinate 3 is invisible for all depths.
        self.assertTrue(np.allclose(quo.encoder@np.array([1,2,3]),quo.encoder@np.array([1,2,9])))

    def test_controlled_action_alphabet_can_force_full_dimension(self):
        a=np.array([[1.,1,0],[0,1,0],[0,0,1]])
        perm=np.array([[1.,0,0],[0,0,1],[0,1,0]])
        c=np.array([[1.,0,0]])
        self.assertEqual(minimal_real_affine_quotient([a],c).dimension,2)
        self.assertEqual(minimal_real_affine_quotient([a,perm],c).dimension,3)

    def test_observability_rank_grows_then_saturates_with_horizon(self):
        a=np.array([[1.,1,0],[0,1,0],[0,0,.5]])
        c=np.array([[1.,0,0]])
        ranks=[np.linalg.matrix_rank(affine_observability_matrix(a,c,k)) for k in (1,2,3,4)]
        self.assertEqual(ranks,[1,2,2,2])
        grams=[observability_gramian(a,c,k) for k in (1,2,3)]
        for g,h in zip(grams,grams[1:]):
            self.assertGreaterEqual(np.linalg.eigvalsh(h-g).min(),-1e-12)

    def test_nonnormal_transient_can_exceed_spectral_radius(self):
        a=np.array([[.5,3.],[0,.5]])
        self.assertAlmostEqual(max(abs(np.linalg.eigvals(a))),.5)
        self.assertGreater(np.linalg.norm(a,2),3.)
        self.assertGreater(np.linalg.norm(a@np.array([0.,1.])),3.)
        self.assertLess(np.linalg.norm(np.linalg.matrix_power(a,20)),.01)

    def test_separated_futures_need_precision_bits(self):
        self.assertEqual(minimum_code_bits(2,1),(1,1))
        self.assertEqual(minimum_code_bits(9,1),(4,4))
        self.assertEqual(minimum_code_bits(9,2),(4,2))
        self.assertEqual(minimum_code_bits(1,8),(0,0))
        with self.assertRaises(ValueError): minimum_code_bits(0,1)

    def test_exact_fraction_rank_separates_rational_dependencies(self):
        self.assertEqual(exact_rational_rank([[1,1,0],[2,2,0],[0,0,1]]),2)
        self.assertEqual(exact_rational_rank([[1,1,0],[1,0,1],[0,1,1]]),3)

    def test_synthetic_receipt_keeps_scientific_scope_explicit(self):
        r=synthetic_receipt()
        self.assertEqual(r["full_real_affine_quotient_rank"],2)
        self.assertEqual(r["integer_rational_krylov_rank"],2)
        self.assertEqual(r["protected_task_rank"],1)
        self.assertTrue(r["task_gradient_orthogonal_to_top_weight_output"])
        self.assertFalse(r["trained_BPD_evidence"])


if __name__ == '__main__':
    unittest.main()
