"""CPU-only tests for MONO-HODGE-01. No pretrained checkpoint tested."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hodge01 as H


class Hodge01(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(42)

    def test_posterior_order_gap_zero_when_commutative(self):
        a=np.array([0.3,1.2,-0.1]);t=a.copy()
        r=H.order_gap_report(a,a,t,['a','b','c'])
        self.assertEqual(r['mean_abs_order_gap_nats'],0.0)
        self.assertFalse(r['has_operator_commutator_measurement'])

    def test_order_gap_detects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            H.order_gap_report([1,2],[1],[1,2],['x','y'])

    def test_reader_needs_explicit_schema(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'pairs.jsonl'
            f.write_text(json.dumps({'family_id':'a','log_odds_ab':1.,'log_odds_ba':2.,'true_log_odds':1.5})+'\n'
                         +json.dumps({'family_id':'b','log_odds_ab':0.2,'log_odds_ba':0.1,'true_log_odds':0.1})+'\n')
            pairs,r=H.read_normalized_pairs(f)
            self.assertEqual(len(pairs),2)
            self.assertAlmostEqual(r['mean_abs_order_gap_nats'],0.55)
            f.write_text('{"foo": 1}\n')
            with self.assertRaises(ValueError):H.read_normalized_pairs(f)

    def test_woodbury_vs_dense(self):
        X=self.rng.normal(size=(20,70));B=self.rng.normal(size=(70,5))
        C=X.T@X/len(X)+0.04*np.eye(X.shape[1])
        np.testing.assert_allclose(H.woodbury_solve(X,B,0.04),np.linalg.solve(C,B),atol=1e-9)

    def test_modes_are_generalized_eigenvectors(self):
        X=self.rng.normal(size=(40,25));D=self.rng.normal(size=(12,25))
        vals,V=H.antisymmetric_modes(D,X,ridge=0.15,topk=4)
        C=X.T@X/len(X)+0.15*np.eye(25)
        B=D.T@D/len(D)
        np.testing.assert_allclose(V.T@C@V,np.eye(4),atol=1e-8)
        np.testing.assert_allclose(B@V,C@V@np.diag(vals),atol=1e-8)

    def test_modes_detect_planted_feature(self):
        X=self.rng.normal(size=(100,7))
        D=self.rng.normal(size=(30,7))*0.01
        D[:,4]+=self.rng.normal(size=30)*4
        vals,V=H.antisymmetric_modes(D,X,ridge=0.1,topk=1)
        self.assertEqual(V.shape,(7,1))
        self.assertGreater(abs(V[4,0]),2*np.max(np.abs(V[[0,1,2,3,5,6],0])))
        self.assertGreater(vals[0],0)

    def test_group_folds_are_row_permutation_invariant(self):
        groups=np.array(['a','b','a','c','d','e','b','f','g','h'])
        folds=H.deterministic_group_folds(groups,3)
        perm=self.rng.permutation(len(groups))
        np.testing.assert_array_equal(folds[perm],H.deterministic_group_folds(groups[perm],3))
        self.assertEqual(folds[0],folds[2])

    def test_predictive_gain_reports_added_signal(self):
        n=300;S=self.rng.normal(size=(n,2));Z=self.rng.normal(size=(n,1))
        y=2*S[:,0]+3*Z[:,0]+self.rng.normal(size=n)*0.2
        groups=np.array([f'group{i}' for i in range(n)])
        r=H.incremental_probe_value(S,Z,y,groups,ridge=3)
        self.assertGreater(r['incremental_mse_gain'],4.0)
        self.assertLess(r['state_plus_latent_mse'],r['state_only_mse'])
        self.assertEqual(r['evidence'],'group_disjoint_predictive_probe_not_mutual_information')

    def test_predictive_negative_control_noise(self):
        n=200;S=self.rng.normal(size=(n,3));Z=self.rng.normal(size=(n,3))
        y=2*S[:,0]+self.rng.normal(size=n)*0.05
        g=[f'item{i}' for i in range(n)]
        r=H.incremental_probe_value(S,Z,y,g,ridge=10)
        self.assertLess(r['incremental_mse_gain'],0.03)

    def test_high_dimensional_probe_uses_dual_ridge(self):
        n=54;S=self.rng.normal(size=(n,3));Z=self.rng.normal(size=(n,300))
        y=2*S[:,0]+0.8*Z[:,0]+self.rng.normal(size=n)*0.1
        report=H.incremental_probe_value(S,Z,y,[f'family_{i}' for i in range(n)],folds=3,ridge=20)
        self.assertTrue(np.isfinite(report['incremental_mse_gain']))

    def test_single_family_interval_is_unavailable(self):
        report=H.order_gap_report([1],[2],[1],['only_family'])
        self.assertEqual(report['mean_abs_order_gap_cluster_ci95'],[None,None])

    def test_affine_commutator_exact_identity(self):
        d=4;W=self.rng.normal(size=(d,d))*0.2
        h=self.rng.normal(size=(9,d));a=self.rng.normal(size=(9,d));b=self.rng.normal(size=(9,d))
        np.testing.assert_allclose(H.commutator(H.affine_evidence_transition,W,h,a,b),
                                   (a-b)@W.T,atol=1e-12)

    def test_rank_one_editor_repairs_toy(self):
        r=H.toy_demo()
        self.assertEqual(r['edit_rank'],1)
        self.assertLess(r['post_test_commutator_l2_mean'],1e-5)
        self.assertGreater(r['reduction_fraction'],0.999)
        self.assertTrue(r['fraction_random_worse_than_edit']>0.99)
        self.assertFalse(r['real_qwen_or_hf_data_loaded'])

    def test_unidentifiable_finite_difference_rejected(self):
        h=np.ones((3,2));a=np.ones_like(h);b=a.copy();W=np.zeros((2,2))
        with self.assertRaises(ValueError):
            H.finite_diff_rankone_edit(H.affine_evidence_transition,W,h,a,b,[1,0],h)

    def test_change_in_truth_does_not_count_as_order_gap(self):
        a=np.array([1.,1.]);b=np.array([1.,1.]);truth=np.array([-2.,3.])
        r=H.order_gap_report(a,b,truth,['a','b'])
        self.assertEqual(r['mean_abs_order_gap_nats'],0.)
        self.assertGreater(r['mean_abs_posterior_error_nats'],1.)

if __name__=='__main__':unittest.main()