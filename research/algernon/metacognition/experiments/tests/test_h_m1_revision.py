"""CPU correctness tests for synthetic grammar-revision environment."""
import unittest
import numpy as np
import torch
from experiments.h_m1_revision import (
    TABLE,SIMPLE,COMPOSED,make_world,posterior,Planner,legal,EXPAND,STOP,
    predicted_labels,expected_accuracy, state_cost,action_mask,feature,Policy,
    evaluate_episode, BUDGET
)

class RevisionTests(unittest.TestCase):
    def test_grammar_is_distinct_and_compositional(self):
        self.assertEqual(TABLE.shape,(12,4))
        self.assertEqual(len({tuple(x) for x in TABLE}),12)
        np.testing.assert_array_equal(TABLE[8],[0,1,1,0])
        np.testing.assert_array_equal(TABLE[6],[0,0,0,1])

    def test_priors_are_normalized_and_leakage_free(self):
        for seed in [10,11,12,13]:
            w=make_world(seed)
            self.assertAlmostEqual(float(w.prior.sum()),1.)
            self.assertAlmostEqual(float(w.weights.sum()),1.)
            self.assertTrue(0<=w.truth<12)
            self.assertTrue(np.isin(w.observed,[0,1]).all())

    def test_expansion_replays_prior_evidence(self):
        w=make_world(445)
        p0=posterior(w,0,0,True)
        np.testing.assert_allclose(p0,w.prior,rtol=1e-12)
        mask=1|4;bits=int(w.observed[0])|int(w.observed[2])<<2
        p=posterior(w,mask,bits,True)
        likelihood=np.prod([np.where(TABLE[:,i]==w.observed[i],1-w.noise[i],w.noise[i]) for i in (0,2)],axis=0)
        expected=w.prior*likelihood;expected/=expected.sum()
        np.testing.assert_allclose(p,expected,rtol=1e-12)

    def test_unexpanded_predictions_use_primitive_grammar(self):
        w=make_world(441)
        p=posterior(w,3,1,False)
        self.assertEqual(len(p),6)
        np.testing.assert_array_equal(predicted_labels(w,3,1,False),(p@TABLE[:6,:]>.5).astype(np.int8))

    def test_unavailable_composition_priors_hidden_until_expand(self):
        w=make_world(778)
        from experiments.h_m1_revision import feature
        before=feature(w,0,0,False)
        after=feature(w,0,0,True)
        np.testing.assert_array_equal(before[18:30],np.r_[w.prior[:6],np.zeros(6)].astype(np.float32))
        np.testing.assert_allclose(after[18:30],w.prior.astype(np.float32))

    def test_pay_and_enforce_budget(self):
        w=make_world(547)
        self.assertEqual(state_cost(w,0,False),0)
        self.assertAlmostEqual(state_cost(w,3,True),w.costs[0]+w.costs[1]+w.expand_cost)
        for mask in range(16):
            for expanded in (False,True):
                if state_cost(w,mask,expanded)<=BUDGET+1e-10:
                    opts=legal(w,mask,expanded)
                    self.assertIn(STOP,opts)
                    for a in opts:
                        if a==STOP:continue
                        self.assertLessEqual(state_cost(w,mask|(1<<a) if a<4 else mask, expanded or a==EXPAND),BUDGET+1e-9)

    def test_exact_planner_beats_stopping_in_expectation(self):
        for seed in range(40,48):
            w=make_world(seed)
            val,act=Planner(w).value()
            self.assertGreaterEqual(val+1e-10, expected_accuracy(w,0,0,False))
            self.assertIn(act,legal(w,0,False))

    def test_network_valid_actions_only(self):
        w=make_world(995)
        model=Policy().eval()
        m=15; expanded=True
        x=torch.from_numpy(feature(w,m,15,expanded))[None]
        a=torch.from_numpy(action_mask(w,m,expanded))[None]
        self.assertEqual(model(x,a).argmax().item(),STOP)

    def test_all_arms_respect_budget(self):
        model=Policy().eval()
        for seed in range(20,26):
            w=make_world(seed)
            for arm in ("simple_only","random","always_expand","myopic_expand","surprise_revision","policy","oracle_dp"):
                r=evaluate_episode(w,arm,model)
                self.assertLessEqual(r['cost'],BUDGET+1e-9)
                self.assertGreaterEqual(r['weighted_accuracy'],0)
                self.assertLessEqual(r['weighted_accuracy'],1+1e-12)
                self.assertEqual(r['expanded'],int('expand' in r['actions']))

if __name__=="__main__":unittest.main()
