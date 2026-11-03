import unittest
import numpy as np
from experiments.h_m0_pruner import make_episode
from experiments.h_m0_terminal import BUDGET
from experiments.h_m0_fair_baselines import greedy_map

class FairStopTests(unittest.TestCase):
    def test_uninformative_evidence_causes_stop(self):
        ep=make_episode(975)
        ep.likelihoods[:,:]=0.5
        r=greedy_map(ep)
        self.assertEqual(r['probes'], [])
        self.assertAlmostEqual(r['cost'],0)

    def test_greedy_cost_limit_and_one_use_per_probe(self):
        for j in range(30):
            r=greedy_map(make_episode(9750+j))
            self.assertLessEqual(r['cost'],BUDGET)
            self.assertEqual(len(set(r['probes'])),len(r['probes']))

    def test_immediate_accuracy_gain_is_nonnegative_without_penalty(self):
        ep=make_episode(376)
        p=ep.prior
        for likelihood in ep.likelihoods:
            p_pos=(p*likelihood)/(p@likelihood)
            p_neg=(p*(1-likelihood))/(p@(1-likelihood))
            expected=(p@likelihood)*p_pos.max()+(p@(1-likelihood))*p_neg.max()
            self.assertGreaterEqual(expected+1e-9,p.max())

if __name__=='__main__':unittest.main()
