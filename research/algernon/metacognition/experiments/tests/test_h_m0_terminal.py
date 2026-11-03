"""Unit tests for exploratory H-M0 terminal utility experiment."""
import unittest
import numpy as np
import torch
from experiments.h_m0_pruner import make_episode, posterior_update
from experiments.h_m0_terminal import (
    BUDGET, COST_PENALTY, N_PROBES, STOP, TerminalPlanner, Selector,
    collect_training_worlds, inputs_for_state, posterior_from_outcomes, run,
)


class TerminalPolicyTests(unittest.TestCase):
    def test_dp_never_worse_than_immediate_halt_in_expectation(self):
        for seed in (2, 11, 24, 123):
            ep = make_episode(seed)
            oracle = TerminalPlanner(ep)
            utility, action = oracle.evaluate()
            self.assertGreaterEqual(utility + 1e-10, ep.prior.max())
            self.assertTrue(0 <= action <= STOP)

    def test_dp_identical_after_reordering_evidence(self):
        ep = make_episode(52)
        outcome1 = posterior_update(posterior_update(ep.prior, ep.likelihoods[0], 1), ep.likelihoods[3], 0)
        outcome2 = posterior_update(posterior_update(ep.prior, ep.likelihoods[3], 0), ep.likelihoods[0], 1)
        outcome3 = posterior_from_outcomes(ep, 1<<0 | 1<<3, 1<<0)
        np.testing.assert_allclose(outcome1, outcome2, atol=1e-13)
        np.testing.assert_allclose(outcome1, outcome3, atol=1e-13)

    def test_cost_penalty_makes_uninformative_probes_unwanted(self):
        ep = make_episode(10)
        ep.likelihoods[:,:] = 0.5
        planner = TerminalPlanner(ep)
        value, action = planner.evaluate()
        self.assertEqual(action, STOP)
        self.assertAlmostEqual(value, ep.prior.max())

    def test_optimal_action_is_legal_and_never_exceeds_budget(self):
        for seed in (3, 7, 22, 111, 131):
            ep = make_episode(seed)
            r = run('exact_dp_oracle', ep, None, teacher=TerminalPlanner(ep))
            self.assertLessEqual(r['cost'], BUDGET+1e-9)
            self.assertEqual(len(r['probes']), len(set(r['probes'])))
            self.assertAlmostEqual(r['terminal_utility'], r['expected_correct'] - COST_PENALTY*r['cost'])

    def test_model_respects_permutation_of_probes(self):
        torch.manual_seed(103)
        ep = make_episode(14)
        model = Selector()
        rows, available, _, _ = inputs_for_state(ep,0,0)
        logits = model(torch.tensor(rows[None,:]),torch.tensor(available[None,:]))[0].detach().numpy()
        perm = np.array([3,4,1,2,6,5,7,0])
        logits_p = model(torch.tensor(rows[perm][None,:]),torch.tensor(available[perm][None,:]))[0].detach().numpy()
        np.testing.assert_allclose(logits_p[:-1], logits[perm],atol=1e-6)
        self.assertAlmostEqual(float(logits_p[-1]),float(logits[-1]),places=6)

    def test_expert_labels_are_legal(self):
        x, m, y, _=collect_training_worlds(600, 5, 2, 777)
        self.assertEqual(x.shape[1:],(N_PROBES,14))
        for mask,label in zip(m,y):
            self.assertTrue(int(label)==STOP or mask[int(label)]==1)

if __name__=='__main__':unittest.main()
