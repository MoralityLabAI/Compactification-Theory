"""Independent finite oracles and exact-rational QO-01 regression suite."""
import unittest
from fractions import Fraction as Q
from itertools import product
from research.ingression_dynamics_v1.goal_aware_partition import (
    minimum_goal_partition, all_goal_words_signature, partitions,
    valid_deterministic_quotient, partition_bits, goal_worst_error,
    goal_mean_bayes_error, optimum_tv_for_partition, pareto_frontier,
    optimal_rate, fixture, deterministic_fixture, fixed_macro_tv_defect,
    kernel_shift_tv, hit_probability,
)


def same_partition(p, q, states):
    return all((p[s] == p[t]) == (q[s] == q[t]) for s in states for t in states)


class ExactGoalQuotientTests(unittest.TestCase):
    def test_bell_partition_counts(self):
        for n, count in ((1, 1), (2, 2), (3, 5), (4, 15), (5, 52)):
            self.assertEqual(sum(1 for _ in partitions(tuple(range(n)))), count)

    def test_six_state_minimal_quotient_and_memory_saving(self):
        states, goals, transitions = deterministic_fixture()
        labels, rounds = minimum_goal_partition(states, transitions, goals)
        self.assertEqual(tuple(labels.values()), (0, 0, 1, 1, 2, 3))
        self.assertEqual(rounds, 2)
        self.assertEqual(partition_bits(labels), 2)
        self.assertTrue(valid_deterministic_quotient(states, transitions, goals, labels))
        oracle = all_goal_words_signature(states, transitions, goals, len(states)-1)
        self.assertTrue(same_partition(labels, oracle, states))
        # The coarsest stable partition is also globally minimum cardinality.
        self.assertEqual(min(len(set(p.values())) for p in partitions(states)
                             if valid_deterministic_quotient(states, transitions, goals, p)), 4)

    def test_exhaustive_three_state_controllers_and_goals(self):
        states = (0, 1, 2)
        compared = 0
        for outputs in product((False, True), repeat=len(states)):
            goals = [dict(zip(states, outputs))]
            for transition_values in product(states, repeat=len(states)):
                f = {"step": dict(zip(states, transition_values))}
                labels, rounds = minimum_goal_partition(states, f, goals)
                oracle = all_goal_words_signature(states, f, goals, len(states)-1)
                self.assertTrue(same_partition(labels, oracle, states))
                self.assertTrue(valid_deterministic_quotient(states, f, goals, labels))
                valid = [p for p in partitions(states) if
                         valid_deterministic_quotient(states, f, goals, p)]
                self.assertEqual(len(set(labels.values())),
                                 min(len(set(p.values())) for p in valid))
                initial_goal_classes = len(set(outputs))
                self.assertLessEqual(rounds, len(states)-initial_goal_classes)
                compared += 1
        self.assertEqual(compared, 216)

    def test_exact_goal_bayes_cost_against_all_deterministic_readouts(self):
        states, goals, _ = fixture()
        mass = {s: Q(1, len(states)) for s in states}
        for p in partitions(states):
            labels = tuple(sorted(set(p.values())))
            oracle = min(sum((mass[s] * int(bool(goals[0][s]) != bool(out[p[s]]))
                              for s in states), Q(0))
                         for readout in product((False, True), repeat=len(labels))
                         for out in [dict(zip(labels, readout))])
            self.assertEqual(goal_mean_bayes_error(states, p, goals, mass), oracle)
            if goal_worst_error(states, p, goals) == Q(0):
                self.assertEqual(oracle, Q(0))

    def test_worst_case_probabilistic_readout_is_discontinuous(self):
        states, goals, _ = fixture()
        p = {s: 0 for s in states}
        self.assertEqual(goal_worst_error(states, p, goals), Q(1, 2))
        p[4] = 1
        self.assertEqual(goal_worst_error(states, p, goals), Q(0))


class RationalFrontierTests(unittest.TestCase):
    def test_analytical_two_outcome_minimax_for_all_goal_visible_partitions(self):
        states, goals, kernels = fixture()
        probabilities = (Q(1,5), Q(2,5), Q(3,5), Q(4,5))
        checked = 0
        for p in partitions(states):
            if goal_worst_error(states, p, goals):
                continue
            observed, macro = optimum_tv_for_partition(states, p, kernels)
            transients = {}
            for s in range(4):
                transients.setdefault(p[s], []).append(probabilities[s])
            expected = max(((max(v)-min(v))/2 for v in transients.values()), default=Q(0))
            self.assertEqual(observed, expected)
            self.assertEqual(fixed_macro_tv_defect(states, p, kernels, macro), expected)
            checked += 1
        self.assertEqual(checked, 15)  # all partitions of four non-goal states

    def test_exact_three_level_rate_distortion_curve(self):
        states, goals, kernels = fixture()
        self.assertIsNone(optimal_rate(states, goals, kernels, 0, Q(0)))
        for bits, error, labels in (
            (1, Q(3,10), (0,0,0,0,1)),
            (2, Q(1,10), (0,0,1,1,2)),
            (3, Q(0), (0,1,2,3,4)),
        ):
            value, partition, macro = optimal_rate(states, goals, kernels, bits, Q(0))
            self.assertEqual(value, error)
            self.assertEqual(partition, labels)
            self.assertEqual(fixed_macro_tv_defect(states,
                dict(zip(states,partition)),kernels,macro),error)

    def test_full_pareto_including_vacuous_constant_map(self):
        states, goals, kernels = fixture()
        frontier = pareto_frontier(states, goals, kernels)
        actual = [(p['bits'],p['tv_error'],p['goal_mean']) for p in frontier]
        self.assertEqual(actual,[
            (0,Q(0),Q(1,5)),
            (1,Q(3,10),Q(0)),
            (2,Q(1,10),Q(0)),
            (3,Q(0),Q(0)),
        ])
        # Repeating the search produces an identical, deterministic certificate.
        self.assertEqual(actual,[(p['bits'],p['tv_error'],p['goal_mean'])
                                 for p in pareto_frontier(states, goals, kernels)])

    def test_frozen_transfer_contract_under_kernel_shift(self):
        states, goals, kernels = fixture()
        _, labels, macro = optimal_rate(states, goals, kernels, 2, Q(0))
        partition = dict(zip(states, labels))
        before = fixed_macro_tv_defect(states, partition, kernels, macro)
        # The new environment permutes response probabilities across groups.
        shuffled = (Q(1,5), Q(4,5), Q(2,5), Q(3,5))
        changed = {s:{s:1-p,4:p} for s,p in enumerate(shuffled)}
        changed[4] = {4:Q(1)}
        new = {'hit':changed,'idle':kernels['idle']}
        shift = kernel_shift_tv(states,kernels,new)
        after = fixed_macro_tv_defect(states,partition,new,macro)
        self.assertEqual(before,Q(1,10))
        self.assertEqual(shift,Q(2,5))
        self.assertEqual(after,Q(1,2))
        self.assertEqual(after,before+shift)   # sharp on this witness

    def test_macro_recovery_vs_micro_horizon_bound(self):
        states, goals, kernels = fixture()
        error, labels, macro = optimal_rate(states,goals,kernels,2,Q(0))
        goal_label = labels[-1]
        for s in states:
            actual = hit_probability(kernels['hit'],s,{4},3)
            predicted = hit_probability(macro['hit'],labels[s],{goal_label},3)
            self.assertLessEqual(abs(actual-predicted),1-(1-error)**3)

if __name__ == '__main__':
    unittest.main()
