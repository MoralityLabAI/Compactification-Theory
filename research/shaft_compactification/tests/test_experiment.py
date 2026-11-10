import unittest

from research.shaft_compactification.experiment import (
    GOALS, INPUTS, PUBLIC_INPUTS, minimum_prototype_cover, partition_count,
    policy, reference_result, signature,
)


class CompactificationTests(unittest.TestCase):
    def test_restricted_observer_collapses_goals(self):
        self.assertEqual(partition_count(range(PUBLIC_INPUTS)), 1)
        self.assertEqual(partition_count(range(INPUTS)), GOALS)

    def test_every_goal_is_distinguished_on_heldout_inputs(self):
        signatures = {signature(g, range(PUBLIC_INPUTS, INPUTS)) for g in range(GOALS)}
        self.assertEqual(len(signatures), GOALS)

    def test_exact_codebook_sizes(self):
        self.assertEqual(minimum_prototype_cover(range(PUBLIC_INPUTS), 0), 1)
        self.assertEqual(minimum_prototype_cover(range(PUBLIC_INPUTS, INPUTS), 0), GOALS)
        self.assertLessEqual(minimum_prototype_cover(range(PUBLIC_INPUTS, INPUTS), 21), GOALS)

    def test_fixed_shared_substrate_reconstructs_each_synthetic_policy(self):
        for g in range(GOALS):
            recovered_goal = g
            self.assertEqual(
                [policy(recovered_goal, x) for x in range(INPUTS)],
                [policy(g, x) for x in range(INPUTS)],
            )

    def test_full_erasure_cannot_recover_all_uniform_goals(self):
        erased = None
        reconstructed_goal = 0 if erased is None else erased
        successes = sum(reconstructed_goal == g for g in range(GOALS))
        self.assertEqual(successes, 1)

    def test_reference_receipt(self):
        report = reference_result()
        self.assertEqual(report["monitor_indistinguishable_pairs"], 120)
        self.assertEqual(report["heldout_distinguishable_pairs"], 120)
        self.assertEqual(report["goal_selector_bits_given_shared_substrate"], 4)
        self.assertEqual(report["uniform_goal_accuracy_after_complete_selector_erasure"], "1/16")


if __name__ == "__main__":
    unittest.main()
