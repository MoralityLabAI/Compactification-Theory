"""QO-15 split safety, QO-07 reachability and decision-margin checks."""
import unittest
from fractions import Fraction as Q

from research.ingression_dynamics_v1.certificate_student_data import (
    TRAIN_RATES, VALIDATION_RATES, TEST_RATES,
    TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS,
    teacher_q_values, evidence_features
)
from research.ingression_dynamics_v1.certificate_boundary_student import (
    ARMS, AUGMENT_CAP_PER_RATE, _successor_states, _frontier_distance,
    parent_ledgers, frontier_examples, swap_action_cells, margin_metrics,
    _model_from_qo14
)
from research.ingression_dynamics_v1.certificate_boundary_test_contract import (
    FROZEN_RATES, FROZEN_SUPERVISED_SEEDS, FROZEN_POLICY_SEEDS,
    FROZEN_SCENARIOS, FROZEN_POLICY_CHECKPOINTS, frozen_contract_digest,
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_rnn, trainable_count, inference_multiplies
)


class BoundaryStudentTests(unittest.TestCase):
    def test_source_frozen_rate_families_and_seeds_are_disjoint(self):
        self.assertFalse(set(FROZEN_RATES) & (
            set(TRAIN_RATES) | set(VALIDATION_RATES) | set(TEST_RATES)))
        old_seeds = set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) | set(TEST_SEEDS)
        self.assertFalse(set(FROZEN_SUPERVISED_SEEDS) & old_seeds)
        self.assertFalse(set(FROZEN_POLICY_SEEDS) & old_seeds)
        self.assertFalse(set(FROZEN_POLICY_SEEDS) & set(FROZEN_SUPERVISED_SEEDS))
        self.assertEqual(len(FROZEN_SCENARIOS), 8)
        self.assertEqual(FROZEN_POLICY_CHECKPOINTS, (32, 56, 80))
        self.assertEqual(ARMS, (
            "bottleneck", "qo14_recurrent", "qo15_boundary_recurrent",
            "exact_h2_teacher"))
        self.assertEqual(len(frozen_contract_digest()), 64)

    def test_counterfactual_successors_are_source_valid_histories(self):
        _, _, _, _, source = next(parent_ledgers("train"))
        source_receipts = tuple(source.receipts)
        successors = list(_successor_states(source))
        self.assertLessEqual(len(successors), 20)
        self.assertGreaterEqual(len(successors), 4)
        self.assertEqual(tuple(source.receipts), source_receipts)
        for depth, path, child in successors:
            self.assertEqual(len(path), depth)
            self.assertEqual(len(child.receipts), len(source_receipts) + depth)
            self.assertEqual(tuple(child.receipts[:len(source_receipts)]),
                             source_receipts)
            self.assertEqual(tuple(child.receipts[-depth:]), tuple(
                (i, 0, bit) for i, bit in path))
            self.assertEqual(child.query_count(), source.query_count() + depth)
            self.assertEqual(len(evidence_features(child)), 2)
            if _frontier_distance(child) is not None:
                self.assertTrue(all(0 <= q <= 1 for q in teacher_q_values(child)))

    def test_permutation_swaps_both_q_labels_and_ties(self):
        _, _, _, _, source = next(parent_ledgers("validation"))
        q = teacher_q_values(source)
        example = {
            "tokens": evidence_features(source), "q": q,
            "teacher_action": 0 if q[0] >= q[1] else 1,
        }
        swapped = swap_action_cells(example)
        self.assertEqual(swapped["q"], q[::-1])
        self.assertEqual(swapped["tokens"], example["tokens"][::-1])
        self.assertEqual(swapped["teacher_action"],
                         0 if q[1] >= q[0] else 1)
        self.assertEqual(swap_action_cells(swapped)["tokens"],
                         example["tokens"])
        self.assertEqual(swap_action_cells(swapped)["q"], q)

    def test_margin_regret_never_exceeds_predicted_delta_error(self):
        reference = create_rnn(seed=23)
        _, _, _, _, source = next(parent_ledgers("validation"))
        base = {
            "tokens": evidence_features(source),
            "q": (Q(0), Q(1, 5)),
            "teacher_action": 1,
        }
        cases = [
            base,
            {**base, "q": (Q(1, 5), Q(0)), "teacher_action": 0},
            {**base, "q": (Q(0), Q(0)), "teacher_action": 0},
            {**base, "q": (Q(1), Q(1)), "teacher_action": 0},
        ]
        metric = margin_metrics(reference, cases)
        self.assertEqual(metric["examples"], 4)
        self.assertEqual(metric["informative"], 2)
        self.assertLessEqual(metric["informative_action_disagreements"], 2)
        self.assertEqual(sum(metric["teacher_margin_buckets"].values()), 4)

    def test_frozen_qo14_reference_weights_unmodified(self):
        model, digest = _model_from_qo14()
        self.assertEqual(len(digest), 64)
        self.assertEqual(trainable_count(model), 138)
        self.assertEqual(inference_multiplies(model), 240)

    def test_validation_frontier_is_bounded_and_labeled_after_selection(self):
        selected, stats = frontier_examples("validation")
        self.assertEqual(stats["split"], "validation")
        self.assertEqual(len(stats["unique_eligible_by_rate"]), 3)
        self.assertLessEqual(len(selected), 3 * AUGMENT_CAP_PER_RATE["validation"])
        self.assertEqual(stats["selected"], len(selected))
        self.assertEqual(stats["exact_teacher_labels"], len(selected))
        self.assertEqual(stats["informative"], sum(
            ex["q"][0] != ex["q"][1] for ex in selected))
        self.assertTrue(all(ex["split"] == "validation_frontier"
                            for ex in selected))
        self.assertTrue(all(ex["seed"] in VALIDATION_SEEDS for ex in selected))
        self.assertTrue(all(len(ex["tokens"]) == 2 for ex in selected))


if __name__ == "__main__":
    unittest.main()
