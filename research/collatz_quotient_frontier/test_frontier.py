import itertools
import unittest
from collections import Counter

from .collatz_core import analyze, modular_labels
from .frontier import grouped_counts, huffman_cost, message_success, run, run_parameter


class FrontierChecks(unittest.TestCase):
    def test_message_bound_by_all_encoders_and_decoders(self):
        labels = {0: 0, 1: 1, 2: 2, 3: 2}
        counts = {0: Counter(labels.values())}
        for messages in (1, 2, 3):
            brute = 0
            for encoding in itertools.product(range(messages), repeat=len(labels)):
                for decoding in itertools.product(range(3), repeat=messages):
                    brute = max(brute, sum(decoding[encoding[n]] == b for n, b in labels.items()))
            self.assertEqual(message_success(counts, messages), brute)

    def test_known_collision_distribution_changes_accuracy(self):
        row = run_parameter(13)
        self.assertEqual(row["quotient_only_correct"], 4)
        self.assertEqual(row["states"], 7)
        self.assertEqual(row["quotient_only_uniform_cycles_fraction"], "2/3")
        self.assertEqual(row["minimum_extra_fixed_bits_for_zero_error"], 1)
        self.assertEqual(row["frontier"][-1]["uniform_core_correct"], 7)

    def test_frontier_monotone_and_attains_zero_error(self):
        for d in (13, 29, 313, 1001, 2021):
            row = run_parameter(d)
            scores = [r["uniform_core_correct"] for r in row["frontier"]]
            self.assertEqual(scores, sorted(scores))
            self.assertEqual(scores[-1], row["states"])
            for step in row["frontier"]:
                for sham in step["hash_side_messages"]:
                    self.assertLessEqual(sham["correct"], step["uniform_core_correct"])

    def test_huffman_sanity_and_known_optimum(self):
        self.assertEqual(huffman_cost([99]), 0)
        self.assertEqual(huffman_cost([3, 3]), 6)
        self.assertEqual(huffman_cost([1, 1, 2, 4]), 14)
        self.assertEqual(huffman_cost([1, 1, 1, 1]), 8)
        with self.assertRaises(ValueError):
            huffman_cost([])

    def test_cycles_stay_inside_quotients(self):
        for d in (13, 29, 73, 313, 2021):
            core, q = analyze(d), modular_labels(d)
            for n in core.states:
                self.assertEqual(q[n % d], q[core.basin[n] % d])

    def test_replay_and_invalid_contract(self):
        self.assertEqual(run(29, 13), run(29, 13))
        with self.assertRaises(ValueError):
            message_success({0: Counter({1: 1})}, 0)
        with self.assertRaises(ValueError):
            grouped_counts({0: 1}, {})


if __name__ == "__main__":
    unittest.main()
