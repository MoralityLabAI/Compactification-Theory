"""Standard-library-only regression tests for the CB01 finite companion oracle."""
import unittest

try:
    from .runner import (ACTIONS, TRANSITIONS, Step, action_words, classes, memory_bits,
                         receipt, shortest_witness, signature, timed_trace,
                         validate_transitions)
except ImportError:
    from runner import (ACTIONS, TRANSITIONS, Step, action_words, classes, memory_bits,
                        receipt, shortest_witness, signature, timed_trace,
                        validate_transitions)


class CanonizationBenchTests(unittest.TestCase):
    def test_equivalence_classes_and_bit_count(self):
        self.assertEqual(classes(3), [[0, 2], [1, 3], [4], [5]])
        self.assertEqual(memory_bits(len(classes(3))), 2)
        self.assertEqual(signature(0, 3), signature(2, 3))
        self.assertEqual(signature(1, 3), signature(3, 3))

    def test_delayed_distinguishing_word(self):
        self.assertIsNone(shortest_witness(0, 4, 2))
        self.assertEqual(shortest_witness(0, 4, 3), ["switch", "read"])

    def test_action_refinement_and_duration_observability(self):
        self.assertEqual(len(classes(3, ("read",))), 3)
        self.assertEqual(len(classes(3, ACTIONS)), 4)
        mutated = dict(TRANSITIONS)
        mutated[(2, "read")] = Step(2, "idle", 0, 2)
        self.assertEqual(shortest_witness(0, 2, 3, transitions=mutated), ["read"])

    def test_finite_closure_of_enumeration(self):
        words = list(action_words(ACTIONS, 3))
        self.assertEqual(len(words), 15)
        for word in words:
            self.assertEqual(timed_trace(0, word, 3), timed_trace(2, word, 3))

    def test_invalid_duration_and_budget(self):
        bad = {(0, "read"): Step(0, "idle", 0, 0)}
        with self.assertRaises(ValueError):
            validate_transitions(bad)
        with self.assertRaises(ValueError):
            timed_trace(0, (), -1)
        with self.assertRaises(ValueError):
            memory_bits(0)

    def test_reproducible_receipt_and_source_counts(self):
        a, b = receipt(), receipt()
        self.assertEqual(a, b)
        self.assertTrue(all(a["checks"].values()))
        self.assertEqual(a["source_metrics"]["canonical_call_sites"],
                         {"consumers": 2, "heldout": 3})


if __name__ == "__main__":
    unittest.main()
