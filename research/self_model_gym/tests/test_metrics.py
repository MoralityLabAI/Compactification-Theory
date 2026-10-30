import math
import unittest

from research.self_model_gym.metrics import behavioral_metrics, trace_metrics


class TestMetrics(unittest.TestCase):
    def test_behavioral_repetition(self):
        measured = behavioral_metrics("Focus focus focus.")
        self.assertEqual(measured["adjacent_repeat_fraction"], 1)
        self.assertEqual(measured["normalized_word_entropy"], 0)
        self.assertEqual(behavioral_metrics("")["word_count"], 0)

    def test_hidden_cycle_higher_than_linear(self):
        cycle = [[1.0, 0.0] if i % 2 else [0.0, 1.0] for i in range(16)]
        linear = [[float(i), float(i*i)] for i in range(16)]
        a = trace_metrics(cycle)
        b = trace_metrics(linear)
        self.assertGreater(a["nonlocal_recurrence_rate"], b["nonlocal_recurrence_rate"])
        self.assertEqual(a["trace_tokens"], 16)

    def test_missing_or_short_trace_is_unscored(self):
        self.assertIsNone(trace_metrics(None)["nonlocal_recurrence_rate"])
        self.assertIsNone(trace_metrics([[1, 2], [3, 4]])["nonlocal_recurrence_rate"])

    def test_trace_input_validation(self):
        for trace in ([[1, 2], [3]], [[math.inf, 1.0]], [[True, 1.0]],
                      [["s", 1.0]], [[0.0]] * 65):
            with self.assertRaises(ValueError):
                trace_metrics(trace)


if __name__ == "__main__":
    unittest.main()
