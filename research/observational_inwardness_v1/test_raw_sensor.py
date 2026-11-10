"""OIC-08/09: finite exact raw-state error/erasure and parity-latch tests."""
import unittest
from itertools import product

from model import trajectory
from robust import corruption_cost
from raw_sensor import (
    best_fixed_word, even_latching_word, even_maximum_raw_distance,
    even_minimum_raw_horizon, minimum_raw_distance, raw_codebook,
    raw_collision, raw_to_audit, reference, terminal_separated,
)


class RawSensorTests(unittest.TestCase):
    def test_clean_raw_observations_reconstruct_audit(self):
        for n in range(2, 8):
            for family in ("fixed", "rotating"):
                for z in range(n):
                    word = (1, 0, 1, 1, 0, 1, 1, 1)
                    trace = trajectory(n, family, z, word)
                    audit = raw_to_audit(trace)
                    expected = tuple(
                        int(bool(u) and (z + (t if family == "rotating" else 0)) % n == 0)
                        for t, u in enumerate(word)
                    )
                    self.assertEqual(audit, expected)

    def test_single_raw_state_error_affects_adjacent_audits(self):
        clean = trajectory(4, "rotating", 0, (1,) * 8)
        original = raw_to_audit(clean)
        for i in range(len(clean)):
            received = list(clean)
            received[i] ^= 1
            inferred = raw_to_audit(received)
            mismatches = {j for j in range(len(original)) if inferred[j] != original[j]}
            adjacent = {j for j in (i - 1, i) if 0 <= j < len(original)}
            self.assertEqual(mismatches, adjacent)
            erased = list(clean)
            erased[i] = None
            inferred_erased = raw_to_audit(erased)
            self.assertEqual({j for j, x in enumerate(inferred_erased) if x is None}, adjacent)

    def test_even_latch_distance_formula(self):
        for n in (2, 4, 6, 8):
            for t in range(n, n + 9):
                word = even_latching_word(n, t)
                self.assertEqual(len(word), t)
                self.assertEqual(sum(word), n)
                self.assertEqual(minimum_raw_distance(n, word), t - n + 1)
                self.assertEqual(even_maximum_raw_distance(n, t), t - n + 1)
                self.assertTrue(terminal_separated(n, word))
            for t in range(n):
                self.assertEqual(even_maximum_raw_distance(n, t), 0)
                self.assertEqual(minimum_raw_distance(n, (1,) * t), 0)

    def test_even_optimum_exhaustive_fixed_schedules(self):
        for n in (2, 4, 6):
            for t in range(0, 10):
                actual, word = best_fixed_word(n, t)
                self.assertEqual(actual, even_maximum_raw_distance(n, t))
                self.assertEqual(minimum_raw_distance(n, word), actual)

    def test_even_error_erasures_exact_bound_and_collision_receipts(self):
        for n in (2, 4, 6):
            for e in range(3):
                for r in range(3):
                    t = even_minimum_raw_horizon(n, e, r)
                    before = even_latching_word(n, t - 1) if t > n else (1,) * (n - 1)
                    at = even_latching_word(n, t)
                    self.assertEqual(minimum_raw_distance(n, at), 2 * e + r + 1)
                    witness = raw_collision(n, before, e, r)
                    self.assertIsNotNone(witness)
                    left, right, receipt = witness
                    self.assertLessEqual(corruption_cost(left, receipt)[0], e)
                    self.assertLessEqual(corruption_cost(right, receipt)[0], e)
                    self.assertLessEqual(corruption_cost(left, receipt)[1], r)
                    self.assertLessEqual(corruption_cost(right, receipt)[1], r)
                    self.assertIsNone(raw_collision(n, at, e, r))

    def test_n_probes_necessary_for_fixed_schedule(self):
        for n in (2, 3, 4):
            for t in range(0, 9):
                for word in product((0, 1), repeat=t):
                    if sum(word) < n:
                        self.assertEqual(minimum_raw_distance(n, word), 0)

    def test_odd_phase_parity_obstruction_terminal_readout(self):
        for n in (3, 5):
            for t in range(0, 9):
                for word in product((0, 1), repeat=t):
                    self.assertFalse(terminal_separated(n, word))

    def test_odd_family_horizon_can_exceed_even_formula(self):
        # Exact exhaustive examples, for fixed schedules only.
        for t, maximum in ((3, 1), (4, 1), (5, 2), (6, 2), (7, 3)):
            self.assertEqual(best_fixed_word(3, t)[0], maximum)
        self.assertEqual(best_fixed_word(5, 7)[0], 3)
        self.assertEqual(best_fixed_word(3, 10)[0], 5)

    def test_reference_and_validation(self):
        data = reference()
        self.assertEqual(len(data["rows"]), 20)
        self.assertEqual(len(data["odd_examples"]), 3)
        self.assertTrue(all(row["collision_before"] and row["separated_at_threshold"] for row in data["rows"]))
        for n, horizon in ((3, 4), (4, 3), (1, 5)):
            with self.assertRaises(ValueError):
                even_latching_word(n, horizon)
        for n in (1, 3):
            with self.assertRaises(ValueError):
                even_maximum_raw_distance(n, 5)
        for n, e, r in ((4, -1, 0), (4, 0, -1), (3, 1, 0)):
            with self.assertRaises(ValueError):
                even_minimum_raw_horizon(n, e, r)
        with self.assertRaises(ValueError):
            raw_to_audit([])
        with self.assertRaises(ValueError):
            raw_codebook(4, "fixed", (1, 2))


if __name__ == "__main__":
    unittest.main()
