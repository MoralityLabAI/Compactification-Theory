"""NOR-23 registered small-population combinatorics and public/target isolation."""
from __future__ import annotations

from itertools import combinations
from math import comb
import unittest

from research.nor23_coordinate_audit import experiment as n
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus


class NOR23CoordinateAuditTests(unittest.TestCase):
    def test_every_small_population_exact_hypergeometric_count(self):
        for m in range(1, 9):
            indices = range(m)
            for r in range(0, m + 1):
                for k in range(0, m + 1):
                    total = detected = 0
                    for altered in combinations(indices, r):
                        attacked = set(altered)
                        for inspected in combinations(indices, k):
                            total += 1
                            if attacked.intersection(inspected):
                                detected += 1
                    self.assertEqual(total, comb(m, r) * comb(m, k))
                    self.assertAlmostEqual(
                        detected / total, n.detection_probability(m, r, k),
                        places=12, msg=(m, r, k))

    def test_trusted_oracle_rejects_any_inspected_nonzero_delta(self):
        real = (17, 23, 31, 47, 53, 71)
        proposed = real
        attacked = frozenset((1, 5))
        for delta in n.MAGNITUDES:
            self.assertFalse(n.trusted_witness_accepts(
                proposed, real, (0, 1), attacked, delta))
            self.assertFalse(n.trusted_witness_accepts(
                proposed, real, (3, 5), attacked, delta))
            self.assertTrue(n.trusted_witness_accepts(
                proposed, real, (0, 2, 3, 4), attacked, delta))
        self.assertTrue(n.trusted_witness_accepts(
            proposed, real, (), attacked, 1))
        self.assertTrue(n.trusted_witness_accepts(
            proposed, real, tuple(range(6)), frozenset(), 1))
        with self.assertRaises(ValueError):
            n.trusted_witness_accepts(proposed, real[:-1], (1,), attacked, 1)

    def test_full_sample_detects_every_nonzero_sparse_alteration(self):
        source = (1, 8, -9, 4, 17)
        for r in range(1, len(source) + 1):
            for attacked in combinations(range(len(source)), r):
                self.assertFalse(n.trusted_witness_accepts(
                    source, source, range(len(source)), set(attacked), 1))
        self.assertEqual(n.detection_probability(5, 1, 5), 1)
        self.assertEqual(n.detection_probability(5, 4, 0), 0)

    def test_informed_attacker_wins_on_every_unobserved_position(self):
        for m in range(1, 8):
            for k in range(m + 1):
                for r in range(1, m - k + 1):
                    for inspected in combinations(range(m), k):
                        attacked = n.informed_positions(inspected, m, r)
                        self.assertEqual(len(attacked), r)
                        self.assertFalse(attacked.intersection(inspected))
                        self.assertTrue(n.trusted_witness_accepts(
                            tuple(range(m)), tuple(range(m)),
                            inspected, attacked, 1))

    def test_random_stream_reproducible_and_validated(self):
        d = "0" * 64
        for k in n.BUDGETS:
            coords = n.challenge_positions(d, 769, k, 11)
            self.assertEqual(coords, n.challenge_positions(d, 769, k, 11))
            self.assertEqual(len(coords), k)
            self.assertEqual(len(set(coords)), k)
            self.assertEqual(tuple(sorted(coords)), coords)
        self.assertEqual(len(n.blind_positions(d, 769, 16)), 16)
        with self.assertRaises(ValueError):
            n.challenge_positions(d, 769, 3, 0)
        with self.assertRaises(ValueError):
            n.positions(n.ATTACK_SEED, d, 3, 4)
        with self.assertRaises(ValueError):
            n.informed_positions((0, 1, 2), 3, 1)

    def test_fresh_source_fixture_does_not_reveal_target(self):
        row = n.generate_source(0, 512, n.FAMILIES[0], set())
        self.assertEqual(row["source_seed"], n.source_seed(0, 512, n.FAMILIES[0]))
        self.assertTrue(row["original_public_accept_mask"] > 0)
        self.assertNotIn("future", row)
        self.assertNotIn("program", row)
        self.assertNotIn("true_target", row)
        public, header = n.public_case(row)
        self.assertIsNone(public.program)
        self.assertEqual(public.future, ())
        proposal = nor21.certify(public, header, 512, 768, "stream16")
        self.assertEqual(proposal["mask"], row["original_public_accept_mask"])
        self.assertTrue(proposal["unanimous"])
        chosen = proposal["chosen_class"]
        predicted = corpus.execute(public.query, nor21.program(header, chosen))
        sealed = n.sealed_future_after_choice(row, predicted)
        self.assertEqual(sealed, predicted)
        # Simulate a future override after public decision freezing.
        altered = list(sealed)
        altered[501] += n.MAGNITUDES[1]
        self.assertEqual(row["full_public_input_sha256"],
                         nor21.public_digest(public, 512))
        self.assertNotEqual(altered, list(predicted))
        self.assertFalse(n.trusted_witness_accepts(
            predicted, sealed, (501,), frozenset((501,)), n.MAGNITUDES[1]))


if __name__ == "__main__":
    unittest.main()
