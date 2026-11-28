import unittest

from research.cog10_locality_bound.run import (
    witness, exact_oracle_error, metric, pair, topology, neighbors, ancestor_bits,
)


class LocalityTests(unittest.TestCase):
    def test_witnesses(self):
        for d in range(65):
            self.assertEqual(witness(d)["distinguished_at"], d+1)

    def test_exact_min_error(self):
        counts = {0: [3, 2], 1: [0, 4], 2: [1, 0]}
        self.assertEqual(exact_oracle_error(counts), 2)
        self.assertEqual(metric(counts, {0: [3, 2], 1: [0, 4]})[
            "train_unseen_signature_examples"], 1)

    def test_first_rule_has_opposite_labels(self):
        a, b = pair(4)
        self.assertEqual(a["rules"], b["rules"])
        self.assertEqual(a["facts"], b["facts"])
        self.assertTrue(ancestor_bits(a, neighbors(topology(a)))[0])
        self.assertFalse(ancestor_bits(b, neighbors(topology(b)))[0])


if __name__ == "__main__":
    unittest.main()
