"""Exact arithmetic and anti-overfit tests for RRC-01."""
from __future__ import annotations

import unittest

try:
    from .rrc import (ResidueRule, euler_exponents, first_mismatch, gap_coefficients,
                      infer_periodic_rule, product_coefficients, run_experiment,
                      staircase_coefficients)
except ImportError:
    from rrc import (ResidueRule, euler_exponents, first_mismatch, gap_coefficients,
                     infer_periodic_rule, product_coefficients, run_experiment,
                     staircase_coefficients)


def brute_gap(n: int, *, minimum_part: int = 1, gap: int = 2) -> int:
    def visit(remaining: int, next_part: int) -> int:
        if remaining == 0:
            return 1
        return sum(visit(remaining - part, part + gap)
                   for part in range(next_part, remaining + 1))
    return visit(n, minimum_part)


def brute_residue(n: int, rule: ResidueRule) -> int:
    def visit(remaining: int, next_part: int) -> int:
        if remaining == 0:
            return 1
        return sum(visit(remaining - part, part) for part in range(next_part, remaining + 1)
                   if part % rule.modulus in rule.residues)
    return visit(n, 1)


class RogersRamanujanTests(unittest.TestCase):
    def test_enumerated_local_oracle(self):
        for gap in (1, 2, 3):
            for minimum_part in (1, 2, 3):
                coefficients = gap_coefficients(17, gap=gap, minimum_part=minimum_part)
                for n in range(18):
                    self.assertEqual(coefficients[n], brute_gap(n, gap=gap, minimum_part=minimum_part),
                                     (gap, minimum_part, n))

    def test_staircase_uses_independent_qseries(self):
        for minimum_part in (1, 2, 3):
            self.assertEqual(gap_coefficients(130, minimum_part=minimum_part),
                             staircase_coefficients(130, minimum_part=minimum_part))

    def test_exponents_exact_rr_masks(self):
        for s, residues in ((1, (1, 4)), (2, (2, 3))):
            exponents = euler_exponents(gap_coefficients(90, minimum_part=s))
            self.assertEqual(exponents[0], 0)
            self.assertEqual(exponents[1:], [int(n % 5 in residues) for n in range(1, 91)])
            rule = infer_periodic_rule(gap_coefficients(64, minimum_part=s), modulus_cap=12)
            self.assertEqual(rule, ResidueRule(5, residues))
            self.assertEqual(rule.code_bits, 10)

    def test_independent_product_enumeration(self):
        for rule in (ResidueRule(5, (1, 4)), ResidueRule(5, (2, 3))):
            values = product_coefficients(16, rule)
            self.assertEqual(values, [brute_residue(n, rule) for n in range(17)])
        self.assertEqual(product_coefficients(6, ResidueRule(5, (1, 4)))[6], 3)

    def test_train_only_and_holdout(self):
        result = run_experiment()
        self.assertEqual(result["parameters"]["train_degree"], 64)
        for family in result["families"]:
            self.assertIsNone(family["training_mismatch"])
            self.assertIsNone(family["heldout_first_mismatch"])
            self.assertEqual(family["heldout_coefficient_count"], 192)
        self.assertIsNotNone(result["negative_controls"]["rotated_rule_heldout_first_mismatch"])

    def test_greater_gap_is_rejected_or_disproven(self):
        train = gap_coefficients(64, gap=3)
        rule = infer_periodic_rule(train, modulus_cap=12)
        if rule is not None:
            full = gap_coefficients(256, gap=3)
            self.assertIsNotNone(first_mismatch(full, product_coefficients(256, rule), start=65))

    def test_fitting_uses_only_training_prefix(self):
        true = gap_coefficients(90)
        corrupted = true.copy()
        corrupted[80] += 500
        train_rule = infer_periodic_rule(true[:65])
        self.assertEqual(train_rule, infer_periodic_rule(corrupted[:65]))
        predicted = product_coefficients(90, train_rule)
        self.assertEqual(first_mismatch(predicted, corrupted, start=65), 80)

    def test_rejects_inadmissible_inputs(self):
        with self.assertRaises(ValueError):
            gap_coefficients(-1)
        with self.assertRaises(ValueError):
            gap_coefficients(5, gap=0)
        with self.assertRaises(ValueError):
            euler_exponents([2, 1])
        with self.assertRaises(ValueError):
            infer_periodic_rule([1, 1], min_observations=1)
        with self.assertRaises(ValueError):
            product_coefficients(5, ResidueRule(5, (7,)))
        with self.assertRaises(ValueError):
            run_experiment(train_degree=10)


if __name__ == "__main__":
    unittest.main()
