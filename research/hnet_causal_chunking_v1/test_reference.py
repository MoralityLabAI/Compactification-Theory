"""Independent brute-force reference checks for HC-01 finite certificates."""

import random
import unittest

from reference import (Machine, audit, behavioral_partition, distinguishing_suffix,
                       horizon_partition, shift_machine)


def independent_equivalence(machine):
    """Greatest fixed-point relation, independent of partition signatures."""
    pairs = {(x, y) for x in machine.states for y in machine.states
             if machine.output[x] == machine.output[y]}
    while True:
        smaller = {(x, y) for x, y in pairs
                   if all((machine.delta[(x, a)], machine.delta[(y, a)]) in pairs
                          for a in machine.alphabet)}
        if smaller == pairs:
            return smaller
        pairs = smaller


class ValidationTests(unittest.TestCase):
    def test_total_transition_required(self):
        with self.assertRaises(ValueError):
            Machine(("a", "b"), ("go",), {"a": "0", "b": "1"}, {("a", "go"): "b"})

    def test_duplicate_and_out_of_domain_rejected(self):
        with self.assertRaises(ValueError):
            Machine(("a", "a"), ("go",), {"a": "0"}, {("a", "go"): "a"})
        with self.assertRaises(ValueError):
            Machine(("a",), ("go",), {"a": "0"}, {("a", "go"): "b"})

    def test_encoding_domain_and_horizon_validation(self):
        with self.assertRaises(ValueError):
            audit(shift_machine(), {"00": "0"})
        with self.assertRaises(ValueError):
            horizon_partition(shift_machine(), -1)
        with self.assertRaises(ValueError):
            horizon_partition(shift_machine(), True)


class TheoryTests(unittest.TestCase):
    def test_intervention_increases_required_bits(self):
        passive = shift_machine(False)
        active = shift_machine(True)
        p = audit(passive, {s: s[0] for s in passive.states})
        a = audit(active, {s: s[0] for s in active.states})
        self.assertTrue(p["observation_sufficient"])
        self.assertTrue(p["transition_factorable"])
        self.assertEqual(p["lower_bound_bits_for_exact_future_observation"], 1)
        self.assertEqual(a["lower_bound_bits_for_exact_future_observation"], 2)
        self.assertFalse(a["observation_sufficient"])
        self.assertFalse(a["transition_factorable"])
        self.assertEqual(a["observation_collision"]["suffix"], ["shift"])

    def test_full_state_encoding_factors(self):
        machine = shift_machine(True)
        a = audit(machine, {s: s for s in machine.states})
        self.assertTrue(a["observation_sufficient"])
        self.assertTrue(a["transition_factorable"])
        self.assertEqual(a["encoding_codes"], 4)

    def test_finite_horizon_budget(self):
        machine = shift_machine(True)
        zero = horizon_partition(machine, 0)
        one = horizon_partition(machine, 1)
        self.assertEqual(len(set(zero.values())), 2)
        self.assertEqual(len(set(one.values())), 4)
        self.assertEqual(one, behavioral_partition(machine))

    def test_observational_sufficiency_does_not_imply_factorization(self):
        states = ("a", "b", "c")
        machine = Machine(states, ("go",), {s: "0" for s in states},
                          {("a", "go"): "c", ("b", "go"): "b", ("c", "go"): "a"})
        r = audit(machine, {"a": "0", "b": "0", "c": "1"})
        self.assertTrue(r["observation_sufficient"])
        self.assertFalse(r["transition_factorable"])
        self.assertEqual(r["factorization_failure"]["kind"], "transition")
        self.assertEqual(r["behavioral_classes"], 1)

    def test_unreachable_observation_is_not_silent(self):
        machine = shift_machine(True)
        with self.assertRaises(ValueError):
            machine.observe("wrong", ())
        with self.assertRaises(ValueError):
            machine.observe("00", ("wrong",))

    def test_independent_randomized_bisimulation(self):
        rng = random.Random(20261029)
        for _ in range(100):
            n = rng.randrange(2, 7)
            states = tuple(str(s) for s in range(n))
            alphabet = ("a", "b")
            outputs = {s: str(rng.randrange(3)) for s in states}
            transitions = {(s, a): rng.choice(states)
                           for s in states for a in alphabet}
            machine = Machine(states, alphabet, outputs, transitions)
            exact = independent_equivalence(machine)
            part = behavioral_partition(machine)
            for x in states:
                for y in states:
                    same = part[x] == part[y]
                    self.assertEqual(same, (x, y) in exact)
                    word = distinguishing_suffix(machine, x, y)
                    self.assertEqual(word is None, same)
                    if word is not None:
                        self.assertNotEqual(machine.observe(x, word), machine.observe(y, word))

    def test_shortest_witness_matches_exhaustive_strings(self):
        machine = shift_machine(True)
        from itertools import product
        for x in machine.states:
            for y in machine.states:
                witness = distinguishing_suffix(machine, x, y)
                first = None
                for k in range(4):
                    for word in product(machine.alphabet, repeat=k):
                        if machine.observe(x, word) != machine.observe(y, word):
                            first = word
                            break
                    if first is not None:
                        break
                self.assertEqual(witness, first)


if __name__ == "__main__":
    unittest.main()
