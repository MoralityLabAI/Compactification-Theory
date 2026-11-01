"""Independent exact checks of the finite-governance example."""
from itertools import product
import unittest

from finite_model import (
    ACTIONS, FiniteSystem, GovernanceState, demonstrate, governance_system,
    minimal_partition, quotient_witness, step,
)


class TestFiniteControlQuotients(unittest.TestCase):
    def test_total_system(self):
        system = governance_system()
        system.validate()
        self.assertEqual(len(system.states), 8)
        self.assertEqual(len(system.transitions), 8 * len(ACTIONS))

    def test_revocation_always_disables_approval(self):
        for s in governance_system().states:
            revoked = step(s, "revoke_mandate")
            self.assertFalse(revoked.mandate)
            self.assertNotEqual("approve", governance_system().observations[revoked])

    def test_unregistered_action_rejected(self):
        with self.assertRaises(ValueError):
            step(GovernanceState(True, True, True), "attack")

    def test_bad_quotient_exposes_obstruction(self):
        system = governance_system()
        coarse = {s: system.observations[s] for s in system.states}
        witness = quotient_witness(system, coarse)
        self.assertIsNotNone(witness)
        self.assertEqual(witness[0], "successor")
        left, right, action = witness[1:]
        self.assertEqual(coarse[left], coarse[right])
        self.assertNotEqual(coarse[step(left, action)], coarse[step(right, action)])

    def test_exact_full_quotient(self):
        system = governance_system()
        classes = minimal_partition(system)
        self.assertEqual(len(set(classes)), 8)
        q = dict(zip(system.states, classes))
        self.assertIsNone(quotient_witness(system, q))

    def test_intervention_alphabet_changes_minimal_quotient(self):
        small = governance_system(("revoke_mandate",))
        self.assertEqual(len(set(minimal_partition(small))), 2)

    def test_exact_quotient_preserves_all_traces_through_horizon_three(self):
        system = governance_system()
        q = dict(zip(system.states, minimal_partition(system)))
        # Verify the consequence of the local quotient conditions on finite traces.
        for left in system.states:
            for right in system.states:
                if q[left] == q[right]:
                    for horizon in range(4):
                        for word in product(system.actions, repeat=horizon):
                            l, r = left, right
                            self.assertEqual(system.observations[l], system.observations[r])
                            for a in word:
                                l = step(l, a)
                                r = step(r, a)
                                self.assertEqual(system.observations[l], system.observations[r])

    def test_reject_incomplete_transitions(self):
        system = governance_system()
        transitions = dict(system.transitions)
        transitions.pop(next(iter(transitions)))
        broken = FiniteSystem(system.states, system.actions, system.observations, transitions)
        with self.assertRaises(ValueError):
            broken.validate()

    def test_reject_missing_observations(self):
        system = governance_system()
        observations = dict(system.observations)
        observations.pop(next(iter(observations)))
        broken = FiniteSystem(system.states, system.actions, observations, system.transitions)
        with self.assertRaises(ValueError):
            broken.validate()

    def test_report_scope(self):
        result = demonstrate()
        self.assertEqual(result["status"], "verified_finite_reference_only")
        self.assertTrue(result["approval_only_fails_quotient"])
        self.assertTrue(result["no_model_calls_or_war_simulation"])


if __name__ == "__main__":
    unittest.main()
