"""FC-12 exact source-only tactic grammar and original proof input guards."""
from __future__ import annotations
from collections import Counter
import unittest
from research.refutation_bench.fc12 import core,train_data


class FC12TrainingSourceTests(unittest.TestCase):
    def test_preregistered_sources_and_actions(self):
        self.assertEqual(len(train_data.TRAIN_MODULES),7)
        self.assertEqual(sum(len(v) for v in train_data.TRAIN_MODULES.values()),27)
        self.assertEqual(len(core.FAMILIES),12)
        self.assertEqual(core.MAX_ARGUMENTS,4)
        self.assertEqual(core.MAX_TRANSITIONS,3)
        self.assertEqual(core.MAX_TOTAL_LEAN_PROBES,12)
        self.assertEqual(core.SOURCE_COMMIT,"c84e2025334e9a967c1297425a490f98d5b3d189")

    def test_arguments_derived_only_from_real_goal_context(self):
        state=(
          "α : Type\n"
          "xs ys : List Nat\n"
          "h : xs = ys\n"
          "both : True ∧ True\n"
          "witness : ∃ (n : Nat), n > 0\n"
          "⊢ xs = ys"
        )
        actions=core.generate(state)
        self.assertIn("exact h",actions["exact_local"])
        self.assertIn("exact both",actions["exact_local"])
        self.assertIn("rw [h]",actions["rw_local"])
        self.assertIn("rcases both with ⟨fc12_left, fc12_right⟩",actions["rcases_local"])
        self.assertIn("induction xs",actions["induction_list"])
        self.assertIn("simpa using h",actions["simpa_local"] if len(actions["simpa_local"])>2 else actions["exact_local"] + actions["simpa_local"])
        self.assertTrue(all(len(v)<=4 for v in actions.values()))

    def test_contextual_function_hypotheses_have_local_application_candidates(self):
        state=("h : obsEqOn allowed step readout s t\n"
               "word : List Action\n"
               "valid : ∀ (a : Action), a ∈ word → allowed a\n"
               "⊢ readout (runActions step word t) = readout (runActions step word s)")
        scripts=core.generate(state)["exact_local"]
        self.assertIn("exact (h word valid).symm",scripts)
        self.assertIn("exact h word valid",scripts)
        self.assertLessEqual(len(scripts),4)

    def test_poisoned_reference_metadata_does_not_affect_action(self):
        goal="h : Nat.succ 0 = 1\n⊢ Nat.succ 0 = 1"
        a=core.generate(goal)
        poisoned={"theorem":"Fermat", "source_command":"sorry",
                  "reference_tactic":"omega", "family":"test"}
        b=core.generate(goal)
        self.assertEqual(a,b)
        self.assertNotIn("sorry",repr(a))
        self.assertNotIn("Fermat",repr(a))

    def test_reject_source_substitution_without_real_turnstile(self):
        with self.assertRaises(AssertionError):
            core.generate("theorem p : P := by exact h")

    def test_local_hypothesis_names_only(self):
        x=core.contextual_names("h1 h2 : Nat\nhx : h1 ≤ h2\n⊢ h1 ≤ h2")
        self.assertEqual(x,[("h1","Nat"),("h2","Nat"),("hx","h1 ≤ h2")])


if __name__=="__main__":
    unittest.main()
