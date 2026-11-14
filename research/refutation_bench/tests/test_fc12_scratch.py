"""FC-12 typed local argument grammar, sorry-free proof trust boundary."""
from __future__ import annotations
import unittest
from research.refutation_bench.fc12 import scratch,core


SOURCE=("import Init\nnamespace Test\n"
        "theorem identity (P : Prop) (h : P) : P := by\n"
        "  exact h\n"
        "end Test\n")


class TypedScratchTests(unittest.TestCase):
    def test_registered_parameterized_tactics_and_no_source_injection(self):
        for s in ("exact h","rw [h]","rw [← h]","apply h",
                  "rcases h with ⟨fc12_left, fc12_right⟩",
                  "induction xs","simpa using h","omega","intro"):
            self.assertTrue(scratch.valid_script(s),s)
        for s in ("sorry","exact h\nsorry","rw [h]; sorry","exact test.x",
                  "run_tac", "exact (by sorry)"):
            self.assertFalse(scratch.valid_script(s),s)

    def test_parameter_tactic_in_final_proof_and_scratch_only_sorry(self):
        kernel=scratch.make_source(SOURCE,"identity",["exact h"],probe=False)
        self.assertIn("  exact h",kernel)
        self.assertNotIn("sorry",kernel)
        self.assertNotIn("FC12_AFTER",kernel)
        preview=scratch.make_source(SOURCE,"identity",["exact h"],probe=True)
        self.assertIn("all_goals sorry",preview)
        self.assertIn("FC12_AFTER_BEGIN",preview)

    def test_context_generator_requires_real_goal(self):
        with self.assertRaises(AssertionError):
            core.generate("this is not a printed Lean goal")
        sample="h : True\nx : List Nat\n⊢ True"
        g=core.generate(sample)
        self.assertIn("induction x",g["induction_list"])
        self.assertTrue(all(scratch.valid_script(s) for v in g.values() for s in v))

    def test_reference_prefix_is_not_future_reference_lemma(self):
        reference=("import Init\nnamespace Test\n"
            "theorem multistep (P : Prop) : P → P := by\n"
            "  intro h\n"
            "  exact h\n"
            "end Test\n")
        first=scratch.make_source(reference,"multistep",["intro"],original_stage=0,probe=True)
        second=scratch.make_source(reference,"multistep",["assumption"],original_stage=1,probe=True)
        self.assertNotIn("  exact h\n",first)
        self.assertNotIn("  exact h\n",second)
        self.assertIn("  intro h\n  assumption",second)


if __name__=="__main__":
    unittest.main()
