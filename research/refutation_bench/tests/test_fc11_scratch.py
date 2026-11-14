"""FC-11 independently checked scratch action and trusted final proof generation."""
from __future__ import annotations
import unittest
from research.refutation_bench.fc11 import source_core as core,scratch


EXAMPLE=("import Init\n"
         "namespace FC11Scratch\n"
         "theorem simple (P Q : Prop) (hp : P) : P → P := by\n"
         "  intro h\n"
         "  exact h\n"
         "end FC11Scratch\n")


class FC11ScratchTests(unittest.TestCase):
    def test_training_prefix_stage_does_not_include_future_reference(self):
        first=scratch.make_source(EXAMPLE,"simple",["intro"],original_stage=0,probe=True)
        second=scratch.make_source(EXAMPLE,"simple",["assumption"],original_stage=1,probe=True)
        self.assertIn("FC11_AFTER_BEGIN",first)
        self.assertIn("all_goals sorry",first)
        self.assertIn("  intro h\n  assumption",second)
        self.assertNotIn("  exact h",second)

    def test_final_candidate_never_includes_sorry(self):
        source=scratch.make_source(EXAMPLE,"simple",["intro","assumption"],probe=False)
        self.assertNotIn("sorry",source)
        self.assertNotIn("FC11_AFTER",source)
        self.assertIn("end FC11Scratch",source)
        self.assertIn("  intro\n  assumption",source)

    def test_reject_unregistered_action_without_proof_repair(self):
        with self.assertRaises(ValueError):
            scratch.make_source(EXAMPLE,"simple",["sorry"],probe=False)
        with self.assertRaises(ValueError):
            scratch.make_source(EXAMPLE,"simple",["exact hp"],probe=False)

    def test_undocumented_training_stage_rejected(self):
        with self.assertRaises(ValueError):
            scratch.make_source(EXAMPLE,"simple",["intro"],original_stage=10,probe=True)

    def test_compiler_state_extraction_is_strict(self):
        log="FC11_AFTER_BEGIN\nP : Prop\n⊢ P\nFC11_AFTER_END"
        self.assertEqual(scratch._extract_after_states(log),["P : Prop\n⊢ P"])
        with self.assertRaises(AssertionError):
            scratch._extract_after_states("FC11_AFTER_BEGIN\nnot a goal\nFC11_AFTER_END")
        nested=("FC11_AFTER_BEGIN\nFC11_AFTER_BEGIN\n"
                "P : Prop\n⊢ P\nQ : Prop\n⊢ Q\n"
                "FC11_AFTER_END\nFC11_AFTER_END")
        extracted=scratch._extract_after_states(nested)
        self.assertEqual(len(extracted),1)
        self.assertEqual(extracted[0].count("⊢"),2)
        with self.assertRaises(AssertionError):
            scratch._extract_after_states("FC11_AFTER_BEGIN\n⊢ P")


if __name__=="__main__":
    unittest.main()
