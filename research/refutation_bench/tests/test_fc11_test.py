"""FC-11 original research theorem family and test trace audit."""
from __future__ import annotations
from collections import Counter
import unittest
from research.refutation_bench.fc11 import source_core,train_data,test_data


class ImportedTestSource(unittest.TestCase):
    def test_disjoint_original_mathcan_modules(self):
        test_data.assert_family_disjoint()
        self.assertEqual(len(test_data.TEST_SOURCE),3)
        self.assertEqual(sum(len(x) for x in test_data.TEST_SOURCE.values()),14)
        self.assertEqual(len(train_data.TRAIN_SOURCE),4)
        self.assertFalse(set(train_data.TRAIN_SOURCE)&set(test_data.TEST_SOURCE))

    def test_original_mathcan_commit_is_frozen(self):
        self.assertEqual(source_core.SOURCE_COMMIT,
                         "c84e2025334e9a967c1297425a490f98d5b3d189")
        self.assertEqual(source_core.TOOLCHAIN,"leanprover/lean4:v4.35.0")
        self.assertEqual(source_core.MAX_TACTIC_STEPS,3)

    def test_heldout_source_proof_tactics_never_used_as_student_inputs(self):
        for name,targets in test_data.TEST_SOURCE.items():
            self.assertTrue(name.startswith("MathCan/"))
            self.assertTrue(name.endswith(".lean"))
            self.assertTrue(targets)
        source="import Init\ntheorem demo (p : Prop) (h : p) : p := by\n  exact h\n"
        code,stages=source_core.instrument(source,"MathCan/Demo.lean",("demo",))
        self.assertEqual(len(stages),1)
        self.assertIn("  exact h",code)
        self.assertIn("trace_state",code)
        self.assertFalse("sorry" in code)

    def test_distinct_goal_markers_across_modules(self):
        labels=[]
        for path,targets in test_data.TEST_SOURCE.items():
            stem=path.rsplit("/",1)[-1].split(".",1)[0]
            for target in targets:
                labels.append(stem+"."+target+".0")
        self.assertEqual(len(labels),len(set(labels)))

    def test_reference_tactic_bank_fixed_before_heldout_source(self):
        self.assertEqual(len(source_core.TACTICS),8)
        self.assertEqual(len(set(source_core.TACTICS)),8)
        self.assertIn("aesop",source_core.TACTICS)
        self.assertNotIn("sorry",source_core.TACTICS)


if __name__=="__main__":
    unittest.main()
