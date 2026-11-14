"""FC-12 fresh original MathCan theorem selection and source guards."""
from __future__ import annotations
import unittest
from research.refutation_bench.fc12 import core,train_data,test_data


class FC12HeldoutSourceTests(unittest.TestCase):
    def test_four_unused_mathcan_modules_and_sixteen_orig_original_theorems(self):
        test_data.check_selection()
        self.assertEqual(len(test_data.TEST_MODULES),4)
        self.assertEqual(sum(len(x) for x in test_data.TEST_MODULES.values()),16)
        self.assertTrue(set(test_data.TRAIN_MODULES).isdisjoint(test_data.TEST_MODULES))

    def test_heldout_family_names_were_not_training(self):
        self.assertNotIn("MathCan/Hodge.lean",train_data.TRAIN_MODULES)
        self.assertNotIn("MathCan/TOELorentzBivector.lean",train_data.TRAIN_MODULES)
        self.assertNotIn("MathCan/TOEDressedMatter.lean",train_data.TRAIN_MODULES)
        self.assertNotIn("MathCan/TOETensorCompletion.lean",train_data.TRAIN_MODULES)

    def test_train_only_commit_precedes_test(self):
        self.assertEqual(core.SOURCE_COMMIT,
                         "c84e2025334e9a967c1297425a490f98d5b3d189")
        self.assertEqual(core.TOOLCHAIN,"leanprover/lean4:v4.35.0")
        self.assertEqual(core.MAX_ARGUMENTS,4)
        self.assertEqual(core.MAX_TRANSITIONS,3)
        self.assertEqual(len(core.FAMILIES),12)

    def test_source_theorem_names_distinct(self):
        selected=[(file,theorem) for file,names in test_data.TEST_MODULES.items()
                  for theorem in names]
        self.assertEqual(len(set(selected)),16)
        self.assertTrue(all(f.startswith("MathCan/") and f.endswith(".lean") for f,_ in selected))


if __name__=="__main__":
    unittest.main()
