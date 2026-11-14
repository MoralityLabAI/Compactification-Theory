"""FC-10 posthoc actual external Lean source instrumentation safety checks."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest

from research.refutation_bench.fc10 import imported_probe as probe


class ImportedProofProbeTests(unittest.TestCase):
    def test_exact_source_revision_and_theorem_targets(self):
        self.assertEqual(probe.MATHCAN_COMMIT,
            "c84e2025334e9a967c1297425a490f98d5b3d189")
        self.assertEqual(probe.LEAN_TOOLCHAIN,"leanprover/lean4:v4.35.0")
        self.assertEqual(sum(len(x) for x in probe.FILES.values()),7)

    def test_instrument_does_not_replace_original_tactic(self):
        source=(
            "import Init\n"
            "namespace Trial\n"
            "theorem one (n : Nat) : n = n := by\n"
            "  rfl\n"
            "theorem two (n : Nat) : n = n := by rfl\n"
            "end Trial\n"
        )
        output=probe.instrument(source,("one","two"),"Trial")
        self.assertIn("import Lean\nimport Init",output)
        self.assertIn("FC10_IMPORTED_BEGIN:Trial:one",output)
        self.assertIn("FC10_IMPORTED_END:Trial:two",output)
        self.assertIn("  trace_state\n",output)
        self.assertIn("  rfl\n",output)
        self.assertEqual(output.count("  rfl"),2)

    def test_missing_theorem_refused_instead_of_guessing(self):
        with self.assertRaises(AssertionError):
            probe.instrument("import Init\n",("nonexistent",),"Fake")

    def test_extractor_needs_exact_markers_and_real_goal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for path,names in probe.FILES.items():
                tag=Path(path).stem
                log="".join(
                   f"FC10_IMPORTED_BEGIN:{tag}:{name}\n"
                   f"testHyp : True\n⊢ True\n"
                   f"FC10_IMPORTED_END:{tag}:{name}\n" for name in names)
                (root/(tag+".log")).write_text(log)
            report=probe.extract(root)
            self.assertEqual(report["real_imported_goal_states"],7)
            self.assertTrue((root/"imported_goal_states.jsonl").exists())
            first=Path(next(iter(probe.FILES))).stem
            (root/(first+".log")).write_text(
                "FC10_IMPORTED_BEGIN:"+first+":one\nno goal\n")
            with self.assertRaises(AssertionError):
                probe.extract(root)


if __name__=="__main__":
    unittest.main()
