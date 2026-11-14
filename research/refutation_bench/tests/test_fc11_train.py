"""FC-11 immutable MathCan import, tactic marker and source-isolation tests."""
from __future__ import annotations
from collections import Counter
import unittest
from research.refutation_bench.fc11 import source_core as core,train_data


class FC11TrainTests(unittest.TestCase):
    def test_provenance_and_train_only_manifest(self):
        self.assertEqual(core.SOURCE_COMMIT,
            "c84e2025334e9a967c1297425a490f98d5b3d189")
        self.assertEqual(core.TOOLCHAIN,"leanprover/lean4:v4.35.0")
        self.assertEqual(len(core.TACTICS),8)
        self.assertEqual(sum(map(len,train_data.TRAIN_SOURCE.values())),16)
        self.assertEqual(len(train_data.TRAIN_SOURCE),4)
        self.assertNotIn("Intervention",repr(train_data.TRAIN_SOURCE))
        self.assertNotIn("HookState",repr(train_data.TRAIN_SOURCE))
        self.assertNotIn("GraphAuthorization",repr(train_data.TRAIN_SOURCE))

    def test_multistep_source_instrumentation_preserves_original_tactics(self):
        original=("import Init\nnamespace Example\n"
            "theorem myProof (p : Prop) (hp : p) : p := by\n"
            "  have h : p := hp\n"
            "  exact h\n"
            "end Example\n")
        inst,manifest=core.instrument(original,"MathCan/TestProof.lean",("myProof",))
        self.assertEqual(len(manifest),2)
        self.assertEqual(inst.count("  trace_state"),2)
        self.assertIn("have h : p := hp",inst)
        self.assertIn("exact h",inst)
        self.assertIn("FC11_BEGIN:TestProof.myProof.0",inst)

    def test_inline_proof_is_retained_after_instrumentation(self):
        original=("import Init\nnamespace Test\n"
            "theorem proof (p : Prop) (h : p) : p := by exact h\n"
            "end Test\n")
        inst,rows=core.instrument(original,"MathCan/Inline.lean",("proof",))
        self.assertEqual(len(rows),1)
        self.assertIn("  exact h\n",inst)
        self.assertIn("trace_state",inst)
        self.assertNotIn("sorry",inst)

    def test_parser_requires_actual_compiler_trace_goals(self):
        item={"stage_id":"Example.proof.0"}
        log="FC11_BEGIN:Example.proof.0\nh : True\n⊢ True\nFC11_END:Example.proof.0"
        rows=core.parse_compiler_trace(log,[item])
        self.assertEqual(rows[0]["goal_count"],1)
        self.assertEqual(rows[0]["actual_lean_goal_state"],"h : True\n⊢ True")
        with self.assertRaises(AssertionError):
            core.parse_compiler_trace(log.replace("⊢ True","not a goal"),[item])
        with self.assertRaises(AssertionError):
            core.parse_compiler_trace("random source syntax",[item])

    def test_model_action_vocabulary_and_step_bound_frozen(self):
        self.assertEqual(core.MAX_TACTIC_STEPS,3)
        self.assertEqual(len(set(core.TACTICS)),len(core.TACTICS))
        self.assertTrue(all("\n" not in(a) for a in core.TACTICS))

    def test_bounds_refuse_unlisted_source_theorem(self):
        lines=("import Init\ntheorem foo : True := by\n  trivial\n").splitlines()
        with self.assertRaises(AssertionError):
            core.theorem_bounds(lines,"notPresent")


if __name__=="__main__":
    unittest.main()
