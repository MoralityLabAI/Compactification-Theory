"""Offline tests for Distill.mechanism_curriculum synthetic oracle-verified export."""
from __future__ import annotations
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from Distill.mechanism_curriculum import export, SCHEMA


class MechanismCurriculumTests(unittest.TestCase):
    def test_deterministic_disjoint_and_oracle_invariants(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            m1 = export(d1)
            m2 = export(d2)
            self.assertEqual(m1, m2)
            self.assertEqual(m1["schema_version"], SCHEMA)
            self.assertEqual(m1["source_kind"], "synthetic_oracle_only")
            self.assertFalse(m1["provider_outputs_admitted"])
            self.assertFalse(m1["training_admission"])
            self.assertEqual(m1["total_rows"], 69)

            seen_ids = set()
            decoupled_count = 0
            receipt_gap_count = 0
            dense_loss_count = 0
            for sp in m1["splits"]:
                pub_lines = (Path(d1) / f"{sp['split']}.public.jsonl").read_text(encoding="utf-8").splitlines()
                ref_lines = (Path(d1) / f"{sp['split']}.referee.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(pub_lines), sp["rows"])
                self.assertEqual(len(ref_lines), sp["rows"])
                for pl, rl in zip(pub_lines, ref_lines):
                    pub = json.loads(pl)
                    ref = json.loads(rl)
                    self.assertEqual(pub["row_id"], ref["row_id"])
                    self.assertNotIn(pub["row_id"], seen_ids)
                    seen_ids.add(pub["row_id"])
                    # Optimal action always has zero regret
                    opt = ref["optimal_action"]
                    self.assertIn(opt, pub["legal_actions"])
                    self.assertEqual(Fraction(ref["action_regrets"][opt]), 0)
                    for a, r in ref["action_regrets"].items():
                        self.assertGreaterEqual(Fraction(r), 0)
                    if pub["mechanism"] == "belief_quantization" and ref["brier_regret_decoupled"]:
                        decoupled_count += 1
                    if pub["mechanism"] == "receipt_quotient" and Fraction(ref["soundness_gap"]) > 0:
                        receipt_gap_count += 1
                    if pub["mechanism"] == "overhead_regime_switch" and Fraction(ref["baselines"]["always_dense_learned"]) > 0:
                        dense_loss_count += 1

            self.assertEqual(len(seen_ids), 69)
            self.assertGreater(receipt_gap_count, 0)
            self.assertGreater(dense_loss_count, 0)
            self.assertGreaterEqual(decoupled_count, 0)


if __name__ == "__main__":
    unittest.main()
