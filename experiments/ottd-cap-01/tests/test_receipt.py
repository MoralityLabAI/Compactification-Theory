"""Offline integrity checks for real/mock OpenTTD native plan receipts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_controller import NativeExecutor, RunStopped
from receipt_verify import check
from test_native import FakeBridge, road_plan


class ReceiptTests(unittest.TestCase):
    def test_completed_native_receipt_matches_plan(self):
        fake = FakeBridge()
        plan = road_plan()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "complete.json"
            NativeExecutor(fake).run(plan, path)
            receipt = json.loads(path.read_text())
            out = check(receipt, plan)
            self.assertTrue(out["valid"])
            self.assertTrue(out["complete"])
            self.assertEqual(out["confirmed_steps"], 8)
            self.assertEqual(out["attempted_writes"], 8)

    def test_modified_trace_or_plan_is_detected(self):
        plan = road_plan()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "complete.json"
            NativeExecutor(FakeBridge()).run(plan, path)
            receipt = json.loads(path.read_text())
            bad = deepcopy(receipt)
            bad["steps"][0]["args"]["to_x"] += 1
            with self.assertRaises(ValueError):
                check(bad, plan)
            new_plan = deepcopy(plan)
            new_plan["limits"]["max_cost"] += 1
            with self.assertRaisesRegex(ValueError, "plan content hash"):
                check(receipt, new_plan)

    def test_failed_partial_receipt_has_explicit_scope(self):
        bridge = FakeBridge()
        bridge.fail_ticket = True
        plan = road_plan()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "partial.json"
            with self.assertRaises(RunStopped):
                NativeExecutor(bridge).run(plan, path)
            out = check(json.loads(path.read_text()), plan)
            self.assertTrue(out["valid"])
            self.assertFalse(out["complete"])
            self.assertLess(out["confirmed_steps"], 8)


if __name__ == "__main__":
    unittest.main()
