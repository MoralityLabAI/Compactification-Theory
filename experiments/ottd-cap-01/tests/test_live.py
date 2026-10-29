"""Offline tests for the live-game fixture's public-observation route selector."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge import BridgeError
from native_controller import PLAN_SCHEMA, compile_route
from native_live_smoke import place_plan, verify_native_ledger


class FakeWorld:
    def __init__(self, reject_all=False):
        self.calls = []
        self.reject_all = reject_all
        self.failed_once = False

    def state(self):
        return {"allowed_modes": ["road", "rail"]}

    def command(self, name, args=None, *, validate=False):
        self.calls.append((name, args or {}, validate))
        if name == "inspect.towns":
            return {"towns": [
                {"name": "Northgate", "x": 108, "y": 128},
                {"name": "Southbridge", "x": 148, "y": 128},
            ]}
        if name == "road.engines":
            return {"engines": [
                {"id": 3, "label": "MAIL", "price": 10, "capacity": 10},
                {"id": 7, "label": "PASS", "price": 200, "capacity": 20},
                {"id": 8, "label": "PASS", "price": 220, "capacity": 30},
            ]}
        if name == "inspect.acceptance":
            return {"produces": {"PASS": 10}, "accepts": {"PASS": 10}}
        if name == "road.road" and validate:
            if self.reject_all or not self.failed_once:
                self.failed_once = True
                raise BridgeError("engine rejected action: ERR_AREA_NOT_CLEAR")
            return {"ok": True, "possible": True, "cost": 100}
        if name == "inspect.tile":
            return {"buildable": True}
        raise AssertionError(f"unexpected request: {name}")


class LivePlanSelectionTests(unittest.TestCase):
    def test_public_reads_and_validation_only(self):
        world = FakeWorld()
        plan, evidence = place_plan(world, stop_after=20)
        self.assertEqual(plan["version"], PLAN_SCHEMA)
        self.assertEqual(plan["mode"], "road")
        self.assertEqual(plan["vehicle"]["engine"], 7)
        self.assertEqual(evidence["selected"]["p1"], 10)
        self.assertTrue(len(compile_route(plan)) >= 8)
        self.assertGreaterEqual(evidence["surveyed_corridors"], 2)
        allowed = {"inspect.towns", "road.engines", "inspect.acceptance",
                   "road.road", "inspect.tile"}
        self.assertTrue({name for name, _, _ in world.calls} <= allowed)
        self.assertTrue(all(validated for name, _, validated in world.calls if name == "road.road"))
        self.assertFalse(any(name in ("vehicles.build", "road.stop") for name, _, _ in world.calls))

    def test_obstructed_corridors_fail_within_bound(self):
        world = FakeWorld(reject_all=True)
        with self.assertRaisesRegex(RuntimeError, "no validated passenger road corridor"):
            place_plan(world, stop_after=10)
        self.assertFalse(any(name == "road.stop" for name, _, _ in world.calls))


class NativeLedgerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "accounting_snapshot": {"fresh": True, "error": None},
            "date": {"year": 1950, "month": 4, "day": 16},
            "horizon_year": 1951,
            "companies": {"0": {
                "accounting": {
                    "valid": True, "version": "company-quarterly-v1",
                    "income": 1170, "expenses": -109, "operating_profit": 1061,
                },
                "operating_profit": 1061, "cargo_delivered": {"PASS": 31},
            }}
        }

    def test_matching_native_ledger_and_cargo(self):
        evidence = verify_native_ledger(self.snapshot, seat=0, delivered=31)
        self.assertEqual(evidence["operating_profit"], 1061)
        self.assertEqual(evidence["native_cargo_PASS"], 31)
        self.assertEqual(evidence["state"], "verified_interim_native_ledger")

    def test_stale_invalid_unbalanced_or_mismatched_ledger(self):
        from copy import deepcopy
        edits = [
            lambda s: s["accounting_snapshot"].update(fresh=False),
            lambda s: s["companies"]["0"]["accounting"].update(valid=False),
            lambda s: s["companies"]["0"]["accounting"].update(income=1171),
            lambda s: s["companies"]["0"].update(operating_profit=999),
            lambda s: s["companies"]["0"]["cargo_delivered"].update(PASS=30),
            lambda s: s["companies"]["0"]["accounting"].update(version="old-vehicle-ledger"),
        ]
        for mutate in edits:
            snapshot = deepcopy(self.snapshot)
            mutate(snapshot)
            with self.assertRaises(RuntimeError):
                verify_native_ledger(snapshot, seat=0, delivered=31)

    def test_native_ledger_requires_actual_profit(self):
        snapshot = self.snapshot
        snapshot["companies"]["0"]["accounting"]["income"] = 0
        snapshot["companies"]["0"]["accounting"]["expenses"] = -10
        snapshot["companies"]["0"]["accounting"]["operating_profit"] = -10
        snapshot["companies"]["0"]["operating_profit"] = -10
        with self.assertRaisesRegex(RuntimeError, "positive operating profit"):
            verify_native_ledger(snapshot, seat=0, delivered=31)


if __name__ == "__main__":
    unittest.main()
