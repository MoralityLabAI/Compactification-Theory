"""Offline tests for the live-game fixture's public-observation route selector."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge import BridgeError
from native_controller import PLAN_SCHEMA, compile_route
from native_live_smoke import place_plan


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


if __name__ == "__main__":
    unittest.main()
