"""Offline contract tests of the native route compiler and mock seat."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_controller import (PLAN_SCHEMA, NativeExecutor, PlanError, RunStopped,
                               compile_route, main, resolve)


def rail_plan():
    return {
        "version": PLAN_SCHEMA, "mode": "rail",
        "stations": {
            "a": {"site": {"x": 40, "y": 40}, "direction": "NE_SW", "platforms": 1, "length": 2},
            "b": {"site": {"x": 60, "y": 40}, "direction": "NE_SW", "platforms": 1, "length": 2}
        },
        "segments": [{"from": {"x": 41, "y": 40}, "via": {"x": 42, "y": 40},
                      "to": {"x": 60, "y": 40}}],
        "depot": {"site": {"x": 50, "y": 41}, "front": {"x": 50, "y": 40}},
        "vehicle": {"engine": 13, "wagon_engine": 14, "wagons": 1},
        "limits": {"max_cost": 1000, "max_wall_seconds": 30, "latest_start_year": 1951}
    }


def road_plan():
    return {
        "version": PLAN_SCHEMA, "mode": "road",
        "stations": {
            "a": {"site": {"x": 10, "y": 21}, "front": {"x": 10, "y": 20},
                  "kind": "bus", "drive_through": False},
            "b": {"site": {"x": 25, "y": 21}, "front": {"x": 25, "y": 20},
                  "kind": "bus", "drive_through": False}
        },
        "segments": [{"start": {"x": 10, "y": 20}, "end": {"x": 25, "y": 20}}],
        "depot": {"site": {"x": 18, "y": 21}, "front": {"x": 18, "y": 20}},
        "vehicle": {"engine": 11, "wagon_engine": None, "wagons": 0},
        "limits": {"max_cost": 1000, "max_wall_seconds": 30, "latest_start_year": 1951}
    }


class FakeBridge:
    """Shaped after upstream JSON /api/batch and /api/ticket, without a server."""
    def __init__(self):
        self.calls = []
        self.fail_validation = None
        self.fail_ticket = False
        self.year = 1950
        self.over = False
        self.road_flags = True
        self.builds = []
        self.orders = []
        self.seat_read_count = 0
        self.tickets = []
        self.cost = 11
        self.ticket_counter = 400

    def state(self):
        self.seat_read_count += 1
        return {"game_over": self.over, "clock_mode": "realtime",
                "date": {"year": self.year, "month": 1, "day": 1},
                "allowed_modes": ["rail", "road"], "horizon_year": 1960}

    def ticket(self, number, timeout=8):
        self.tickets.append(number)
        return {"ticket": number, "status": "failed" if self.fail_ticket else "done",
                "ok": not self.fail_ticket}

    def command(self, name, args=None, *, validate=False):
        self.calls.append((name, args, validate))
        if validate:
            return {"ok": True, "possible": name != self.fail_validation,
                    "executed": False, "cost": self.cost}
        if name == "inspect.tile":
            return {"rail": self.road_flags, "road": self.road_flags,
                    "rail_depot": self.road_flags, "road_depot": self.road_flags}
        if name == "inspect.station":
            return {"id": args["station"]}
        if name == "inspect.vehicle":
            running = any(op == "vehicles.start" for op, _ in self.builds)
            return {"id": args["vehicle"], "state": "running" if running else "stopped"}
        if name == "inspect.orders":
            return {"vehicle": args["vehicle"], "orders": list(self.orders)}
        receipt = {"ok": True, "possible": True, "executed": True, "cost": self.cost}
        self.builds.append((name, args))
        if name in ("rail.station", "road.stop"):
            receipt["station"] = 100 if len([n for n, a in self.builds if n in ("rail.station", "road.stop")]) == 1 else 101
        if name == "vehicles.build":
            receipt["vehicle"] = 200
        if name == "orders.append":
            self.orders.append({"station": args["station"], "position": len(self.orders)})
            self.ticket_counter += 1
            receipt["ticket"] = self.ticket_counter
        return receipt


class CompilerTest(unittest.TestCase):
    def test_rail_sequence_and_dependencies(self):
        steps = compile_route(rail_plan())
        self.assertEqual([s.cmd for s in steps], ["rail.station", "rail.station", "rail.track",
                                                   "rail.depot", "vehicles.build", "vehicles.add_wagon",
                                                   "orders.append", "orders.append", "vehicles.start"])
        self.assertEqual(steps[-3].args["vehicle"], "$vehicle")
        self.assertEqual(steps[-3].args["station"], "$station_a")
        self.assertEqual(steps[-2].args["station"], "$station_b")
        self.assertEqual(resolve(steps[-2].args, {"vehicle": 3, "station_b": 9}),
                         {"vehicle": 3, "station": 9, "flags": []})

    def test_road_sequence(self):
        steps = compile_route(road_plan())
        self.assertEqual([s.cmd for s in steps], ["road.road", "road.stop", "road.stop",
                                                   "road.depot", "vehicles.build", "orders.append",
                                                   "orders.append", "vehicles.start"])
        self.assertEqual(steps[0].args["to_x"], 25)

    def test_reject_malformed_and_unknown_commands(self):
        bad = deepcopy(road_plan())
        bad["shell"] = "rm -rf /"
        with self.assertRaises(PlanError):
            compile_route(bad)
        for change in (lambda x: x["stations"]["a"]["front"].update(y=23),
                       lambda x: x["segments"][0]["end"].update(y=23),
                       lambda x: x["vehicle"].update(engine="$vehicle"),
                       lambda x: x["vehicle"].update(wagons=1),
                       lambda x: x["limits"].update(max_wall_seconds=0),
                       lambda x: x["limits"].update(latest_start_year=1960)):
            item = deepcopy(road_plan())
            change(item)
            with self.assertRaises(PlanError):
                compile_route(item)
        with self.assertRaises(RunStopped):
            resolve({"vehicle": "$vehicle"}, {})

    def test_offline_compilation_needs_no_server(self):
        with tempfile.TemporaryDirectory() as d:
            planfile = Path(d) / "plan.json"
            planfile.write_text(json.dumps(road_plan()))
            output = StringIO()
            with redirect_stdout(output):
                result = main([str(planfile)])
            self.assertEqual(result, 0)
            data = json.loads(output.getvalue())
            self.assertEqual(data["mode"], "offline_compilation")
            self.assertEqual(len(data["steps"]), 8)


class ExecutorTest(unittest.TestCase):
    def test_rail_end_to_end_mock_confirmation_and_hash(self):
        fake = FakeBridge()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "receipt.json"
            receipt = NativeExecutor(fake).run(rail_plan(), path)
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(len(receipt["steps"]), 9)
            self.assertEqual(receipt["captures"], {"station_a": 100, "station_b": 101, "vehicle": 200})
            self.assertEqual(fake.tickets, [401, 402])
            self.assertEqual(len([a for a in fake.calls if a[2]]), 6)
            self.assertEqual(receipt["validated_cost_total"], 66)
            self.assertEqual(json.loads(path.read_text()), receipt)
            self.assertEqual(fake.orders[0]["station"], 100)
            self.assertEqual(fake.orders[1]["station"], 101)
            self.assertNotEqual(receipt["trace_sha256"], "0" * 64)
            with self.assertRaises(PlanError):
                NativeExecutor(fake).run(rail_plan(), path)

    def test_road_end_to_end_mock(self):
        fake = FakeBridge()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "road.json"
            receipt = NativeExecutor(fake).run(road_plan(), path)
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["writes_attempted"], 8)
            self.assertEqual(len(fake.orders), 2)

    def test_failed_validation_leaves_partial_receipt(self):
        fake = FakeBridge()
        fake.fail_validation = "rail.track"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "failed.json"
            with self.assertRaises(RunStopped):
                NativeExecutor(fake).run(rail_plan(), path)
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["status"], "stopped")
            self.assertEqual(receipt["writes_attempted"], 2)
            self.assertEqual(receipt["steps"][-1]["status"], "failed")
            self.assertNotIn("vehicles.start", [x[0] for x in fake.builds])

    def test_failed_ticket_never_starts_vehicle(self):
        fake = FakeBridge()
        fake.fail_ticket = True
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "failed.json"
            with self.assertRaises(RunStopped):
                NativeExecutor(fake).run(road_plan(), path)
            self.assertNotIn("vehicles.start", [x[0] for x in fake.builds])
            self.assertEqual(json.loads(path.read_text())["status"], "stopped")

    def test_start_ack_without_running_state_is_failure(self):
        class StubbornBridge(FakeBridge):
            def command(self, name, args=None, *, validate=False):
                if name == "inspect.vehicle":
                    self.calls.append((name, args, validate))
                    return {"id": args["vehicle"], "state": "stopped"}
                return super().command(name, args, validate=validate)
        fake = StubbornBridge()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "stubborn.json"
            with self.assertRaises(RunStopped):
                NativeExecutor(fake, sleep=lambda _: None).run(road_plan(), path)
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["steps"][-1]["cmd"], "vehicles.start")
            self.assertEqual(receipt["steps"][-1]["status"], "failed")
            self.assertEqual(receipt["status"], "stopped")

    def test_cost_cap_blocks_before_mutation(self):
        fake = FakeBridge()
        plan = road_plan()
        plan["limits"]["max_cost"] = 1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "failure.json"
            with self.assertRaises(RunStopped):
                NativeExecutor(fake).run(plan, path)
            self.assertEqual(fake.builds, [])
            self.assertEqual(json.loads(path.read_text())["writes_attempted"], 0)

    def test_game_horizon_and_clock_fail_closed(self):
        for year, over in ((1952, False), (1950, True)):
            fake = FakeBridge()
            fake.year, fake.over = year, over
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(RunStopped):
                    NativeExecutor(fake).run(road_plan(), Path(d) / "blocked.json")
                self.assertEqual(fake.builds, [])


if __name__ == "__main__":
    unittest.main()
