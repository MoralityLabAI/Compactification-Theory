"""OTTD-CAP-01 native seat controller: deterministic compiled road/rail build plans.

Targets Sophia Xu's openttd-bench pinned API through SeatBridge. Compile is
fully offline. Native execution is explicit, limited, non-resumable, and leaves
an audit receipt after any partial failure. No claim of live integration here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from bridge import BridgeError, SeatBridge, UPSTREAM_PIN

PLAN_SCHEMA = "ottd-cap01.native-route.v1"
WRITE_COMMANDS = frozenset({
    "road.road", "road.stop", "road.depot", "rail.station", "rail.track",
    "rail.depot", "vehicles.build", "vehicles.add_wagon", "vehicles.start",
    "orders.append",
})
VALIDATABLE = WRITE_COMMANDS - {"orders.append", "vehicles.start"}
MAX_STEPS = 40


class PlanError(ValueError):
    """Unsafe or structurally invalid route plan."""


class RunStopped(RuntimeError):
    """A live operation failed or hit a predeclared resource gate."""


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _nonnegative(value: Any, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise PlanError(f"{field} must be an integer >= {int(positive)}")
    return value


def _coord(raw: Any, field: str) -> dict[str, int]:
    if not isinstance(raw, dict) or set(raw) != {"x", "y"}:
        raise PlanError(f"{field} must have exactly x and y")
    point = {key: _nonnegative(raw[key], f"{field}.{key}") for key in ("x", "y")}
    if max(point.values()) >= 4096:
        raise PlanError("coordinates must fit the declared 4096-tile safety bound")
    return point


def _adjacent(a: dict, b: dict, description: str) -> None:
    if abs(a["x"] - b["x"]) + abs(a["y"] - b["y"]) != 1:
        raise PlanError(description + " must be orthogonally adjacent")


def _station(raw: Any, mode: str, name: str) -> dict:
    if not isinstance(raw, dict):
        raise PlanError(f"stations.{name} must be an object")
    if mode == "road":
        if set(raw) != {"site", "front", "kind", "drive_through"}:
            raise PlanError(f"road station {name}: expected site/front/kind/drive_through")
        site, front = _coord(raw["site"], "station site"), _coord(raw["front"], "station front")
        _adjacent(site, front, "station front")
        if raw["kind"] not in ("bus", "truck") or type(raw["drive_through"]) is not bool:
            raise PlanError("road stations require kind bus|truck and boolean drive_through")
        return {**site, "front_x": front["x"], "front_y": front["y"],
                "kind": raw["kind"], "drive_through": raw["drive_through"]}
    if set(raw) != {"site", "direction", "platforms", "length"}:
        raise PlanError(f"rail station {name}: expected site/direction/platforms/length")
    if raw["direction"] not in ("NE_SW", "NW_SE"):
        raise PlanError("rail station direction must match a platform axis")
    platforms = _nonnegative(raw["platforms"], "platforms", positive=True)
    length = _nonnegative(raw["length"], "length", positive=True)
    if platforms > 8 or length > 16:
        raise PlanError("rail stations require at most 8 platforms and 16 tiles")
    return {**_coord(raw["site"], "station site"), "direction": raw["direction"],
            "platforms": platforms, "length": length}


def _depot(raw: Any) -> dict:
    if not isinstance(raw, dict) or set(raw) != {"site", "front"}:
        raise PlanError("depot requires site and front")
    site, front = _coord(raw["site"], "depot site"), _coord(raw["front"], "depot front")
    _adjacent(site, front, "depot front")
    return {**site, "front_x": front["x"], "front_y": front["y"]}


@dataclass(frozen=True)
class Step:
    name: str
    cmd: str
    args: dict[str, Any]
    capture: str | None = None


def _ref(key: str) -> str:
    return "$" + key


def compile_route(document: dict) -> tuple[Step, ...]:
    """Compile an explicitly mapped two-stop route to upstream primitive calls.

    The plan contains *no* free-form command strings or arbitrary API endpoints.
    A fixed grammar makes it possible to review exactly which writes can occur.
    """
    if not isinstance(document, dict) or set(document) != {
        "version", "mode", "stations", "segments", "depot", "vehicle", "limits"
    }:
        raise PlanError("plan has unknown or missing top-level keys")
    if document["version"] != PLAN_SCHEMA:
        raise PlanError("unsupported plan schema")
    mode = document["mode"]
    if mode not in ("road", "rail"):
        raise PlanError("mode must be road or rail")
    st = document["stations"]
    if not isinstance(st, dict) or set(st) != {"a", "b"}:
        raise PlanError("stations must contain exactly a and b")
    first = _station(st["a"], mode, "a")
    second = _station(st["b"], mode, "b")
    if (first["x"], first["y"]) == (second["x"], second["y"]):
        raise PlanError("station origins must be different")
    depot = _depot(document["depot"])
    segments = document["segments"]
    if not isinstance(segments, list) or not 1 <= len(segments) <= 20:
        raise PlanError("segments must be a nonempty list of at most 20")
    vehicle = document["vehicle"]
    if not isinstance(vehicle, dict) or set(vehicle) != {"engine", "wagon_engine", "wagons"}:
        raise PlanError("vehicle requires engine, wagon_engine and wagons")
    engine = _nonnegative(vehicle["engine"], "vehicle.engine")
    wagons = _nonnegative(vehicle["wagons"], "vehicle.wagons")
    if mode == "road":
        if wagons != 0 or vehicle["wagon_engine"] is not None:
            raise PlanError("road vehicles carry their own cargo; wagons must be zero")
        if first["kind"] != second["kind"]:
            raise PlanError("road route needs matching bus or truck station kinds")
    else:
        if not 1 <= wagons <= 6:
            raise PlanError("rail route requires 1..6 explicit cargo wagons")
        _nonnegative(vehicle["wagon_engine"], "vehicle.wagon_engine")
    limits = document["limits"]
    if not isinstance(limits, dict) or set(limits) != {"max_cost", "max_wall_seconds", "latest_start_year"}:
        raise PlanError("limits require max_cost/max_wall_seconds/latest_start_year")
    _nonnegative(limits["max_cost"], "limits.max_cost", positive=True)
    seconds = limits["max_wall_seconds"]
    if type(seconds) not in (float, int) or not 0 < seconds <= 3600:
        raise PlanError("max_wall_seconds must be in (0,3600]")
    start_year = _nonnegative(limits["latest_start_year"], "latest_start_year")
    if not 1950 <= start_year < 1960:
        raise PlanError("latest_start_year must be from 1950 to 1959")

    steps: list[Step] = []
    if mode == "road":
        parsed = []
        for i, raw in enumerate(segments):
            if not isinstance(raw, dict) or set(raw) != {"start", "end"}:
                raise PlanError("road segments require start/end")
            start, end = _coord(raw["start"], f"segments[{i}].start"), _coord(raw["end"], f"segments[{i}].end")
            if start == end or (start["x"] != end["x"] and start["y"] != end["y"]):
                raise PlanError("road segments must be nonzero axis-aligned runs")
            if parsed and parsed[-1][1] != start:
                raise PlanError("road segments must form a contiguous waypoint chain")
            parsed.append((start, end))
            steps.append(Step(f"road_{i}", "road.road",
                              {**start, "to_x": end["x"], "to_y": end["y"], "full": True}))
        def on_corridor(p: dict) -> bool:
            return any((start["x"] == end["x"] == p["x"] and
                        min(start["y"], end["y"]) <= p["y"] <= max(start["y"], end["y"])) or
                       (start["y"] == end["y"] == p["y"] and
                        min(start["x"], end["x"]) <= p["x"] <= max(start["x"], end["x"]))
                       for start, end in parsed)

        # Check the declared route topology before any live request. Terrain,
        # station acceptance, road direction and construction remain game-validated.
        for side, data in (("a", first), ("b", second)):
            access = {"x": data["x"], "y": data["y"]} if data["drive_through"] else {
                "x": data["front_x"], "y": data["front_y"]}
            if not on_corridor(access):
                raise PlanError(f"station_{side} does not reach a declared road segment")
        if not on_corridor({"x": depot["front_x"], "y": depot["front_y"]}):
            raise PlanError("depot front must touch the declared road corridor")
        for side, data in (("a", first), ("b", second)):
            steps.append(Step(f"station_{side}", "road.stop", data, f"station_{side}"))
    else:
        for side, data in (("a", first), ("b", second)):
            steps.append(Step(f"station_{side}", "rail.station", data, f"station_{side}"))
        for i, raw in enumerate(segments):
            if not isinstance(raw, dict) or set(raw) != {"from", "via", "to"}:
                raise PlanError("rail segments need from/via/to, matching upstream rail.track")
            left = _coord(raw["from"], "track.from")
            mid = _coord(raw["via"], "track.via")
            right = _coord(raw["to"], "track.to")
            if left == mid or mid == right:
                raise PlanError("rail from/via/to must be distinct where they meet")
            steps.append(Step(f"track_{i}", "rail.track",
                              {"from_x": left["x"], "from_y": left["y"],
                               "via_x": mid["x"], "via_y": mid["y"],
                               "to_x": right["x"], "to_y": right["y"]}))
    steps.append(Step("depot", f"{mode}.depot", depot))
    steps.append(Step("vehicle", "vehicles.build",
                      {"depot_x": depot["x"], "depot_y": depot["y"], "engine": engine}, "vehicle"))
    if mode == "rail":
        for index in range(wagons):
            steps.append(Step(f"wagon_{index}", "vehicles.add_wagon",
                              {"vehicle": _ref("vehicle"), "engine": vehicle["wagon_engine"]}))
    for side in ("a", "b"):
        steps.append(Step(f"order_{side}", "orders.append",
                          {"vehicle": _ref("vehicle"), "station": _ref(f"station_{side}"), "flags": []}))
    steps.append(Step("start", "vehicles.start", {"vehicle": _ref("vehicle")}))
    if len(steps) > MAX_STEPS or any(s.cmd not in WRITE_COMMANDS for s in steps):
        raise PlanError("native plan exceeds declared step/action allowlist")
    return tuple(steps)


def resolve(args: dict[str, Any], captures: dict[str, int]) -> dict[str, Any]:
    out = {}
    for key, value in args.items():
        if isinstance(value, str) and value.startswith("$"):
            name = value[1:]
            if name not in captures:
                raise RunStopped(f"unresolved receipt dependency: {name}")
            out[key] = captures[name]
        else:
            out[key] = value
    return out


class NativeExecutor:
    """Bounded, explicitly enabled live execution with a fail-closed audit log."""

    def __init__(self, bridge: SeatBridge, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.bridge = bridge
        self.clock = clock
        self.sleep = sleep

    def _check_state(self, latest_start_year: int, mode: str) -> dict:
        state = self.bridge.state()
        if type(state.get("game_over")) is not bool:
            raise RunStopped("state missing authoritative game_over boolean")
        if state["game_over"]:
            raise RunStopped("game horizon passed")
        if state.get("clock_mode") != "realtime":
            raise RunStopped("this controller requires the upstream real-time clock mode")
        if mode not in state.get("allowed_modes", []):
            raise RunStopped("requested transport mode is disallowed by upstream world rules")
        date = state.get("date")
        if not isinstance(date, dict) or type(date.get("year")) is not int:
            raise RunStopped("state missing authoritative game year")
        horizon = state.get("horizon_year")
        if type(horizon) is not int or date["year"] >= horizon:
            raise RunStopped("horizon year unavailable or already reached")
        if date["year"] > latest_start_year:
            raise RunStopped("remaining horizon below the plan's latest_start_year gate")
        return {"year": date["year"], "month": date.get("month"), "day": date.get("day"),
                "game_over": state["game_over"], "clock_mode": state["clock_mode"]}

    def _confirm(self, step: Step, args: dict, result: dict, captures: dict) -> dict:
        """A second read checks identity or tile occupancy after each write.

        Upstream async construction may apply on the following tick. Polls are
        bounded and have no retries of mutation calls.
        """
        cmd = step.cmd
        if cmd in ("road.road", "rail.track", "road.depot", "rail.depot"):
            if cmd == "road.road":
                position = {"x": args["to_x"], "y": args["to_y"]}
                field = "road"
            elif cmd == "rail.track":
                position = {"x": args["via_x"], "y": args["via_y"]}
                field = "rail"
            else:
                position = {"x": args["x"], "y": args["y"]}
                field = "road_depot" if cmd.startswith("road.") else "rail_depot"
            last = None
            for _ in range(5):
                last = self.bridge.command("inspect.tile", position)
                if isinstance(last, dict) and bool(last.get(field)):
                    return {"verified": field, "tile": position}
                self.sleep(0.2)
            raise RunStopped(f"{step.name}: engine acknowledged construction but {field} tile not observed")
        if cmd in ("rail.station", "road.stop"):
            station = captures[step.capture]
            observed = self.bridge.command("inspect.station", {"station": station})
            if not isinstance(observed, dict) or observed.get("id") != station:
                raise RunStopped(f"{step.name}: station identity not confirmed")
            return {"verified": "station_id", "id": station}
        if cmd in ("vehicles.build", "vehicles.add_wagon", "vehicles.start"):
            vehicle = captures["vehicle"]
            observed = self.bridge.command("inspect.vehicle", {"vehicle": vehicle})
            if not isinstance(observed, dict) or observed.get("id") != vehicle:
                raise RunStopped(f"{step.name}: vehicle identity not confirmed")
            return {"verified": "vehicle_id", "id": vehicle, "state": observed.get("state")}
        if cmd == "orders.append":
            tid = result.get("ticket")
            if type(tid) is not int or tid < 0:
                raise RunStopped(f"{step.name}: upstream order ticket missing")
            ticket = self.bridge.ticket(tid, timeout=8)
            if ticket.get("status") != "done" or ticket.get("ok") is not True:
                raise RunStopped(f"{step.name}: order ticket {tid} failed or timed out")
            vehicle = captures["vehicle"]
            station = args["station"]
            observed = self.bridge.command("inspect.orders", {"vehicle": vehicle})
            if not isinstance(observed, dict) or not isinstance(observed.get("orders"), list):
                raise RunStopped("unable to verify vehicle order list")
            seen = [r.get("station") for r in observed["orders"] if isinstance(r, dict)]
            if station not in seen:
                raise RunStopped(f"{step.name}: ticket done but station {station} absent from orders")
            return {"verified": "resolved_ticket_and_station_order", "ticket": tid, "station": station}
        raise RunStopped("unexpected command confirmation type")

    def run(self, document: dict, receipt_path: Path) -> dict:
        steps = compile_route(document)
        if receipt_path.exists():
            raise PlanError("receipt exists: refusing unsafe replay or overwrite of a partially executed route")
        limits = document["limits"]
        deadline = self.clock() + float(limits["max_wall_seconds"])
        receipt = {"schema": PLAN_SCHEMA, "upstream_pin": UPSTREAM_PIN,
                   "plan_sha256": sha256(_canonical(document).encode()).hexdigest(),
                   "mode": document["mode"], "status": "started", "steps": [],
                   "captures": {}, "validated_cost_total": 0,
                   "writes_attempted": 0, "trace_sha256": "0" * 64}
        captures: dict[str, int] = {}

        def persist() -> None:
            receipt["captures"] = dict(captures)
            tmp = receipt_path.with_name(receipt_path.name + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, receipt_path)

        def append(entry: dict) -> None:
            receipt["trace_sha256"] = sha256((receipt["trace_sha256"] + _canonical(entry)).encode()).hexdigest()
            receipt["steps"].append(entry)
            persist()

        try:
            # Acquire the receipt name exclusively. A second executor cannot race
            # this job and replay side-effecting operations against the same file.
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with receipt_path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            for step in steps:
                if self.clock() >= deadline:
                    raise RunStopped("live wall-time limit reached")
                before = self._check_state(limits["latest_start_year"], document["mode"])
                args = resolve(step.args, captures)
                entry = {"name": step.name, "cmd": step.cmd, "args": args,
                         "game_date_before": before, "wall_elapsed_seconds": 0.0}
                t0 = self.clock()
                try:
                    if step.cmd in VALIDATABLE:
                        check = self.bridge.command(step.cmd, args, validate=True)
                        if not isinstance(check, dict) or check.get("possible") is not True:
                            raise RunStopped(f"{step.name}: validation rejected or returned no affirmative possibility")
                        cost = check.get("cost")
                        if type(cost) is not int or cost < 0:
                            raise RunStopped(f"{step.name}: invalid validation cost")
                        if receipt["validated_cost_total"] + cost > limits["max_cost"]:
                            raise RunStopped(f"{step.name}: projected capex exceeds max_cost")
                        receipt["validated_cost_total"] += cost
                        entry["validation"] = {"possible": True, "cost": cost}
                    if self.clock() >= deadline:
                        raise RunStopped("live wall-time limit reached after validation")
                    self._check_state(limits["latest_start_year"], document["mode"])
                    receipt["writes_attempted"] += 1
                    # No retries: an HTTP error might occur after the server committed a write.
                    result = self.bridge.command(step.cmd, args)
                    if not isinstance(result, dict):
                        raise RunStopped(f"{step.name}: mutation receipt is not an object")
                    if result.get("ok") is not True:
                        raise RunStopped(f"{step.name}: engine mutation not affirmed")
                    if step.capture is not None:
                        key = "station" if step.capture.startswith("station_") else "vehicle"
                        item = result.get(key)
                        if type(item) is not int or item < 0:
                            raise RunStopped(f"{step.name}: missing authoritative {key} ID")
                        captures[step.capture] = item
                    entry["execution"] = result
                    entry["confirmation"] = self._confirm(step, args, result, captures)
                    after = self.bridge.state()
                    entry["game_date_after"] = after.get("date")
                    entry["game_over_after"] = after.get("game_over")
                    entry["status"] = "confirmed"
                except Exception as exc:
                    entry["status"] = "failed"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    entry["wall_elapsed_seconds"] = round(self.clock() - t0, 6)
                    append(entry)
            receipt["status"] = "completed"
            receipt["game_date_after"] = self._check_state(limits["latest_start_year"], document["mode"])
        except Exception as exc:
            receipt["status"] = "stopped"
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            persist()
            raise
        persist()
        return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path, help="JSON route specification; no arbitrary API commands")
    p.add_argument("--execute", action="store_true", help="enable live writes (requires both gates)")
    p.add_argument("--allow-clock-start", action="store_true")
    p.add_argument("--allow-writes", action="store_true")
    p.add_argument("--receipt", type=Path, help="required for live execution; never overwritten")
    args = p.parse_args(argv)
    try:
        doc = json.loads(args.plan.read_text())
        steps = compile_route(doc)
        if not args.execute:
            print(json.dumps({"mode": "offline_compilation", "upstream_pin": UPSTREAM_PIN,
                              "plan_sha256": sha256(_canonical(doc).encode()).hexdigest(),
                              "steps": [asdict(s) for s in steps]}, indent=2))
            return 0
        if not (args.allow_writes and args.allow_clock_start and args.receipt):
            p.error("live execution requires --allow-clock-start --allow-writes --receipt PATH")
        NativeExecutor(SeatBridge(allow_writes=True)).run(doc, args.receipt)
        print(f"Native plan completed; receipt: {args.receipt}")
        return 0
    except (ValueError, RunStopped, BridgeError, PermissionError, OSError) as exc:
        p.exit(1, f"native controller stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
