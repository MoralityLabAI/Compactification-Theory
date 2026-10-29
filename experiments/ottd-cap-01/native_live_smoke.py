"""Native real-engine smoke for OTTD-CAP-01, designed for clean Linux CI.

The prepared world is a SHORT, explicit derivative of the upstream two_towns
fixture: one agent seat, two founded towns, road+rail allowed, 400% speed, one
game year. No model harness or paid inference is needed. Live calls use the
genuine pinned 15.3 engine and authenticated upstream control plane.

Usage:
  python native_live_smoke.py prepare --upstream /path/to/pinned/upstream --work /tmp/ottd-live
  python /path/to/pinned/upstream/image/init_game.py --root /path/to/pinned/upstream/image \
      --world /tmp/ottd-live/world.json --seats /tmp/ottd-live/seats.json \
      --dev-home /tmp/ottd-live/dev
  python native_live_smoke.py exercise --upstream /path/to/pinned/upstream --work /tmp/ottd-live
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import secrets
import sys
import time
from urllib.request import Request, urlopen

from bridge import SeatBridge, BridgeError, UPSTREAM_PIN
from native_controller import NativeExecutor, PLAN_SCHEMA, compile_route

SEED = 101


def atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare(upstream: Path, work: Path) -> None:
    if not (upstream / "image" / "init_game.py").exists():
        raise RuntimeError("upstream checkout must contain the real game bootstrap")
    # The upstream top-level package imports its verifiers inference harness.
    # This smoke only needs the independently importable Pydantic scenario/schema
    # modules, so mark the package search path without executing that __init__.
    import types
    package = types.ModuleType("openttd_bench")
    package.__path__ = [str(upstream / "openttd_bench")]
    sys.modules["openttd_bench"] = package
    from openttd_bench.tasks.two_towns_rail import spec
    from openttd_bench.task_spec import Rules
    from openttd_bench.schema import schema_dict

    task = spec(SEED)
    task.rules = Rules(allowed_modes={"road", "rail"}, duration_years=1,
                       game_speed_percent=400, max_loan=0, clock="realtime")
    task.name = "ottd_cap01_native_smoke_s101"
    work.mkdir(parents=True, exist_ok=True)
    atomic(work / "world.json", task.world())
    # Seat and admin secrets are deliberately generated locally and never
    # printed or included in the CI artifact upload.
    if (work / "seats.json").exists():
        raise RuntimeError("seats already exist: native smoke requires an empty workdir")
    atomic(work / "seats.json", {
        "tokens": {secrets.token_hex(24): 0},
        "admin": secrets.token_hex(24),
    })
    (work / "seats.json").chmod(0o600)
    atomic(upstream / "image" / "schema.json", schema_dict())
    atomic(work / "fixture.json", {
        "upstream_pin": UPSTREAM_PIN, "game_engine": "15.3",
        "task_type": "shortened_two_towns_road_smoke",
        "seed": SEED, "clock": "realtime", "speed_percent": 400,
        "start_year": 1950, "horizon_year": 1951,
        "LLM_inference": "none", "paid_model_calls": 0,
        "world_sha256": sha256((work / "world.json").read_bytes()).hexdigest(),
    })
    print("Prepared bounded synthetic native fixture, seed 101; no game started.")


def admin_api(url: str, token: str, path: str) -> dict:
    req = Request(url + path, headers={"Authorization": "Bearer " + token})
    with urlopen(req, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("native admin endpoint returned non-object")
    return result


def ensure_boot(url: str, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url + "/health", timeout=5) as response:
                status = json.load(response)
            if status.get("boot_complete") and status.get("status") == "ok":
                return
            last = str({k: status.get(k) for k in ("status", "boot_complete", "gs_configured")})
        except Exception as exc:
            last = type(exc).__name__ + ": " + str(exc)
        time.sleep(1)
    raise RuntimeError("native server never completed boot: " + last)


def field_cargo(table: dict, label: str) -> list[dict]:
    items = table.get("engines")
    if not isinstance(items, list):
        raise RuntimeError("road.engines missing engine list")
    return [row for row in items if row.get("label") == label and
            type(row.get("id")) is int and row.get("capacity", 0) > 0]


def place_plan(bridge: SeatBridge, *, stop_after: int = 480) -> tuple[dict, dict]:
    """Survey two towns and seek a straight, station-viable road corridor.

    The search consists only of real visible API observations and test-mode
    validations, with bounded requests. No private map, world seed or grading
    information is supplied to the route selection function.
    """
    towns_result = bridge.command("inspect.towns", {"limit": 50})
    towns = towns_result.get("towns") if isinstance(towns_result, dict) else None
    if not isinstance(towns, list):
        raise RuntimeError("town list unavailable in live seat")
    by_name = {t.get("name"): t for t in towns}
    if "Northgate" not in by_name or "Southbridge" not in by_name:
        raise RuntimeError("expected two founded towns missing from native world")
    first, second = by_name["Northgate"], by_name["Southbridge"]
    if not (first["x"] < second["x"] and first["y"] == second["y"]):
        raise RuntimeError("unrecognized town geometry")
    engine_result = bridge.command("road.engines", {"limit": 50})
    choices = field_cargo(engine_result, "PASS")
    if not choices:
        raise RuntimeError("no passenger road engine available from live catalog")
    engine = min(choices, key=lambda r: (r.get("price", 10**10), r["id"]))
    if "road" not in bridge.state().get("allowed_modes", []):
        raise RuntimeError("native fixture disallows road construction")

    center_y = int(first["y"])
    a_x, b_x = int(first["x"]), int(second["x"])
    # Favors existing passenger activity while keeping the corridor off the town
    # center. Each exploratory command consumes real-time game-clock budget.
    offsets = [3, -3, 4, -4, 2, -2, 5, -5, 1, -1, 0, 6, -6]
    endpoint_offsets = [3, 4, 2, 5, 1, 6, 7]
    inspected = 0
    viable = []
    for dy in offsets:
        y = center_y + dy
        for da in endpoint_offsets:
            for db in endpoint_offsets:
                if inspected >= stop_after:
                    break
                x1, x2 = a_x + da, b_x - db
                if x2 - x1 < 10:
                    continue
                inspected += 1
                acc1 = bridge.command("inspect.acceptance", {"x": x1, "y": y})
                acc2 = bridge.command("inspect.acceptance", {"x": x2, "y": y})
                if not isinstance(acc1, dict) or not isinstance(acc2, dict):
                    continue
                # Both stops must both produce and accept passenger cargo.
                p1 = int(acc1.get("produces", {}).get("PASS", 0))
                p2 = int(acc2.get("produces", {}).get("PASS", 0))
                a1 = int(acc1.get("accepts", {}).get("PASS", 0))
                a2 = int(acc2.get("accepts", {}).get("PASS", 0))
                if min(p1, p2, a1, a2) < 8:
                    continue
                # A through-stop cannot replace a house, an existing station
                # or some town-owned road junctions. Demand vacant, genuinely
                # buildable landing tiles before trying the long road.
                endpoints = [
                    bridge.command("inspect.tile", {"x": x1, "y": y}),
                    bridge.command("inspect.tile", {"x": x2, "y": y}),
                ]
                if any(not isinstance(tile, dict) or tile.get("buildable") is not True
                       or tile.get("house") or tile.get("road_station")
                       or tile.get("road_depot") for tile in endpoints):
                    continue
                try:
                    test = bridge.command("road.road",
                                          {"x": x1, "y": y, "to_x": x2, "to_y": y, "full": True},
                                          validate=True)
                except BridgeError:
                    # A candidate's real OpenTTD test-mode construction can fail
                    # (existing road, house, slope, authority); continue surveying.
                    continue
                if not isinstance(test, dict) or test.get("possible") is not True:
                    continue
                mid = (x1 + x2) // 2
                # A depot placed directly beside an east-west road can have
                # no accessible north-south roadbit. Reserve a one-tile access
                # spur and site the depot beyond it to make a true T-junction.
                spur = bridge.command("inspect.tile", {"x": mid, "y": y + 1})
                depot_site = bridge.command("inspect.tile", {"x": mid, "y": y + 2})
                if (not isinstance(spur, dict) or not isinstance(depot_site, dict)
                        or spur.get("buildable") is not True
                        or depot_site.get("buildable") is not True):
                    continue
                try:
                    spur_test = bridge.command("road.road",
                                               {"x": mid, "y": y, "to_x": mid, "to_y": y + 1,
                                                "full": True}, validate=True)
                except BridgeError:
                    continue
                if not isinstance(spur_test, dict) or spur_test.get("possible") is not True:
                    continue
                viable.append((min(p1, p2), min(a1, a2), -(x2 - x1),
                               (x1, x2, y, mid, p1, p2, a1, a2)))
                if len(viable) >= 8:
                    break
            if len(viable) >= 8 or inspected >= stop_after:
                break
        if len(viable) >= 8 or inspected >= stop_after:
            break
    if not viable:
        raise RuntimeError(f"surveyed {inspected} corridors; no validated passenger road corridor found")
    chosen = max(viable)[-1]
    x1, x2, y, mid, p1, p2, a1, a2 = chosen
    plan = {
        "version": PLAN_SCHEMA, "mode": "road",
        "stations": {
            "a": {"site": {"x": x1, "y": y}, "front": {"x": x1 + 1, "y": y},
                  "kind": "bus", "drive_through": True},
            "b": {"site": {"x": x2, "y": y}, "front": {"x": x2 - 1, "y": y},
                  "kind": "bus", "drive_through": True},
        },
        "segments": [
            {"start": {"x": x1, "y": y}, "end": {"x": x2, "y": y}},
            {"start": {"x": mid, "y": y}, "end": {"x": mid, "y": y + 1}},
        ],
        "depot": {"site": {"x": mid, "y": y + 2}, "front": {"x": mid, "y": y + 1}},
        "vehicle": {"engine": engine["id"], "wagon_engine": None, "wagons": 0},
        "limits": {"max_cost": 750000, "max_wall_seconds": 90,
                   "latest_start_year": 1950},
    }
    compile_route(plan)
    return plan, {
        "surveyed_corridors": inspected, "candidates_with_passenger_production": len(viable),
        "selected": {"x1": x1, "x2": x2, "y": y, "p1": p1, "p2": p2,
                     "a1": a1, "a2": a2},
        "engine": {k: engine.get(k) for k in ("id", "name", "label", "price", "capacity")},
    }


def exercise(work: Path) -> int:
    world = json.loads((work / "world.json").read_text())
    secrets_doc = json.loads((work / "seats.json").read_text())
    token = next(iter(secrets_doc["tokens"]))
    admin_token = secrets_doc["admin"]
    url = "http://127.0.0.1:" + str(world["ports"]["api"])
    report_path = work / "report.json"
    report = {"status": "started", "task_type": "shortened_two_towns_road_smoke",
              "upstream_pin": UPSTREAM_PIN, "native_execution": True,
              "model_calls": 0, "game_engine": world["engine"]["version"]}
    atomic(report_path, report)
    try:
        ensure_boot(url)
        bridge = SeatBridge(api_url=url, token=token, allow_writes=True)
        state = bridge.state()
        if state.get("date", {}).get("year") != 1950 or state.get("clock_mode") != "realtime":
            raise RuntimeError("native test clock and initial world do not match fixture")
        report["game_start"] = {"date": state["date"], "clock_mode": state["clock_mode"]}
        report["own_company_before"] = bridge.command("inspect.company", {})
        plan, survey = place_plan(bridge)
        report["survey"] = survey
        atomic(work / "native-plan.json", plan)
        atomic(report_path, report)
        receipt = NativeExecutor(bridge).run(plan, work / "native-receipt.json")
        junction = plan["depot"]["front"]
        main = {"x": junction["x"], "y": junction["y"] - 1}
        depot = plan["depot"]["site"]
        report["depot_access"] = {
            "main_to_spur": bridge.command("road.connected",
                {"x": main["x"], "y": main["y"],
                 "to_x": junction["x"], "to_y": junction["y"]}),
            "spur_to_depot": bridge.command("road.connected",
                {"x": junction["x"], "y": junction["y"],
                 "to_x": depot["x"], "to_y": depot["y"]}),
        }
        atomic(report_path, report)
        if not all(isinstance(q, dict) and q.get("connected") is True
                   for q in report["depot_access"].values()):
            raise RuntimeError("the built depot spur fails upstream road.connected checks")
        report["receipt_sha256"] = sha256((work / "native-receipt.json").read_bytes()).hexdigest()
        report["completed_steps"] = len(receipt["steps"])
        report["vehicle"] = bridge.command("inspect.vehicle", {"vehicle": receipt["captures"]["vehicle"]})
        report["orders"] = bridge.command("inspect.orders", {"vehicle": receipt["captures"]["vehicle"]})
        report["stations"] = [
            bridge.command("inspect.station", {"station": receipt["captures"][f"station_{n}"]})
            for n in ("a", "b")
        ]
        # PASS is monitored by the borrowed two_towns objective. Poll the
        # actual upstream cargo accumulator. The fixture is only one year long.
        deadline = time.monotonic() + 115
        delivered = 0
        snapshots = []
        while time.monotonic() < deadline:
            delivery = bridge.command("inspect.deliveries", {})
            cargo = delivery.get("cargos", {}) if isinstance(delivery, dict) else {}
            passenger = cargo.get("PASS", {})
            delivered = int(passenger.get("total") or 0)
            company = bridge.command("inspect.company", {})
            vehicle_live = bridge.command("inspect.vehicle", {"vehicle": receipt["captures"]["vehicle"]})
            station_live = [
                bridge.command("inspect.station", {"station": receipt["captures"][f"station_{n}"]})
                for n in ("a", "b")
            ]
            snapshots.append({"date": bridge.state().get("date"),
                              "delivered_PASS": delivered,
                              "operating_profit_observed": company.get("operating_profit"),
                              "cash_observed": company.get("cash"),
                              "vehicle": {k: vehicle_live.get(k) for k in ("x", "y", "state", "speed", "orders", "profit_this_year", "cargo")},
                              "station_cargo": [s.get("cargo") for s in station_live]})
            if delivered > 0:
                break
            if bridge.state().get("game_over"):
                break
            time.sleep(5)
        report["cargo_delivered_PASS"] = delivered
        report["sampled_metrics"] = snapshots[-30:]
        report["final_vehicle"] = bridge.command("inspect.vehicle", {"vehicle": receipt["captures"]["vehicle"]})
        report["final_stations"] = [
            bridge.command("inspect.station", {"station": receipt["captures"][f"station_{n}"]})
            for n in ("a", "b")
        ]
        report["acceptance_after_construction"] = [
            bridge.command("inspect.acceptance", station["site"])
            for station in (plan["stations"]["a"], plan["stations"]["b"])
        ]
        report["town_production"] = [
            bridge.command("inspect.town", {"town": t["id"]})
            for t in bridge.command("inspect.towns", {"limit": 50})["towns"]
            if t.get("name") in ("Northgate", "Southbridge")
        ]
        report["native_grading_snapshot"] = admin_api(url, admin_token, "/api/admin/snapshot")
        report["own_company_after"] = bridge.command("inspect.company", {})
        if delivered <= 0:
            raise RuntimeError("route was built, but no actual passenger delivery was recorded")
        report["status"] = "native_cargo_delivery_verified"
        atomic(report_path, report)
        print("Native OpenTTD 15.3 cargo delivery verified, PASS=", delivered)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        atomic(report_path, report)
        print(report["failure"], file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "exercise"):
        s = modes.add_parser(name)
        s.add_argument("--upstream", type=Path, required=True)
        s.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        prepare(args.upstream.resolve(), args.work.resolve())
        return 0
    return exercise(args.work.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
