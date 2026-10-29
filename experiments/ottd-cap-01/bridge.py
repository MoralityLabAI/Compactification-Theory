"""Explicit boundary to Sophia Xu's openttd-bench control plane.

No OpenTTD processes, Prime inference or agent seats are started here.
The user provisions a seat via the upstream harness. Transport defaults to the
seat-local authenticated API. Map reads can start its realtime clock.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable
import urllib.error
import urllib.request


UPSTREAM = "thesophiaxu/openttd-bench"
UPSTREAM_PIN = "fee25ab19837959fd9c9749808b117144b9d1b3e"
READ_COMMANDS = frozenset({
    "inspect.tile", "inspect.region", "inspect.acceptance",
    "inspect.towns", "inspect.town", "inspect.industries", "inspect.industry",
    "inspect.industry_types", "inspect.cargos", "inspect.companies",
    "inspect.company", "inspect.stations", "inspect.station",
    "inspect.vehicles", "inspect.vehicle", "inspect.orders",
    "inspect.bridges", "inspect.subsidies", "inspect.deliveries",
})
WRITE_COMMANDS = frozenset({
    "road.road", "road.stop", "road.depot", "rail.track", "rail.station",
    "rail.depot", "vehicles.build", "vehicles.add_wagon", "vehicles.start", "orders.append",
})


class BridgeError(RuntimeError):
    pass


class SeatBridge:
    """JSON /api/state and /api/batch bridge with a constrained command allowlist.

    Accepting a transport callable makes real response envelopes testable offline.
    No endpoint may be inferred from an LLM's free-form output.
    """

    def __init__(self, transport: Callable[[str, str, dict | None], dict] | None = None,
                 allow_writes: bool = False, api_url: str | None = None,
                 token: str | None = None):
        self.allow_writes = allow_writes
        self.api_url = (api_url or os.environ.get("OPENTTD_API_URL")
                        or "http://127.0.0.1:8000").rstrip("/")
        self.token = token if token is not None else os.environ.get("OPENTTD_API_TOKEN", "")
        self.transport = transport or self._http

    def _http(self, method: str, path: str, payload: dict | None) -> dict:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.api_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                body = json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
            raise BridgeError("OpenTTD seat request failed: " + str(exc)) from exc
        if not isinstance(body, dict):
            raise BridgeError("expected JSON object from OpenTTD control plane")
        return body

    def state(self) -> dict:
        result = self.transport("GET", "/api/state", None)
        if not isinstance(result, dict):
            raise BridgeError("invalid state payload")
        return result

    def command(self, name: str, args: dict | None = None, *, validate: bool = False) -> Any:
        is_read = name in READ_COMMANDS
        if not is_read and name not in WRITE_COMMANDS:
            raise BridgeError("command is absent from the pinned allowlist")
        if not is_read and not self.allow_writes and not validate:
            raise PermissionError("write gate closed; run explicit seat integration with allow_writes=True")
        if is_read and validate:
            raise BridgeError("read commands have no validate twin")
        if args is not None and not isinstance(args, dict):
            raise BridgeError("command arguments must be a JSON object")
        payload = {"items": [{"cmd": name, "args": args or {}, "validate": validate}],
                   "stop_on_error": True}
        response = self.transport("POST", "/api/batch", payload)
        if not isinstance(response, dict) or len(response.get("results", [])) != 1:
            raise BridgeError("invalid /api/batch envelope")
        reply = response["results"][0]
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise BridgeError("command rejected: " + str(reply.get("error") if isinstance(reply, dict) else reply))
        result = reply.get("result")
        if isinstance(result, dict) and result.get("ok") is False:
            raise BridgeError("engine rejected action: " + str(result.get("error")))
        return result

    def ticket(self, ticket: int, timeout: float = 8.0) -> dict:
        """Read an actual upstream asynchronous order ticket.

        Reads do not mutate an order. A pending, failed or unknown ticket is
        never silently treated as success. The remote endpoint limits waits
        to 60 seconds; this adapter additionally caps a single wait at 15.
        """
        if type(ticket) is not int or ticket < 0:
            raise BridgeError("ticket must be a nonnegative integer")
        if type(timeout) not in (float, int) or not 0 <= timeout <= 15:
            raise BridgeError("ticket timeout must be in [0,15]")
        result = self.transport("GET", f"/api/ticket/{ticket}?timeout={timeout:g}", None)
        if not isinstance(result, dict) or result.get("ticket") != ticket:
            raise BridgeError("upstream ticket identity mismatch")
        if result.get("status") not in ("pending", "done", "failed"):
            raise BridgeError("unrecognized ticket status")
        return result

    def probe(self, include_map: bool = False) -> dict:
        """Only state by default. Reads via /api/batch can START the live clock."""
        status = self.state()
        if not include_map or status.get("game_over"):
            return {"origin": UPSTREAM, "pin": UPSTREAM_PIN, "state": status}
        towns = self.command("inspect.towns", {"limit": 20})
        return {"origin": UPSTREAM, "pin": UPSTREAM_PIN,
                "state": status, "towns": towns}


def atif_summary(path: str | Path) -> dict:
    """Read actual upstream ATIF v1.8 without inventing a score or model results."""
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema_version") != "ATIF-v1.8" or not isinstance(doc.get("steps"), list):
        raise ValueError("expected an ATIF-v1.8 trajectory")
    counts = {"steps": 0, "tool_calls": 0, "subagents": 0}
    def visit(tree: dict) -> None:
        if not isinstance(tree.get("steps"), list):
            raise ValueError("invalid nested trajectory")
        counts["steps"] += len(tree["steps"])
        counts["tool_calls"] += sum(len(s.get("tool_calls") or []) for s in tree["steps"])
        for child in tree.get("subagent_trajectories", []):
            counts["subagents"] += 1
            visit(child)
    visit(doc)
    return {
        "origin": UPSTREAM,
        "upstream_pin": UPSTREAM_PIN,
        "source_sha256": sha256(raw).hexdigest(),
        "trajectory_id": doc.get("trajectory_id"),
        "agent": doc.get("agent"),
        "usage_reported_by_upstream": doc.get("final_metrics"),
        "counts": counts,
        "operating_profit": None,  # requires authoritative grading snapshot
        "grade_verified": False,
    }
