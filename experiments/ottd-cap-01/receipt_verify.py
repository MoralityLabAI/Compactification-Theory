"""Offline verification of an OTTD-CAP-01 native action receipt.

This verifies the deterministic record against an expected route plan. SHA-256
provides consistency and tamper evidence relative to a trusted saved digest;
it is not a signature from the upstream server or proof that game actions ran.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from native_controller import PLAN_SCHEMA, compile_route, resolve
from bridge import UPSTREAM_PIN


def canonical(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def check(receipt: dict, plan: dict) -> dict:
    expected = compile_route(plan)
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be an object")
    if receipt.get("schema") != PLAN_SCHEMA or receipt.get("upstream_pin") != UPSTREAM_PIN:
        raise ValueError("receipt schema or pinned source mismatch")
    if receipt.get("plan_sha256") != sha256(canonical(plan).encode()).hexdigest():
        raise ValueError("receipt plan content hash mismatch")
    if receipt.get("mode") != plan["mode"]:
        raise ValueError("receipt route mode mismatch")
    rows = receipt.get("steps")
    if not isinstance(rows, list) or not 0 <= len(rows) <= len(expected):
        raise ValueError("invalid receipt step count")
    captures = receipt.get("captures")
    if not isinstance(captures, dict) or any(type(v) is not int or v < 0 for v in captures.values()):
        raise ValueError("invalid captured game IDs")
    digest = "0" * 64
    committed_writes = 0
    confirmed = 0
    validated_total = 0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("receipt step must be an object")
        command = expected[idx]
        if row.get("name") != command.name or row.get("cmd") != command.cmd:
            raise ValueError(f"step {idx} differs from compiled route")
        if "args" in row:
            try:
                args = resolve(command.args, captures)
            except Exception:
                args = None  # A failed partial run may lack the next capture.
            if args is not None and row.get("args") != args:
                raise ValueError(f"step {idx} arguments differ from compiled plan")
        validation = row.get("validation")
        if validation is not None:
            if not isinstance(validation, dict) or validation.get("possible") is not True:
                raise ValueError(f"step {idx} contains invalid validation data")
            cost = validation.get("cost")
            if type(cost) is not int or cost < 0:
                raise ValueError("validation costs must be nonnegative integers")
            validated_total += cost
        if row.get("status") == "confirmed":
            confirmed += 1
            if not isinstance(row.get("execution"), dict) or row["execution"].get("ok") is not True:
                raise ValueError(f"step {idx} has no confirmed execution receipt")
            if not isinstance(row.get("confirmation"), dict):
                raise ValueError(f"step {idx} has no independent confirmation")
        elif row.get("status") != "failed":
            raise ValueError("unrecognized step status")
        digest = sha256((digest + canonical(row)).encode()).hexdigest()
    if digest != receipt.get("trace_sha256"):
        raise ValueError("SHA-256 receipt chain mismatch")
    if validated_total != receipt.get("validated_cost_total"):
        raise ValueError("validated spend total differs from steps")
    attempted = receipt.get("writes_attempted")
    if type(attempted) is not int or not 0 <= attempted <= len(rows):
        raise ValueError("invalid attempted-write count")
    state = receipt.get("status")
    if state == "completed":
        if len(rows) != len(expected) or confirmed != len(rows) or attempted != len(rows):
            raise ValueError("completed receipt is missing confirmed route steps")
    elif state == "stopped":
        if not isinstance(receipt.get("error"), str) or not receipt["error"]:
            raise ValueError("stopped receipt requires an error")
    else:
        raise ValueError("receipt is incomplete or still running")
    return {
        "valid": True, "complete": state == "completed",
        "confirmed_steps": confirmed, "recorded_steps": len(rows),
        "attempted_writes": attempted,
        "validated_cost_total": validated_total,
        "trace_sha256": digest, "plan_sha256": receipt["plan_sha256"],
        "source": "local_record_consistency_only",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("plan", type=Path)
    args = p.parse_args(argv)
    output = check(json.loads(args.receipt.read_text()),
                   json.loads(args.plan.read_text()))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
