"""Upstream command envelopes, opt-in reads, write gate and ATIF provenance."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge import SeatBridge, BridgeError, UPSTREAM_PIN, atif_summary


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.nested_failure = False

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        if method == "GET":
            return {"game_over": False, "date": {"year": 1950}}
        if self.nested_failure:
            return {"results": [{"ok": True, "result": {"ok": False, "error": "build blocked"}}]}
        return {"results": [{"ok": True, "result": [{"id": 1, "name": "Town"}]}]}


class BridgeTests(unittest.TestCase):
    def test_probe_state_only_by_default(self):
        transport = FakeTransport()
        result = SeatBridge(transport=transport).probe()
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][:2], ("GET", "/api/state"))
        self.assertEqual(result["pin"], UPSTREAM_PIN)

    def test_map_probe_is_explicit(self):
        transport = FakeTransport()
        result = SeatBridge(transport=transport).probe(include_map=True)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.calls[-1][2]["items"][0]["cmd"], "inspect.towns")
        self.assertEqual(result["towns"][0]["name"], "Town")

    def test_write_gate_and_allowlist(self):
        transport = FakeTransport()
        bridge = SeatBridge(transport=transport)
        with self.assertRaises(PermissionError):
            bridge.command("road.stop", {"x": 1})
        self.assertFalse(transport.calls)
        for forbidden in ("sys.status", "shell.exec"):
            with self.assertRaises(BridgeError):
                bridge.command(forbidden)
        with self.assertRaises(BridgeError):
            bridge.command("inspect.towns", validate=True)
        bridge.command("road.stop", {"x": 1}, validate=True)
        self.assertTrue(transport.calls[-1][2]["items"][0]["validate"])
        SeatBridge(transport=transport, allow_writes=True).command("road.stop", {"x": 1})
        self.assertFalse(transport.calls[-1][2]["items"][0]["validate"])

    def test_nested_engine_rejection(self):
        t = FakeTransport()
        t.nested_failure = True
        with self.assertRaises(BridgeError):
            SeatBridge(transport=t, allow_writes=True).command("road.stop", {})

    def test_atif_is_provenance_only(self):
        doc = {"schema_version": "ATIF-v1.8", "trajectory_id": "t1",
               "agent": {"name": "fake"}, "steps": [{"tool_calls": [{"id": "x"}]}, {}],
               "subagent_trajectories": [{"steps": [{}]}],
               "final_metrics": {"total_prompt_tokens": 42}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trajectory.json"
            content = json.dumps(doc).encode()
            path.write_bytes(content)
            report = atif_summary(path)
            self.assertEqual(report["source_sha256"], sha256(content).hexdigest())
            self.assertEqual(report["counts"], {"steps": 3, "tool_calls": 1, "subagents": 1})
            self.assertFalse(report["grade_verified"])
            self.assertIsNone(report["operating_profit"])
            path.write_text("{}")
            with self.assertRaises(ValueError):
                atif_summary(path)


if __name__ == "__main__":
    unittest.main()
