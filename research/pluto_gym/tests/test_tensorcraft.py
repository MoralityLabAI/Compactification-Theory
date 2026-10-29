"""Source-pin gates and optional tests against the actual TensorCraft snapshot.

Run the integration tests offline with TENSORCRAFT_TEST_CHECKOUT pointing at a
clean f12f3241b686231d09fbfe40a9f03706ebc7e068 checkout or complete snapshot.
Native programs are never launched. Dummy asset bytes test preparation only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.pluto_gym import tensorcraft as tc


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_spec(root):
    """Independent synthetic local assets accepted by the real upstream planner."""
    filenames = ["StarCraft.exe", "bwapi-data/AI/pluto.dll",
                 "bwapi-data/AI/pluto/pluto_infer.exe",
                 "bwapi-data/AI/pluto/pluto_weights.bin", "bwapi-data/bwapi.ini"]
    variants = ["control", "devourer_speed_550"]
    conditions, mods = {}, {}
    for variant in variants:
        original = root / ("original-" + variant)
        for relative in filenames:
            target = original / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"SYNTHETIC_ASSET:" + relative.encode())
        mod = root / (variant + ".mpq")
        mod.write_bytes(b"SYNTHETIC_MOD:" + variant.encode())
        conditions[variant] = {"game_dir": str(original), "mod_file": str(mod)}
        mods[variant] = {"mpq": str(mod), "sha256": sha(mod.read_bytes())}
    worker = root / "worker"
    for relative in filenames:
        target = worker / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"SYNTHETIC_ASSET:" + relative.encode())
    prefix = root / "wineprefix"
    prefix.mkdir()
    game_map, bandit = root / "map.scx", root / "bandit.txt"
    game_map.write_bytes(b"SYNTHETIC_MAP")
    bandit.write_bytes(b"seq 1\nend\n")
    spec = {
        "experiment_id": "adapter_offline_fixture", "pairs": 2, "schedule_seed": 19,
        "map_file": str(game_map), "initial_bandit_file": str(bandit),
        "conditions": conditions, "agent_model": "md09b_1331000",
        "opponent": "SyntheticEnemy", "player_race": "Zerg", "opponent_race": "Zerg",
    }
    runtime = {
        "schema": 1, "mods": mods,
        "capacity": {"max_parallel": 1, "cpu_budget_threads": 6, "threads_per_match": 6,
                     "ram_budget_mb": 6000, "ram_per_match_mb": 6000},
        "slots": [{"id": "worker1", "game_dir": str(worker), "wineprefix": str(prefix),
                   "display": ":91", "port_base": 46100,
                   "bandit_paths": [str(worker / "bwapi-data/write/pluto_bandit_SyntheticEnemy.txt")]}],
        "launcher": {"argv": ["NEVER_EXECUTED_MISSING_NATIVE_LAUNCHER", "{run_id}",
                              "{map_path}", "{mod_mpq}", "{replay_path}"], "timeout_seconds": 30},
    }
    spec_file, runtime_file = root / "spec.json", root / "runtime.json"
    write_json(spec_file, spec)
    write_json(runtime_file, runtime)
    return spec_file, runtime_file, runtime


def trace_events():
    events = [{"schema": 1, "event": "start", "run_id": "synthetic_adapter_trace",
               "variant": "control", "producer": "bwapi_tournament_trace_v1",
               "frame_step": 1, "full_information_at_start": False,
               "replay_mode": False, "self_player_id": 0,
               "start_tile_x": 4, "start_tile_y": 7,
               "observed_rng_seed": 19, "map_sha1": "a" * 40}]
    for frame in range(4):
        events.append({"schema": 1, "event": "frame", "frame": frame,
                       "complete_information": False, "dropped_units": 0,
                       "main_ai_event_ms": 1, "logical_fps": 60, "units": []})
    events.append({"schema": 1, "event": "end", "frame": 4,
                   "observed_frames": 4, "truncated": False,
                   "win_reported_for_current_player": False})
    return events


def write_trace(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


class PinAndInputTests(unittest.TestCase):
    def test_content_pin_and_extra_code_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            relative, data = "starcraft_lab/__init__.py", b"# checked\n"
            path = root / relative
            path.parent.mkdir()
            path.write_bytes(data)
            spec = {"repository": "fixture", "commit": "1" * 40,
                    "source_files": {relative: {"size_bytes": len(data), "git_blob_sha1": blob_sha(data)}},
                    "capabilities": {"native_reset_step_bridge": False},
                    "planner_variants": ["control", "devourer_speed_550"],
                    "mod_variants": ["control"]}
            with patch.object(tc, "_source_spec", return_value=spec):
                adapter = tc.TensorCraftAdapter(root)
                self.assertEqual(adapter.inspect()["verified_files"], 1)
                path.write_bytes(b"# altered\n")
                with self.assertRaisesRegex(tc.TensorCraftError, "blob mismatch"):
                    adapter.inspect()
                path.write_bytes(data)
                (path.parent / "injected.py").write_text("raise RuntimeError('must not import')")
                with self.assertRaisesRegex(tc.TensorCraftError, "file set differs"):
                    adapter.inspect()

    def test_native_requires_literal_true_before_any_source_or_process_access(self):
        adapter = tc.TensorCraftAdapter("/missing/dependency")
        for value in (False, 1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(tc.TensorCraftError, "allow_native=True"):
                adapter.run("/missing/workspace", allow_native=value)

    def test_scenario_bounds_reject_ineffective_or_unbounded_inputs(self):
        adapter = tc.TensorCraftAdapter("/missing/dependency")
        valid = {"id": "test", "A": {"mutalisk": 2}, "B": {"mutalisk": 2}}
        for name, value in (("horizon_frames", 1.5), ("focus", float("nan")),
                            ("step_frames", True), ("ignored_rule", 2)):
            scenario = dict(valid, **{name: value})
            with self.subTest(name=name), self.assertRaises(tc.TensorCraftError):
                adapter.simulate_batch(scenario)
        invalid_count = copy.deepcopy(valid)
        invalid_count["A"]["mutalisk"] = -1
        with self.assertRaises(tc.TensorCraftError):
            adapter.simulate_batch(invalid_count)
        for count in (True, 0, 5001):
            with self.assertRaises(tc.TensorCraftError):
                adapter.simulate_batch(valid, trials=count)

    def test_missing_dependency_and_invalid_timeout(self):
        with self.assertRaises(tc.TensorCraftError):
            tc.TensorCraftAdapter("/missing/dependency").inspect()
        for timeout in (True, 0, 1.5, 3601):
            with self.assertRaises(tc.TensorCraftError):
                tc.TensorCraftAdapter("/missing/dependency", offline_timeout_seconds=timeout)


@unittest.skipUnless(os.environ.get("TENSORCRAFT_TEST_CHECKOUT"),
                     "set TENSORCRAFT_TEST_CHECKOUT for pinned upstream offline integration")
class PinnedUpstreamTests(unittest.TestCase):
    def setUp(self):
        self.adapter = tc.TensorCraftAdapter(os.environ["TENSORCRAFT_TEST_CHECKOUT"])

    def test_complete_upstream_source_pin_and_capabilities(self):
        result = self.adapter.inspect()
        self.assertEqual(result["commit"], "f12f3241b686231d09fbfe40a9f03706ebc7e068")
        self.assertEqual(result["verified_files"], 70)
        self.assertFalse(result["capabilities"]["pluto_weight_training"])
        self.assertFalse(result["native_runtime_verified"])

    def test_actual_batch_and_farm_prepare_with_synthetic_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, runtime, _ = fixture_spec(root)
            # Shell metacharacters are ordinary path characters here.
            output = root / "planned space;$(unused)"
            receipt = self.adapter.prepare(spec, runtime, output)
            self.assertEqual(receipt["status"], "frozen_native_experiment")
            prepared = receipt["result"]["prepared"]
            self.assertEqual(prepared["status"], "prepared_not_executed")
            self.assertEqual(prepared["games"], 4)
            self.assertEqual(prepared["pairs"], 2)
            self.assertFalse(receipt["native_game_verified"])
            self.assertTrue((output / "farm/bandit_baseline.txt").is_file())
            self.assertFalse((output / "farm/execution_summary.json").exists())
            with self.assertRaises(tc.TensorCraftError):
                self.adapter.prepare(spec, runtime, output)
            with self.assertRaises(tc.TensorCraftError):
                self.adapter.run(output / "farm")

    def test_bad_asset_hash_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, runtime_path, runtime = fixture_spec(root)
            runtime["mods"]["control"]["sha256"] = "0" * 64
            write_json(runtime_path, runtime)
            with self.assertRaisesRegex(tc.TensorCraftError, "MPQ SHA-256 mismatch"):
                self.adapter.prepare(spec, runtime_path, root / "blocked")
            self.assertFalse((root / "blocked").exists())

    def test_strict_trace_parser_preserves_synthetic_origin_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.jsonl"
            events = trace_events()
            write_trace(path, events)
            receipt = self.adapter.analyze_trace(path)
            self.assertEqual(receipt["status"], "observed_trace")
            self.assertEqual(receipt["result"]["analysis"]["observed_frames"], 4)
            self.assertTrue(receipt["result"]["analysis"]["strict_calibration_eligible"])
            self.assertFalse(receipt["native_game_verified"])
            self.assertFalse(receipt["full_observation_action_training_data"])
            self.assertIn("authenticity_unverified", receipt["result"]["input_origin"])
            events[2]["complete_information"] = True
            write_trace(path, events)
            with self.assertRaisesRegex(tc.TensorCraftError, "fog-limited"):
                self.adapter.analyze_trace(path)

    def test_actual_upstream_surrogate_is_seeded_and_labeled(self):
        scenario = {"id": "offline_support_fixture", "A": {"mutalisk": 2, "devourer": 1},
                    "B": {"mutalisk": 3}, "starting_distance_px": 200,
                    "formation_spread_px": 28, "horizon_frames": 60}
        first = self.adapter.simulate_batch(scenario, "control", trials=2, seed=13)
        second = self.adapter.simulate_batch(scenario, "control", trials=2, seed=13)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "synthetic_surrogate")
        self.assertEqual(first["result"]["source"], "stochastic_surrogate_not_observed_games")
        self.assertEqual(first["result"]["trials"], 2)
        self.assertFalse(first["native_game_verified"])
        receipt = self.adapter.simulate("zvz_equal_spend_clustered", trials=1, seed=13)
        self.assertEqual(receipt["result"]["scenario_id"], "zvz_equal_spend_clustered")


if __name__ == "__main__":
    unittest.main()
