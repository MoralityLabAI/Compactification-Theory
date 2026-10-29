"""Budget matching, provenance and evaluation isolation for the small runner."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.pluto_gym.design import DesignBudget, DesignerError, OfflineDesigner
from research.pluto_gym.env import RuleSpec
from research.pluto_gym.io import digest, load_json
from research.pluto_gym.run import (
    RunConfig, _supervision, holdout_variants, interfaces, run_experiment,
    split_seeds, training_variants,
)


def small_config(**changes):
    values = dict(seed=3, width=4, pretrain_updates=4, iterations=2,
                  updates_per_iteration=3, batch_size=4, train_episodes=4,
                  probe_episodes=2, test_episodes=2, replay_capacity=16,
                  max_replay_bytes=1024, max_seconds=20)
    values.update(changes)
    return RunConfig(**values)


def read_traces(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "NumPy is optional for training runs")
class RunTests(unittest.TestCase):
    def test_matched_adaptation_budgets_and_frozen_checkpoint(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = run_experiment(config, output)
            manifest = load_json(output / "manifest.json")
            base = load_json(output / "base_checkpoint.json")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(load_json(output / "receipt.json"), result)
            self.assertEqual(result["manifest_sha256"], digest(manifest))
            self.assertEqual(result["arms"]["frozen"]["checkpoint_sha256"], digest(base))
            self.assertEqual(result["arms"]["frozen"]["updates"], 0)
            self.assertEqual(base["training_updates"], config.pretrain_updates)
            for arm in ("random", "designer"):
                saved = load_json(output / f"{arm}_checkpoint.json")
                self.assertEqual(digest(saved), result["arms"][arm]["checkpoint_sha256"])
                self.assertEqual(saved["training_updates"], config.pretrain_updates +
                                 config.iterations * config.updates_per_iteration)
                timeline = result["arms"][arm]["timeline"]
                self.assertEqual(timeline[0]["fit"]["initial_checkpoint_sha256"], digest(base))
                for index, row in enumerate(timeline):
                    self.assertEqual(row["collection"]["rows_retained"], config.train_episodes)
                    self.assertEqual(row["collection"]["unique_retained_episode_seeds"], config.train_episodes)
                    self.assertEqual(row["fit"]["updates_completed"], config.updates_per_iteration)
                    self.assertEqual(row["fit"]["examples_drawn"], config.updates_per_iteration * config.batch_size)
                    self.assertEqual(row["replay"]["retained_rows"], (index + 2) * config.train_episodes)
                    self.assertEqual(row["replay"]["allocated_payload_bytes"], config.replay_capacity * 17)
            for left, right in zip(result["arms"]["random"]["timeline"], result["arms"]["designer"]["timeline"]):
                self.assertEqual(left["collection"]["rows_retained"], right["collection"]["rows_retained"])
                self.assertEqual(left["fit"]["examples_drawn"], right["fit"]["examples_drawn"])
            expected_config = OfflineDesigner(DesignBudget(max_calls=config.iterations)).configuration()
            self.assertEqual(manifest["designer_configuration"], expected_config)
            expected_id = digest({"config": asdict(config), "sources": manifest["source_sha256"],
                                  "designer_configuration": expected_config})[:20]
            self.assertEqual(manifest["run_id"], expected_id)

    def test_supervision_keeps_one_row_per_declared_episode(self):
        seeds = [101, 202, 303, 404]
        for rules in training_variants().values():
            rows, receipt = _supervision(rules, interfaces()["camera"], seeds)
            self.assertEqual([row["episode_seed"] for row in rows], seeds)
            self.assertEqual(receipt["rows_retained"], len(seeds))
            self.assertGreaterEqual(receipt["oracle_rows_generated"], len(seeds))

    def test_seed_splits_and_rule_combinations_are_disjoint(self):
        config = small_config()
        groups = split_seeds(config)
        all_seeds = [seed for group in groups.values() for seed in group]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertEqual(groups, split_seeds(config))
        self.assertNotEqual(groups, split_seeds(small_config(seed=4)))
        train = {digest(r.to_dict()) for r in training_variants().values()}
        train.add(digest(RuleSpec().to_dict()))
        tests = {digest(r.to_dict()) for r in holdout_variants().values()}
        self.assertFalse(train & tests)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "marker.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_experiment(small_config(), output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["marker.txt"])

    def test_invalid_designer_response_leaves_failure_without_success(self):
        class BadDesigner(OfflineDesigner):
            def propose(self, request):
                self.calls += 1
                return {"schema": request["schema"], "request_id": request["request_id"],
                        "variant_id": request["candidates"][0]["variant_id"],
                        "rationale": "injected test response", "command": "invalid"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            designer = BadDesigner(DesignBudget(max_calls=1))
            with self.assertRaises(DesignerError):
                run_experiment(small_config(iterations=1), output, designer=designer)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "failure.json").is_file())
            self.assertFalse((output / "receipt.json").exists())
            failure = load_json(output / "failure.json")
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error_type"], "DesignerError")
            self.assertEqual(failure["designer"]["calls"], 1)

    def test_heldout_measurements_cannot_change_design_or_checkpoint(self):
        import research.pluto_gym.run as runner
        original_suite = runner._suite
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "normal"
            second = Path(temporary) / "changed_measurements"
            # Fix the source input so unrelated concurrent editor activity cannot
            # change request IDs while this behavioral isolation test executes.
            with patch.object(runner, "source_hashes", return_value={"test-source": "0" * 64}):
                normal = run_experiment(config, first)
                def changed_suite(*args, **kwargs):
                    measured = original_suite(*args, **kwargs)
                    for row in measured.values():
                        row["mean_reward"] = 0.123456
                        row["mean_interface_regret"] = 0.876544
                        row["sentinel"] = "HELDOUT_ONLY_CHANGED"
                    return measured
                with patch.object(runner, "_suite", side_effect=changed_suite):
                    changed = run_experiment(config, second)
            self.assertEqual(read_traces(first / "designer_traces.jsonl"),
                             read_traces(second / "designer_traces.jsonl"))
            for arm in ("frozen", "random", "designer"):
                self.assertEqual(normal["arms"][arm]["checkpoint_sha256"],
                                 changed["arms"][arm]["checkpoint_sha256"])
            self.assertNotEqual(normal["arms"]["designer"]["heldout"],
                                changed["arms"]["designer"]["heldout"])
            trace_text = (second / "designer_traces.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("HELDOUT_ONLY_CHANGED", trace_text)

    def test_overlap_and_insufficient_call_budget_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "no_run"
            with self.assertRaises(ValueError):
                run_experiment(small_config(), output,
                               designer=OfflineDesigner(DesignBudget(max_calls=1)))
            self.assertFalse(output.exists())
            with patch("research.pluto_gym.run.holdout_variants", return_value={"bad": RuleSpec()}):
                with self.assertRaises(ValueError):
                    run_experiment(small_config(), output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
