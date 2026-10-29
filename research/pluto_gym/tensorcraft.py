"""Pinned adapters to TensorCraft's existing StarCraft lab.

Each call verifies every upstream file against the Git blob manifest, then runs
an isolated temporary copy. A source snapshot works without Git. Native calls
keep TensorCraft's own preparation, evidence classes and explicit execution gate.
The adapter provides no native reset/step bridge or Pluto weight-training API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Literal, TypedDict

from .io import load_json, strict_json


class TensorCraftError(ValueError):
    """Unavailable dependency, invalid configuration, or failed upstream call."""


class TensorCraftReceipt(TypedDict):
    schema: str
    status: Literal["synthetic_surrogate", "observed_trace", "frozen_native_experiment"]
    operation: str
    source_commit: str
    source_verified: bool
    native_game_verified: bool
    full_observation_action_training_data: bool
    timeout_seconds: int
    request: dict
    adapter_source_sha256: str
    dependency_lock_sha256: str
    result: dict


_LOCK = Path(__file__).with_name("dependencies.lock.json")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_STDOUT_BYTES = 8 * 1024 * 1024


def _source_spec() -> dict:
    return json.loads(_LOCK.read_text(encoding="utf-8"))["tensorcraft"]


def _json_file(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise TensorCraftError("JSON input must be a file of at most 2 MiB")
    try:
        value = load_json(path, max_bytes=_MAX_JSON_BYTES)
    except (OSError, ValueError, UnicodeError) as exc:
        raise TensorCraftError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise TensorCraftError("JSON input must be an object")
    return value


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise TensorCraftError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


# This driver imports only the checked copy plus the isolated Python standard
# library. It calls upstream APIs rather than reimplementing their validators.
_DRIVER = r'''
import json, pathlib, runpy, sys
root, operation, payload = pathlib.Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
sys.path.insert(0, str(root))
def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
if operation == "prepare":
    from starcraft_lab.batch import build_plan, save_plan
    from starcraft_lab.farm.schema import validate_runtime
    from starcraft_lab.farm.runner import prepare
    spec, runtime = read(payload["spec_file"]), read(payload["runtime_file"])
    output = pathlib.Path(payload["output"])
    plan, _ = build_plan(spec)
    validate_runtime(runtime, plan, output / "farm")
    output.mkdir(parents=True, exist_ok=False)
    save_plan(spec, output / "plan")
    prepared = prepare(output / "plan" / "plan.json", pathlib.Path(payload["runtime_file"]), output / "farm")
    result = {"prepared": prepared, "workspace": str(output / "farm"),
              "plan_file": str(output / "plan" / "plan.json")}
elif operation == "run":
    # Keep the existing CLI, its explicit gate, exit code and evidence result.
    sys.argv = ["starcraft_lab.farm.runner", "run", "--workspace", payload["workspace"], "--allow-native"]
    runpy.run_module("starcraft_lab.farm.runner", run_name="__main__")
    raise SystemExit(0)
elif operation == "analyze_trace":
    from starcraft_lab.world_model.trace_analysis import analyze_file
    analysis = analyze_file(pathlib.Path(payload["path"]), require_complete=payload["require_complete"])
    binding = None
    if payload.get("manifest_file"):
        from starcraft_lab.world_model.provenance import bind
        binding = bind(read(payload["manifest_file"]), analysis, read(payload["replay_report_file"]),
                       pathlib.Path(payload["map_file"]) if payload.get("map_file") else None,
                       read(payload["spawn_tiles_file"]) if payload.get("spawn_tiles_file") else None)
    result = {"analysis": analysis, "provenance": binding,
              "input_origin": "caller_supplied_trace_authenticity_unverified"}
elif operation in ("simulate", "simulate_batch"):
    from starcraft_lab.world_model.experiments import load_scenarios
    from starcraft_lab.world_model.combat import simulate_batch
    if operation == "simulate_batch":
        scenario = payload["scenario"]
    else:
        matches = [s for s in load_scenarios() if s["id"] == payload["scenario_id"]]
        if len(matches) != 1:
            raise ValueError("unknown pinned TensorCraft scenario_id")
        scenario = matches[0]
    result = simulate_batch(scenario, payload["variant"], trials=payload["trials"], seed=payload["seed"])
else:
    raise ValueError("unsupported adapter operation")
print(json.dumps(result, sort_keys=True, allow_nan=False))
'''


class TensorCraftAdapter:
    """Use a clean, pinned upstream checkout or a complete pinned snapshot.

    ``offline_timeout_seconds`` bounds inspection/analysis/preparation calls.
    Native runs derive their outer deadline from the explicitly supplied farm
    plan and runtime. Source is copied into a private temporary directory, so
    upstream imports cannot alter or race the verified checkout.
    """

    def __init__(self, checkout: str | Path, *, offline_timeout_seconds: int = 120):
        self.checkout = Path(checkout).expanduser().resolve()
        self.offline_timeout_seconds = _integer(
            offline_timeout_seconds, "offline_timeout_seconds", 1, 3600)

    def _verified_source(self) -> tuple[dict, dict[str, bytes]]:
        spec = _source_spec()
        if not self.checkout.is_dir():
            raise TensorCraftError(f"TensorCraft checkout missing: {self.checkout}")
        expected = spec["source_files"]
        found: set[str] = set()
        for parent, dirs, files in os.walk(self.checkout, followlinks=False):
            directory = Path(parent)
            if directory == self.checkout:
                dirs[:] = [d for d in dirs if d != ".git"]
                files = [f for f in files if f != ".git"]
            for name in dirs:
                if (directory / name).is_symlink():
                    raise TensorCraftError("symlink directories are outside the pinned source contract")
            for name in files:
                path = directory / name
                if path.is_symlink() or not path.is_file():
                    raise TensorCraftError("source files must be regular files")
                found.add(path.relative_to(self.checkout).as_posix())
        if found != set(expected):
            missing = sorted(set(expected) - found)
            extra = sorted(found - set(expected))
            raise TensorCraftError(
                f"source file set differs from pin; missing={missing[:5]}, extra={extra[:5]}; "
                "use a clean checkout or complete snapshot without generated files")
        content = {}
        for relative, pin in expected.items():
            path = self.checkout / relative
            if path.stat().st_size != pin["size_bytes"]:
                raise TensorCraftError(f"source size mismatch: {relative}")
            data = path.read_bytes()
            if _blob_sha(data) != pin["git_blob_sha1"]:
                raise TensorCraftError(f"source Git blob mismatch: {relative}")
            content[relative] = data
        git_head = None
        if (self.checkout / ".git").exists():
            try:
                read = subprocess.run(
                    ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
                    shell=False, check=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=10, text=True)
            except (OSError, subprocess.SubprocessError) as exc:
                raise TensorCraftError("cannot verify TensorCraft Git HEAD") from exc
            git_head = read.stdout.strip()
            if git_head != spec["commit"]:
                raise TensorCraftError(f"TensorCraft Git HEAD differs from pinned {spec['commit']}")
        report = {
            "schema": "pluto_gym.tensorcraft_inspect.v1",
            "status": "source_verified",
            "repository": spec["repository"],
            "commit": spec["commit"],
            "git_head": git_head,
            "source_verification": "complete_git_blob_manifest",
            "verified_files": len(content),
            "capabilities": dict(spec["capabilities"]),
            "planner_variants": list(spec["planner_variants"]),
            "mod_variants": list(spec["mod_variants"]),
            "native_runtime_verified": False,
        }
        return report, content

    def inspect(self) -> dict:
        return self._verified_source()[0]

    @staticmethod
    def _stop_process(proc: subprocess.Popen) -> None:
        # Upstream already owns per-launcher timeouts and cleanup. This catches
        # a stuck adapter process. Wine that daemonizes into another session
        # still needs an operator check before any retry.
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               shell=False, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=15)
            except (OSError, subprocess.SubprocessError):
                proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                proc.kill()
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait(timeout=5)

    def _invoke(self, operation: str, payload: dict, timeout: int,
                status: Literal["synthetic_surrogate", "observed_trace", "frozen_native_experiment"]
                ) -> TensorCraftReceipt:
        source, files = self._verified_source()
        with tempfile.TemporaryDirectory(prefix="pluto_gym_tensorcraft_") as temp:
            root = Path(temp) / "source"
            root.mkdir()
            for relative, data in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            argv = [sys.executable, "-I", "-S", "-B", "-c", _DRIVER,
                    str(root), operation, json.dumps(payload, allow_nan=False)]
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                try:
                    proc = subprocess.Popen(argv, cwd=root, env=env, shell=False,
                                            stdout=stdout_file, stderr=stderr_file,
                                            start_new_session=(os.name != "nt"))
                except OSError as exc:
                    raise TensorCraftError("cannot start the isolated TensorCraft adapter") from exc
                try:
                    returncode = proc.wait(timeout=timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    self._stop_process(proc)
                    suffix = ("; inspect the owned native workers before retrying, "
                              "because Wine process cleanup is unverified") if operation == "run" else ""
                    raise TensorCraftError(f"TensorCraft {operation} interrupted or timed out{suffix}") from exc
                stdout_file.seek(0, os.SEEK_END)
                if stdout_file.tell() > _MAX_STDOUT_BYTES:
                    raise TensorCraftError("upstream JSON output exceeded 8 MiB")
                stdout_file.seek(0)
                raw = stdout_file.read()
                stderr_file.seek(0, os.SEEK_END)
                stderr_file.seek(max(0, stderr_file.tell() - 4096))
                diagnostic = stderr_file.read().decode("utf-8", errors="replace")
            try:
                result = strict_json(raw, max_bytes=_MAX_STDOUT_BYTES)
            except (ValueError, UnicodeError) as exc:
                raise TensorCraftError(
                    f"TensorCraft {operation} returned no valid JSON (exit {returncode}): {diagnostic}") from exc
            acceptable = returncode == 0 or (
                operation == "run" and returncode == 2 and isinstance(result, dict)
                and result.get("status") == "execution_attempted")
            if not acceptable or not isinstance(result, dict):
                raise TensorCraftError(f"TensorCraft {operation} failed (exit {returncode}): {diagnostic}")
        return {
            "schema": "pluto_gym.tensorcraft_receipt.v1",
            "status": status, "operation": operation,
            "source_commit": source["commit"], "source_verified": True,
            "native_game_verified": False,
            "full_observation_action_training_data": False,
            "timeout_seconds": timeout, "request": payload,
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_lock_sha256": hashlib.sha256(_LOCK.read_bytes()).hexdigest(),
            "result": result,
        }

    def prepare(self, spec_file: str | Path, runtime_file: str | Path,
                output: str | Path) -> TensorCraftReceipt:
        """Prepare frozen-Pluto A/B runs using upstream validation; start no game."""
        spec_path, runtime_path = Path(spec_file).resolve(), Path(runtime_file).resolve()
        destination = Path(output).resolve()
        _json_file(spec_path)
        _json_file(runtime_path)
        if destination.exists():
            raise TensorCraftError("prepare output already exists; choose a fresh directory")
        if destination == self.checkout or self.checkout in destination.parents:
            raise TensorCraftError("prepare output must be outside the pinned source checkout")
        return self._invoke("prepare", {
            "spec_file": str(spec_path), "runtime_file": str(runtime_path),
            "output": str(destination),
        }, self.offline_timeout_seconds, "frozen_native_experiment")

    def run(self, workspace: str | Path, *, allow_native: bool = False) -> TensorCraftReceipt:
        """Run the existing frozen-Pluto farm only after explicit native opt-in.

        The finite outer timeout follows the supplied job count, worker count,
        native match timeout and replay-parser allowance. This method can take
        hours for a configured batch. It performs no weight updates.
        """
        if allow_native is not True:
            raise TensorCraftError("native execution requires allow_native=True")
        work = Path(workspace).resolve()
        plan, runtime = _json_file(work / "plan.json"), _json_file(work / "runtime.json")
        games = plan.get("games")
        if not isinstance(games, list) or len(games) % 2:
            raise TensorCraftError("native plan must contain whole two-arm pairs")
        total = _integer(len(games), "native game count", 2, 4000)
        if not isinstance(runtime.get("capacity"), dict) or not isinstance(runtime.get("launcher"), dict):
            raise TensorCraftError("native runtime requires capacity and launcher objects")
        workers = _integer(runtime["capacity"].get("max_parallel"), "max_parallel", 1, 16)
        match_timeout = _integer(runtime["launcher"].get("timeout_seconds"),
                                 "launcher timeout_seconds", 30, 86400)
        batches = math.ceil((total // 2) / workers)
        timeout = 2 * batches * (match_timeout + 150) + 120
        return self._invoke("run", {"workspace": str(work)}, timeout, "frozen_native_experiment")

    def analyze_trace(self, path: str | Path, *, require_complete: bool = True,
                      manifest_file: str | Path | None = None,
                      replay_report_file: str | Path | None = None,
                      map_file: str | Path | None = None,
                      spawn_tiles_file: str | Path | None = None) -> TensorCraftReceipt:
        """Apply upstream strict telemetry checks and optional artifact binding.

        ``observed_trace`` describes the input protocol. A syntactically valid
        caller-supplied trace can be synthetic; authenticity stays unverified.
        No trace is promoted into observation/action examples for training.
        """
        if type(require_complete) is not bool:
            raise TensorCraftError("require_complete must be bool")
        if (manifest_file is None) != (replay_report_file is None):
            raise TensorCraftError("artifact binding requires manifest and replay report together")
        if manifest_file is None and (map_file is not None or spawn_tiles_file is not None):
            raise TensorCraftError("map/spawn binding requires manifest and replay report")
        trace = Path(path).resolve()
        if not trace.is_file() or trace.stat().st_size > 128 * 1024 * 1024:
            raise TensorCraftError("trace must be a file of at most 128 MiB")
        payload = {"path": str(trace), "require_complete": require_complete}
        for name, value in (("manifest_file", manifest_file), ("replay_report_file", replay_report_file),
                            ("map_file", map_file), ("spawn_tiles_file", spawn_tiles_file)):
            if value is not None:
                resolved = Path(value).resolve()
                if name != "map_file":
                    _json_file(resolved)
                elif not resolved.is_file():
                    raise TensorCraftError("map_file must be a file")
                payload[name] = str(resolved)
        return self._invoke("analyze_trace", payload, self.offline_timeout_seconds, "observed_trace")

    def simulate(self, scenario_id: str, *, variant: str = "control",
                 trials: int = 16, seed: int = 0) -> TensorCraftReceipt:
        """Run an existing pinned combat scenario; return an uncalibrated prior."""
        if not isinstance(scenario_id, str) or not 1 <= len(scenario_id) <= 120:
            raise TensorCraftError("scenario_id must name a pinned TensorCraft scenario")
        if variant not in _source_spec()["mod_variants"]:
            raise TensorCraftError("unknown TensorCraft mod variant")
        _integer(trials, "trials", 1, 5000)
        _integer(seed, "seed", 0, 2**63 - 1)
        return self._invoke("simulate", {"scenario_id": scenario_id, "variant": variant,
                                        "trials": trials, "seed": seed},
                            self.offline_timeout_seconds, "synthetic_surrogate")

    def simulate_batch(self, scenario: dict, variant: str = "control", *,
                       trials: int = 16, seed: int = 20261025) -> TensorCraftReceipt:
        """Run a bounded supplied scenario through the pinned upstream surrogate.

        Compositions are maps such as ``{"mutalisk": 8, "devourer": 2}`` under
        keys ``A`` and ``B``. Mechanics retain TensorCraft's documented low
        fidelity. These outputs supply hypotheses for native experiments.
        """
        if variant not in _source_spec()["mod_variants"]:
            raise TensorCraftError("unknown TensorCraft mod variant")
        _integer(trials, "trials", 1, 5000)
        _integer(seed, "seed", 0, 2**63 - 1)
        allowed = {"id", "A", "B", "starting_distance_px", "formation_spread_px",
                   "hit_chance", "spore_decay_frames", "focus", "kite",
                   "gas_shadow_price", "horizon_frames", "step_frames",
                   "question", "evidence_status", "assumptions"}
        if not isinstance(scenario, dict) or set(scenario) - allowed:
            raise TensorCraftError("scenario must use the supported TensorCraft fields")
        if not isinstance(scenario.get("id"), str) or not 1 <= len(scenario["id"]) <= 120:
            raise TensorCraftError("scenario.id must be a nonempty string of at most 120 characters")
        for side in ("A", "B"):
            composition = scenario.get(side)
            if not isinstance(composition, dict) or not 1 <= len(composition) <= 32:
                raise TensorCraftError("each side requires a unit-count mapping")
            for kind, count in composition.items():
                if not isinstance(kind, str) or not 1 <= len(kind) <= 80:
                    raise TensorCraftError("unit type names must be bounded strings")
                _integer(count, f"{side}.{kind}", 0, 100)
            _integer(sum(composition.values()), f"total units on {side}", 1, 128)
        for name, low, high in (("horizon_frames", 1, 5000), ("step_frames", 1, 500),
                                ("spore_decay_frames", 1, 100000)):
            if name in scenario:
                _integer(scenario[name], name, low, high)
        for name, low, high in (("hit_chance", 0, 1), ("focus", 0, 1), ("kite", 0, 1),
                                ("gas_shadow_price", 0.01, 100),
                                ("starting_distance_px", 0, 10000),
                                ("formation_spread_px", 0, 10000)):
            if name in scenario:
                value = scenario[name]
                if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                    raise TensorCraftError(f"{name} must be finite and in [{low}, {high}]")
        for name in ("question", "evidence_status"):
            if name in scenario and (not isinstance(scenario[name], str) or len(scenario[name]) > 2000):
                raise TensorCraftError(f"{name} must be a bounded string")
        if "assumptions" in scenario and (not isinstance(scenario["assumptions"], list)
                or len(scenario["assumptions"]) > 32
                or any(not isinstance(v, str) or len(v) > 1000 for v in scenario["assumptions"])):
            raise TensorCraftError("assumptions must be a bounded string list")
        encoded = json.dumps(scenario, allow_nan=False)
        if len(encoded.encode("utf-8")) > 16384:
            raise TensorCraftError("scenario exceeds 16 KiB")
        return self._invoke("simulate_batch", {"scenario": json.loads(encoded), "variant": variant,
                                               "trials": trials, "seed": seed},
                            self.offline_timeout_seconds, "synthetic_surrogate")
