"""Run the four symbolic AEG-01 policies on its public development panel."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from research.authorized_exact_gym import gym


def run():
    public, referee = gym.generate("development")
    _, training_referee = gym.generate("train")
    for field in ("query_sha256", "task_sha256"):
        gym.require({r[field] for r in training_referee}.isdisjoint(
            {r[field] for r in referee}), "public development overlaps training content")
    reports = {}
    for policy in gym.POLICIES:
        start = time.perf_counter_ns()
        actions = [gym.act(challenge, policy) for challenge in public]
        elapsed = time.perf_counter_ns() - start
        reports[policy] = gym.score(
            public, referee, actions,
            {"candidate_kind": "fixed_symbolic_implementation", "policy": policy,
             "diagnostic_single_sequential_pass_ns": elapsed, "model_run": False},
            expected_public_sha256=gym.digest(public), expected_referee_sha256=gym.digest(referee))
    source_hashes = gym.source_hashes()
    source_hashes["research/authorized_exact_gym/smoke.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"schema": "aeg01.four_symbolic_policy_development_smoke.v1",
            "evidence_status": "PUBLIC_DEVELOPMENT_SMOKE", "train_count": len(training_referee),
            "development_count": len(referee), "query_and_task_splits_disjoint": True,
            "policies": reports, "all_policies_correct": all(
                report["correct_count"] == len(referee) for report in reports.values()),
            "source_sha256": source_hashes, "python": platform.python_version(),
            "platform": platform.platform(), "dependencies": {name: importlib.metadata.version(name)
                for name in ("cryptography", "gmpy2")},
            "model_run": False, "GPU_used": False, "sealed_test_generated": False,
            "scope": "Finite public development correctness. Elapsed times are one-pass diagnostics; NOR60 owns registered comparative performance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gym.require(not args.out.exists(), "first smoke receipt already exists")
    result = run()
    gym._write_new(args.out, result)
    print(json.dumps({"all_policies_correct": result["all_policies_correct"],
                      "development_count": result["development_count"],
                      "policy_count": len(result["policies"])}))


if __name__ == "__main__":
    main()
