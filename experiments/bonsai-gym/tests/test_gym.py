import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bonsai_gym.gym import (LinUCB, Ledger, RegressionGuard, Task, Thresholdout, Vault,  # noqa: E402
                            VaultBudgetExceeded, contamination, paired_gain, run_cycle)


def test_vault_budget_and_log(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    v = Vault("sealed", [{"q": "a", "answer": "x"}], budget=2, ledger=led)
    r = v.confirm(lambda items: [0.73], "test")
    assert r["reported_mean"] == 0.75 and r["bits_total"] > 4.39
    v.confirm(lambda items: [0.0], "test")
    with pytest.raises(VaultBudgetExceeded):
        v.confirm(lambda items: [1.0], "test")
    # budget survives a restart: reads are counted from the ledger
    assert Vault("sealed", [{"q": "a", "answer": "x"}], budget=2, ledger=led).reads == 2
    assert "answer" not in v.texts()[0]


def test_paired_gain_requires_ci_above_zero():
    rng = np.random.default_rng(0)
    before = rng.random(100)
    assert paired_gain(before, before + 0.2)["verified"]
    noise = paired_gain(before, before + rng.normal(0, 0.3, 100))
    assert not noise["verified"] or noise["ci95"][0] > 0
    # reused dev set: the floor after 50 generations x 4 candidates on 100 items is 0.163
    assert not paired_gain(before, before + 0.15, reuse=(50, 4))["verified"]
    assert paired_gain(before, before + 0.2, reuse=(50, 4))["verified"]


def test_contamination_flags_overlap():
    train = ["the quick brown fox jumps over the lazy dog and keeps running far away from here"]
    sealed = ["unrelated text entirely", "x the quick brown fox jumps over the lazy dog and keeps running far away"]
    c = contamination(train, sealed, n=8)
    assert c["contaminated"] == 1 and c["indices"] == [1]


def test_thresholdout_answers_train_when_close():
    t = Thresholdout(threshold=0.1, sigma=0.0)
    assert t.query(0.5, 0.52) == 0.5
    assert t.query(0.9, 0.5) == 0.5 and t.overfit_signals == 1


def test_cycle_learns_the_productive_arm_and_reverts_regressions(tmp_path):
    rng = np.random.default_rng(1)
    level = {"good": 0.0, "bad": 0.0, "harm": 0.0}
    last = {}
    effect = {"good": 0.05, "bad": 0.0, "harm": 0.05}

    def mk(name):
        def train(r):
            last[name] = level[name]
            level[name] += effect[name]
            if name == "harm":
                level["anchor_hit"] = level.get("anchor_hit", 0) + 0.1
            return {"r": r}

        def revert():
            level[name] = last[name]
            if name == "harm":
                level["anchor_hit"] -= 0.1
        return Task(name, features=lambda: np.ones(2), train=train,
                    dev_eval=lambda r: list(level[name] + rng.normal(0, 0.01, 64)), revert=revert)

    tasks = {k: mk(k) for k in ("good", "bad", "harm")}
    guard = RegressionGuard(tolerance=0.05, baseline={"anchor": 1.0})
    anchors = lambda: {"anchor": 1.0 - level.get("anchor_hit", 0.0)}
    led = Ledger(tmp_path / "c.jsonl")
    recs = run_cycle(tasks, LinUCB(list(tasks), 2, alpha=0.1), anchors, guard, led, rounds=30)
    picks = [r["arm"] for r in recs]
    assert picks[-10:].count("good") >= 8
    assert all(not r["kept"] for r in recs if r["arm"] == "harm")
    assert level["harm"] == 0.0 and level.get("anchor_hit", 0.0) == pytest.approx(0.0)
    assert len(led.read()) == 30
