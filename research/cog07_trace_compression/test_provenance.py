from __future__ import annotations

import copy
import json

import pytest
from audit_provenance import SCHEMA, audit, candidate_features, digest


def fixture():
    world = {"atoms": [0, 1, 2, 3], "facts": [0], "goal": 3,
             "rules": [{"premises": [0], "conclusion": 1},
                       {"premises": [1], "conclusion": 2},
                       {"premises": [2], "conclusion": 3}]}
    root = digest({"original_world": world})
    task = {"schema_version": SCHEMA, "root_id": root, "task_id": digest(world),
            "split": "train", "family": "chain", "world": world,
            "provenance": {"kind": "synthetic_horn_proof_dag", "source_world_sha256": root}}
    task["task_digest"] = digest(task)
    # These expected transition receipts are specified directly, independently
    # of the auditor's checked_step implementation.
    specifications = [
        (2, False, "missing_premise", None, [0], False),
        (0, True, "proved_lemma", 1, [0, 1], False),
        (0, True, "duplicate_lemma", None, [0, 1], False),
        (1, True, "proved_lemma", 2, [0, 1, 2], False),
        (2, True, "proved_lemma", 3, [0, 1, 2, 3], True),
    ]
    known, history, events = [0], [], []
    previous = digest({"task_digest": task["task_digest"], "policy": "teacher_with_sentinels"})
    for visit, (action, accepted, code, atom, after, solved) in enumerate(specifications):
        receipt = {"accepted": accepted, "code": code, "next_known": after,
                   "new_fact": atom, "solved": solved, "verifier_calls": 1}
        event = {"visit": visit, "observation": {"world": copy.deepcopy(world),
                 "known": list(known), "history": copy.deepcopy(history)},
                 "action": action, "verification": receipt, "previous_digest": previous,
                 "cost": {"verifier_calls": 1, "proposal_calls": 1}}
        event["step_digest"] = digest(event)
        events.append(event)
        previous, known = event["step_digest"], after
        history.append({"action": action, "accepted": accepted, "code": code, "new_fact": atom})
    trace = {"schema_version": SCHEMA, "root_id": root, "task_id": task["task_id"],
             "task_digest": task["task_digest"], "split": "train", "family": "chain",
             "policy": "teacher_with_sentinels", "provenance": copy.deepcopy(task["provenance"]),
             "steps": events, "solved": True, "verifier_calls": 5, "final_digest": previous}
    trace["trace_digest"] = digest(trace)
    ranking = []
    for known, accepted, useful in [([0], [True, False, False], [True, False, False]),
                                    ([0, 1], [True, True, False], [False, True, False]),
                                    ([0, 1, 2], [True, True, True], [False, False, True])]:
        ranking.append({"schema_version": SCHEMA, "root_id": root, "task_id": task["task_id"],
                        "split": "train", "family": "chain",
                        "public": {"world": copy.deepcopy(world), "known": known, "history": []},
                        "candidates": [{"action": i, "features": candidate_features(world, set(known), i)}
                                       for i in range(3)],
                        "labels": {"accepted": accepted, "useful": useful},
                        "collection_cost": {"verifier_calls": 3}})
    return [task], [trace], ranking


def write_inputs(tmp_path, data):
    paths = []
    for name, rows in zip(("tasks", "traces", "ranking"), data):
        path = tmp_path / (name + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)
    return paths


def rehash_trace(trace):
    previous = digest({"task_digest": trace["task_digest"], "policy": trace["policy"]})
    for event in trace["steps"]:
        event["previous_digest"] = previous
        event["step_digest"] = digest({k: v for k, v in event.items() if k != "step_digest"})
        previous = event["step_digest"]
    trace["final_digest"] = previous
    trace["trace_digest"] = digest({k: v for k, v in trace.items() if k != "trace_digest"})


def test_independent_receipts_and_complete_state_coverage(tmp_path):
    result = audit(*write_inputs(tmp_path, fixture()))
    assert result["passed"]
    assert result["teacher"] == {"verifier_calls": 5, "missing_premise": 1,
                                 "proved_lemma": 3, "duplicate_lemma": 1, "completed_teacher_roots": 1}
    assert result["ranking"]["ranking_states_authenticated"] == 3
    assert result["ranking"]["candidate_labels_and_features_checked"] == 9


@pytest.mark.parametrize("kind", ["success", "state", "cost"])
def test_rehashed_forgery_is_rejected(tmp_path, kind):
    data = fixture()
    trace = data[1][0]
    if kind == "success":
        trace["steps"][0]["verification"]["accepted"] = True
    elif kind == "state":
        trace["steps"][0]["observation"]["known"].append(3)
    else:
        trace["steps"][0]["cost"]["verifier_calls"] = 0
    rehash_trace(trace)
    with pytest.raises(ValueError):
        audit(*write_inputs(tmp_path, data))


@pytest.mark.parametrize("kind", ["trace", "state", "candidate"])
def test_missing_trace_state_or_candidate_is_rejected(tmp_path, kind):
    data = fixture()
    if kind == "trace":
        data[1].clear()
    elif kind == "state":
        data[2].pop()
    else:
        data[2][0]["candidates"].pop()
    with pytest.raises(ValueError):
        audit(*write_inputs(tmp_path, data))


@pytest.mark.parametrize("kind", ["label", "features", "known"])
def test_substituted_ranking_evidence_is_rejected(tmp_path, kind):
    data = fixture()
    row = data[2][0]
    if kind == "label":
        row["labels"]["useful"][1] = True
    elif kind == "features":
        row["candidates"][0]["features"][2] = 0.0
    else:
        row["public"]["known"].append(3)
    with pytest.raises(ValueError):
        audit(*write_inputs(tmp_path, data))


@pytest.mark.parametrize("kind", ["split", "world_alias"])
def test_registry_custody_and_independent_root_count_are_checked(tmp_path, kind):
    data = fixture()
    other = copy.deepcopy(data[0][0])
    if kind == "split":
        other["split"] = "ordinary_test"
    else:
        other["root_id"] = "f" * 64
        other["provenance"]["source_world_sha256"] = "f" * 64
    other["task_digest"] = digest({k: v for k, v in other.items() if k != "task_digest"})
    data[0].append(other)
    with pytest.raises(ValueError):
        audit(*write_inputs(tmp_path, data))
