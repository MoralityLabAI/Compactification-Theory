"""AEG-01 public development gym with a separate exact reference grader.

The participant receives only the public challenge. The original NOR45
interpreter supplies the grader's expected vector. Fixture signing keys are
public TEST keys. This module never creates a confirmatory or sealed split.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.nor43_authenticated_witness import signed_witness as trees
from research.nor44_session_protocol import session_protocol as session
from research.nor45_authorized_program import independent_execution_verifier as reference
from research.nor45_authorized_program import program_authority as authority
from research.nor59_authorized_merkle.cohort import program as program_for_profile

SCHEMA = "aeg01.authorized_exact_development.v1"
SPLITS = ("train", "development")
POLICIES = {
    "original": ("original", "original"),
    "root_only": ("original", "warm"),
    "interpreter_only": ("support_memoized", "original"),
    "combined": ("support_memoized", "warm"),
}
KINDS = ("valid", "authority_signature", "query_binding", "session_binding",
         "provider_output", "expired", "separately_authorized_program")
TRUSTED_FIELDS = {"query_decimal", "session_context", "program_authority_key_hex",
                  "verifier_key_hex", "provider_key_hex", "authority_id", "now_epoch"}
UNTRUSTED_FIELDS = {"certificate", "precommit_receipt", "provider_reference"}
PUBLIC_FIELDS = {"schema", "split", "case_id", "trusted", "untrusted", "public_sha256"}
ACTION_FIELDS = {"case_id", "public_sha256", "decision", "values_decimal", "root_sha256"}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=constant)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_decimals(values):
    require(type(values) is list and 1 <= len(values) <= 1537, "invalid vector length")
    require(all(type(v) is str and len(v) <= 20000 for v in values), "invalid decimal vector")
    try:
        integers = tuple(int(v) for v in values)
    except ValueError as exc:
        raise ValueError("noninteger vector") from exc
    require(all(str(n) == s for n, s in zip(integers, values)), "noncanonical integer")
    return integers


def query_identity(query):
    """Metadata-free identity of an exact vector, shared across gym splits."""
    return digest([str(v) for v in query])


def task_identity(query, program, degree):
    return digest({"query_decimal": [str(v) for v in query],
                   "program": program, "observed_degree": degree})


def _signed_case(query, program, degree, source_id, nonce, wrong_provider=False):
    # This independent reference path never imports the optimized interpreter.
    output = reference.execute_authorized_program(program, query, degree)
    expected_root = reference.independently_computed_root(output)
    provider = trees.test_private_key()
    verifier = session.test_verifier_private_key()
    signer = authority.test_authority_private_key()
    provider_pub = trees.test_reference_public_key()
    verifier_pub = session.public_key_bytes(verifier)
    authority_pub = authority.public_key_of(signer)
    verifier_session = session.VerifierSession(
        source_sha256=source_id, observed_degree=degree,
        coefficient_count=len(query), provider_id=trees.PROVIDER_ID,
        pinned_provider_public_key=provider_pub, verifier_signing_key=verifier,
        expires_epoch=100, nonce=nonce, test_only=True)
    precommit = verifier_session.precommit(
        session.proposal_digest(query, verifier_session.context), now_epoch=1)
    provider_output = list(output)
    if wrong_provider:
        provider_output[8] += 1
    provider_ref = session.sign_provider_reference(
        trees.OutputTree(provider_output), precommit, verifier_session.context,
        verifier_pub, provider, now_epoch=2)
    certificate = authority.sign_authorized_program(
        program=program, query=query, context=verifier_session.context,
        precommit_receipt=precommit, pinned_verifier_public_key=verifier_pub,
        expected_output_merkle_root=expected_root, program_authority_signer=signer,
        now_epoch=2)
    trusted = {"query_decimal": [str(v) for v in query],
               "session_context": verifier_session.context,
               "program_authority_key_hex": authority_pub.hex(),
               "verifier_key_hex": verifier_pub.hex(), "provider_key_hex": provider_pub.hex(),
               "authority_id": authority.TEST_AUTHORITY_ID, "now_epoch": 2}
    untrusted = {"certificate": certificate, "precommit_receipt": precommit,
                 "provider_reference": provider_ref}
    return trusted, untrusted, output, expected_root


def generate(split):
    require(split in SPLITS, "only public train/development splits exist")
    public, referee = [], []
    for degree in (512, 1024):
        for kind_index, kind in enumerate(KINDS):
            seed_bytes = hashlib.sha256(f"AEG01/{split}/{degree}/{kind_index}".encode()).digest()
            rng = random.Random(int.from_bytes(seed_bytes, "big"))
            stride = (1, 4, 8)[kind_index % 3]
            m = 3 * degree // 2 + 1
            query = [1] + [0] * (m - 1)
            # Short support polynomial; full output remains the signed NOR45 length.
            for i in range(stride, min(m, 17 * stride), stride):
                query[i] = rng.choice((-3, -2, -1, 1, 2, 3))
            profile = (kind_index + int(kind == "separately_authorized_program")) % 5
            program = program_for_profile(profile)
            source_id = task_identity(query, program, degree)
            nonce = hashlib.sha256(b"AEG01/public-fixture/" + seed_bytes).digest()
            trusted, untrusted, values, root = _signed_case(
                tuple(query), program, degree, source_id, nonce,
                wrong_provider=kind == "provider_output")
            if kind == "authority_signature":
                untrusted["certificate"]["signature_hex"] = "00" * 64
            elif kind == "query_binding":
                trusted["query_decimal"][1] = str(int(trusted["query_decimal"][1]) + 1)
            elif kind == "session_binding":
                trusted["session_context"]["nonce_hex"] = hashlib.sha256(nonce).hexdigest()
            elif kind == "expired":
                trusted["now_epoch"] = 101
            case_id = hashlib.sha256(b"AEG01/case/" + seed_bytes).hexdigest()
            core = {"schema": SCHEMA, "split": split, "case_id": case_id,
                    "trusted": trusted, "untrusted": untrusted}
            challenge = {**core, "public_sha256": digest(core)}
            valid = kind in ("valid", "separately_authorized_program")
            public.append(challenge)
            referee.append({"case_id": case_id, "public_sha256": challenge["public_sha256"],
                            "expected_accept": valid, "mutation_family": kind,
                            "query_sha256": query_identity(exact_decimals(trusted["query_decimal"])),
                            "task_sha256": task_identity(exact_decimals(trusted["query_decimal"]),
                                                        program, degree),
                            "values_decimal": [str(v) for v in values] if valid else None,
                            "root_sha256": root if valid else None})
    require(len({r["query_sha256"] for r in referee}) == len(referee), "duplicate query")
    require(len({r["task_sha256"] for r in referee}) == len(referee), "duplicate task")
    return public, referee


def kwargs_from_public(challenge):
    require(type(challenge) is dict and set(challenge) == PUBLIC_FIELDS, "public schema mismatch")
    require(challenge["schema"] == SCHEMA and challenge["split"] in SPLITS, "unregistered split/schema")
    core = {k: v for k, v in challenge.items() if k != "public_sha256"}
    require(digest(core) == challenge["public_sha256"], "public challenge changed")
    trusted, untrusted = challenge["trusted"], challenge["untrusted"]
    require(type(trusted) is dict and set(trusted) == TRUSTED_FIELDS, "trusted schema mismatch")
    require(type(untrusted) is dict and set(untrusted) == UNTRUSTED_FIELDS, "envelope schema mismatch")
    keys = {}
    for field in ("program_authority_key_hex", "verifier_key_hex", "provider_key_hex"):
        encoded = trusted[field]
        require(type(encoded) is str and len(encoded) == 64, "invalid pinned key")
        key = bytes.fromhex(encoded)
        require(key.hex() == encoded, "noncanonical pinned key")
        keys[field] = key
    return {"certificate": untrusted["certificate"],
            "query": exact_decimals(trusted["query_decimal"]),
            "session_context": trusted["session_context"],
            "precommit_receipt": untrusted["precommit_receipt"],
            "provider_reference": untrusted["provider_reference"],
            "pinned_program_authority_key": keys["program_authority_key_hex"],
            "expected_program_authority_id": trusted["authority_id"],
            "pinned_verifier_public_key": keys["verifier_key_hex"],
            "pinned_provider_public_key": keys["provider_key_hex"],
            "now_epoch": trusted["now_epoch"]}


def act(challenge, policy="original"):
    require(policy in POLICIES, "unknown fixed policy")
    interpreter_mode, root_mode = POLICIES[policy]
    kwargs = kwargs_from_public(challenge)
    action = {"case_id": challenge["case_id"], "public_sha256": challenge["public_sha256"],
              "decision": "reject", "values_decimal": None, "root_sha256": None}
    try:
        output = reference.verify_authorized_execution(
            **kwargs, _interpreter_policy=interpreter_mode, _root_policy=root_mode)
    except (ValueError, TypeError, KeyError, OverflowError):
        return action
    action.update(decision="accept", values_decimal=[str(v) for v in output["values"]],
                  root_sha256=output["root_sha256"])
    return action


def grade(challenge, truth, action):
    # Validate public framing before matching an answer to a trusted oracle row.
    kwargs_from_public(challenge)
    require(truth["case_id"] == challenge["case_id"] and
            truth["public_sha256"] == challenge["public_sha256"], "oracle/challenge mismatch")
    require(type(action) is dict and set(action) == ACTION_FIELDS, "action schema mismatch")
    require(action["case_id"] == challenge["case_id"] and
            action["public_sha256"] == challenge["public_sha256"], "answer replay or input mismatch")
    require(action["decision"] in ("accept", "reject", "abstain"), "unknown decision")
    valid, accepted = truth["expected_accept"], action["decision"] == "accept"
    require(type(valid) is bool, "invalid referee verdict")
    if accepted:
        exact_decimals(action["values_decimal"])
        require(type(action["root_sha256"]) is str and
                re.fullmatch("[0-9a-f]{64}", action["root_sha256"]) is not None,
                "invalid answer root")
    else:
        require(action["values_decimal"] is None and action["root_sha256"] is None,
                "rejection/abstention must omit output")
    exact = accepted and valid and action["values_decimal"] == truth["values_decimal"] and \
        action["root_sha256"] == truth["root_sha256"]
    if action["decision"] == "abstain":
        verdict = "abstention"
    elif accepted and not valid:
        verdict = "false_acceptance"
    elif valid and accepted:
        verdict = "true_acceptance" if exact else "wrong_exact_output"
    elif valid:
        verdict = "false_rejection"
    else:
        verdict = "true_rejection"
    return {"case_id": challenge["case_id"], "verdict": verdict,
            "correct": verdict in ("true_acceptance", "true_rejection"),
            "mutation_family": truth["mutation_family"]}


class AuthorizedExactGym:
    """One-step Python interface. Each reset copies the public observation."""
    def __init__(self, public, referee):
        self._public, self._referee = copy.deepcopy(public), copy.deepcopy(referee)
        require(len(public) == len(referee) > 0, "empty/mismatched panel")
        self._cursor, self._active = 0, None

    def reset(self, *, index=None):
        if index is None:
            index = self._cursor % len(self._public)
            self._cursor += 1
        require(type(index) is int and 0 <= index < len(self._public), "bad task index")
        self._active = index
        return copy.deepcopy(self._public[index]), {"backend": "python-one-step", "split": self._public[index]["split"]}

    def step(self, action):
        require(self._active is not None, "reset required before each step")
        index = self._active
        verdict = grade(self._public[index], self._referee[index], action)
        self._active = None
        return None, int(verdict["correct"]), True, False, verdict


def _write_new(path, obj):
    raw = canonical(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".aeg01-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export(split, destination):
    public, referee = generate(split)
    destination = Path(destination)
    # Candidate tools receive the public file only. Development labels are public
    # to auditors; separate files are an interface boundary, not access control.
    _write_new(destination / "public" / f"{split}.json", public)
    _write_new(destination / "referee" / f"{split}.json", referee)
    manifest = {"schema": "aeg01.export_manifest.v1", "split": split, "count": len(public),
                "public_sha256": digest(public), "referee_sha256": digest(referee),
                "source_sha256": source_hashes(),
                "evidence_status": "PUBLIC_DEVELOPMENT_FIXTURE"}
    _write_new(destination / f"{split}.manifest.json", manifest)
    return manifest


def source_hashes():
    paths = ["research/authorized_exact_gym/gym.py",
             "research/nor45_authorized_program/independent_execution_verifier.py",
             "research/nor45_authorized_program/program_authority.py",
             "research/nor44_session_protocol/session_protocol.py",
             "research/nor43_authenticated_witness/signed_witness.py",
             "research/nor59_authorized_merkle/cohort.py",
             "research/nor59_authorized_merkle/authorized_root.py",
             "research/nor58_static_merkle/template.py",
             "research/nor52_sparse_inverse/algebra.py",
             "research/nor60_authorized_interpreter/optimized_execution.py"]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def score(public, referee, actions, provenance, *, expected_public_sha256,
          expected_referee_sha256):
    require(digest(public) == expected_public_sha256, "public file differs from trusted manifest")
    require(digest(referee) == expected_referee_sha256, "oracle file differs from trusted manifest")
    require(len(public) == len(referee) == len(actions) > 0, "missing/excess actions or oracle rows")
    expected_ids = {r["case_id"] for r in public}
    require(len(expected_ids) == len(public), "duplicate public case id")
    require({r["case_id"] for r in referee} == expected_ids, "oracle roster mismatch")
    require({r["case_id"] for r in actions} == expected_ids, "answer roster mismatch")
    by_truth = {r["case_id"]: r for r in referee}
    by_action = {r["case_id"]: r for r in actions}
    rows = [grade(c, by_truth[c["case_id"]], by_action[c["case_id"]]) for c in public]
    counts = Counter(r["verdict"] for r in rows)
    return {"schema": "aeg01.score.v1", "evidence_status": "PUBLIC_DEVELOPMENT_SMOKE",
            "case_count": len(rows), "correct_count": sum(r["correct"] for r in rows),
            "verdict_counts": dict(sorted(counts.items())), "rows": rows,
            "public_sha256": digest(public), "referee_sha256": digest(referee),
            "actions_sha256": digest(actions), "source_sha256": source_hashes(),
            "participant_reported_provenance": provenance,
            "independently_verified_model_execution": False,
            "claim_boundary": "Finite public fixtures; no model evaluation, sealed confirmation, or external authority honesty claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    out = sub.add_parser("export")
    out.add_argument("--split", choices=SPLITS, required=True)
    out.add_argument("--out-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--public", type=Path, required=True)
    run.add_argument("--policy", choices=POLICIES, required=True)
    run.add_argument("--out", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--public", type=Path, required=True)
    scoring.add_argument("--referee", type=Path, required=True)
    scoring.add_argument("--answers", type=Path, required=True)
    scoring.add_argument("--manifest", type=Path, required=True)
    scoring.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        print(json.dumps(export(args.split, args.out_dir), sort_keys=True))
    elif args.command == "run":
        public = read_json(args.public)
        start = time.perf_counter_ns()
        actions = [act(c, args.policy) for c in public]
        elapsed = time.perf_counter_ns() - start
        payload = {"actions": actions, "provenance": {"candidate_kind": "fixed_symbolic_reference",
                   "policy": args.policy, "python": platform.python_version(),
                   "platform": platform.platform(), "harness_wall_ns": elapsed,
                   "timing_scope": "one sequential public development pass; diagnostic only",
                   "GPU_used": False, "model_run": False}}
        _write_new(args.out, payload)
        print(json.dumps({"actions": len(actions), "policy": args.policy, "wall_ns": elapsed}))
    else:
        public = read_json(args.public)
        referee = read_json(args.referee)
        answers = read_json(args.answers)
        manifest = read_json(args.manifest)
        require(manifest.get("schema") == "aeg01.export_manifest.v1", "unknown manifest")
        require(manifest.get("source_sha256") == source_hashes(),
                "grader/source code changed since fixture export")
        require(type(answers) is dict and set(answers) == {"actions", "provenance"}, "answer file schema")
        result = score(public, referee, answers["actions"], answers["provenance"],
                       expected_public_sha256=manifest["public_sha256"],
                       expected_referee_sha256=manifest["referee_sha256"])
        _write_new(args.out, result)
        print(json.dumps({k: result[k] for k in ("case_count", "correct_count", "verdict_counts")}))


if __name__ == "__main__":
    main()
