"""NOR60 source/program roster. Generation requires the frozen commit marker."""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.nor59_authorized_merkle import cohort as prior

SCHEMA = "nor60.first_public_only_60_signed_authorized_program_sources.v1"
CASE_SCHEMA = "nor60.public_exact_signed_NOR45_source.v1"
DOMAIN = "NOR60/SOURCE/v1"
DEGREES = (512, 1024)
STRIDES = (1, 4, 8)
PROFILES = (0, 1, 2, 3, 4)
REPLICATES = (0, 1)
PROFILE_BITS = (2, 8, 16, 2, 8)
PRIOR_ARCHIVES = prior.PRIOR_ARCHIVES + ((
    "research/nor59_authorized_merkle/results/nor59_first_original_hosted.zip",
    "3249ffb69ebd0eba11669609a79e3e52f3d43b176c6320254a469cdb057e3e0b",
    "nor59_first_original_60_public_only.json",
    "be516ca2149cf6d5970842891280f7f7d3407706b9c08b02c4719d153418b0f9", 60),)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def registration_sha():
    path = Path(__file__).with_name("REGISTRATION_SHA.txt")
    require(path.is_file(), "NOR60 registration commit marker is absent; source generation is sealed")
    value = path.read_text(encoding="ascii").strip()
    require(re.fullmatch("[0-9a-f]{40}", value) is not None, "invalid NOR60 registration commit marker")
    return value


def _query_strings(row):
    fields = [name for name in ("source_coefficients_decimal", "public_query_coefficients_decimal")
              if name in row]
    require(len(fields) == 1, "historical row lacks a unique recognized exact coefficient vector")
    values = row[fields[0]]
    require(type(values) is list and bool(values), "historical vector must be a nonempty list")
    require(all(type(v) is str and str(int(v)) == v for v in values),
            "noncanonical signed decimal coefficient in historical vector")
    length_keys = [key for key in ("length", "full_length") if key in row]
    require(len(length_keys) == 1 and type(row[length_keys[0]]) is int
            and row[length_keys[0]] == len(values), "historical vector length mismatch")
    require(values[0] == "1", "historical source is not a unit series")
    return values


def query_content_sha256(row):
    """Same vector gets the same hash across experiment schemas and metadata."""
    values = _query_strings(row)
    return sha256(canonical({"length": len(values), "coefficients_decimal": values})).hexdigest()


def program_input_sha256(row):
    require("authorized_program" in row and "observed_degree" in row,
            "program-input identity requires explicit program and degree")
    return sha256(canonical({"query_content_sha256": query_content_sha256(row),
                             "observed_degree": row["observed_degree"],
                             "authorized_program": row["authorized_program"]})).hexdigest()


def historical_evidence():
    ids, content_ids, program_ids = set(), set(), set()
    archive_records = []
    row_count = 0
    for path, zip_sha, member, raw_sha, count in PRIOR_ARCHIVES:
        p = ROOT / path
        require(sha256(p.read_bytes()).hexdigest() == zip_sha, "historical ZIP SHA mismatch: " + path)
        with ZipFile(p) as archive:
            require(archive.testzip() is None, "historical ZIP integrity failure: " + path)
            raw = archive.read(member)
        require(sha256(raw).hexdigest() == raw_sha, "historical source SHA mismatch: " + path)
        rows = json.loads(raw)["cases"]
        new_ids = {row["source_sha256"] for row in rows}
        require(len(rows) == len(new_ids) == count and ids.isdisjoint(new_ids),
                "historical identity cardinality or overlap failure: " + path)
        ids.update(new_ids)
        for row in rows:
            content_ids.add(query_content_sha256(row))
            if "authorized_program" in row:
                program_ids.add(program_input_sha256(row))
        row_count += len(rows)
        archive_records.append({"path": path, "zip_sha256": zip_sha, "source_member": member,
                                "source_sha256": raw_sha, "source_count": count})
    require(len(ids) == row_count == 588, "expected all 588 historical first identities")
    return {"source_ids": ids, "query_content_ids": content_ids, "program_input_ids": program_ids,
            "source_row_count": row_count, "archive_records": archive_records}


def historical_ids():
    return historical_evidence()["source_ids"]


def program(profile):
    require(type(profile) is int and profile in PROFILES, "unregistered NOR60 profile")
    return prior.program(profile)


def source_seed(degree, stride, profile, replicate):
    require(degree in DEGREES and stride in STRIDES and profile in PROFILES
            and replicate in REPLICATES, "unregistered source grid")
    return int.from_bytes(sha256(f"{DOMAIN}|{degree}|{stride}|{profile}|{replicate}".encode("ascii")).digest(), "big")


def source_series(degree, stride, profile, replicate):
    registration_sha()
    rng = random.Random(source_seed(degree, stride, profile, replicate))
    length, bits = 3 * degree // 2 + 1, PROFILE_BITS[profile]
    values = [1]
    for index in range(1, length):
        if index % stride:
            values.append(0)
        else:
            magnitude = rng.randint(1 << (bits - 1), (1 << bits) - 1)
            sign = (-1 if rng.randrange(2) else 1) if replicate == 0 else (-1 if (index // stride) % 2 else 1)
            values.append(sign * magnitude)
    return tuple(values)


def case_core(degree, stride, profile, replicate):
    return {"case_schema": CASE_SCHEMA, "observed_degree": degree, "full_length": 3 * degree // 2 + 1,
            "source_stride": stride, "program_profile": profile,
            "coefficient_magnitude_bits": PROFILE_BITS[profile], "replicate": replicate,
            "family": "signed_random" if replicate == 0 else "alternating",
            "authorized_program": program(profile),
            "public_query_coefficients_decimal": [str(v) for v in source_series(degree, stride, profile, replicate)]}


def make_fixture():
    registered = registration_sha()
    history = historical_evidence()
    rows = []
    for d in DEGREES:
        for s in STRIDES:
            for p in PROFILES:
                for rep in REPLICATES:
                    core = case_core(d, s, p, rep)
                    rows.append({**core, "source_sha256": sha256(canonical(core)).hexdigest(),
                                 "query_content_sha256": query_content_sha256(core),
                                 "program_input_sha256": program_input_sha256(core)})
    ids = {r["source_sha256"] for r in rows}
    queries = {r["query_content_sha256"] for r in rows}
    inputs = {r["program_input_sha256"] for r in rows}
    require(len(rows) == len(ids) == len(queries) == len(inputs) == 60, "new first input roster contains duplicates")
    require(ids.isdisjoint(history["source_ids"]), "new source identity repeats historical first input")
    require(queries.isdisjoint(history["query_content_ids"]), "new query vector repeats historical first input")
    require(inputs.isdisjoint(history["program_input_ids"]), "new program/input pair repeats history")
    return {"schema": SCHEMA, "first_registration_sha": registered, "source_domain": DOMAIN,
            "degrees": list(DEGREES), "strides": list(STRIDES), "profiles": list(PROFILES),
            "replicates": list(REPLICATES), "public_only": True, "source_count": 60,
            "excluded_first_source_IDs": 588,
            "historical_query_content_unique_count": len(history["query_content_ids"]),
            "historical_program_input_unique_count": len(history["program_input_ids"]),
            "query_content_hash_ignores_experiment_schema_and_metadata": True,
            "new_query_content_disjoint_from_all_historical_rows": True,
            "no_output_root_witness_session_signatures_templates_or_timings": True,
            "historical_archives": history["archive_records"], "cases": rows}


def read_fixture(path):
    raw = Path(path).read_bytes()
    fixture = json.loads(raw)
    require(canonical(fixture) == raw, "first source fixture is not canonical JSON")
    # Rebuild only the source/program specification and hashes after registration.
    # Any omitted, reordered, edited or extra source is rejected.
    require(fixture == make_fixture(), "source fixture differs from frozen NOR60 generator")
    return fixture, sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-only-out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.public_only_out.exists(), "first source fixture already exists; overwrite forbidden")
    raw = canonical(make_fixture())
    args.public_only_out.parent.mkdir(parents=True, exist_ok=True)
    with args.public_only_out.open("xb") as stream:
        stream.write(raw)
    _, digest = read_fixture(args.public_only_out)
    print("NOR60 FIRST ORIGINAL PUBLIC ONLY SHA256", len(raw), digest, flush=True)


if __name__ == "__main__":
    main()
