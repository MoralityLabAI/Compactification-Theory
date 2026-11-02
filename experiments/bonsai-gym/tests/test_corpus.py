"""Tests for the meta_sft v0.1 corpus builder and its committed outputs.

Run from the repository root:  python -m pytest experiments/bonsai-gym/tests/test_corpus.py -q

The rebuild and provenance tests read the pinned sources through `git show`; they are skipped
(with the reason) when a source repository is not on disk.
"""
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_corpus.py"
DATA = ROOT / "datasets" / "meta_sft_v0_1"
SPLITS = ("train", "val", "heldout")


@pytest.fixture(scope="module")
def bc():
    spec = importlib.util.spec_from_file_location("build_corpus_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records():
    out = {}
    for name in SPLITS:
        out[name] = [json.loads(line) for line in (DATA / (name + ".jsonl")).read_text(encoding="utf-8").splitlines()]
    return out


def _require_sources(bc):
    missing = [s["local_path"] for s in bc.SOURCES.values() if not Path(s["local_path"]).exists()]
    if missing:
        pytest.skip("source repositories not on disk: %s" % sorted(set(missing)))


def test_manifest_hashes_and_counts_match_files(manifest, records):
    for name in SPLITS:
        raw = (DATA / (name + ".jsonl")).read_bytes()
        info = manifest["files"][name + ".jsonl"]
        assert hashlib.sha256(raw).hexdigest() == info["sha256"]
        assert info["records"] == len(records[name]) == manifest["counts"][name]["records"]
        assert info["bytes"] == len(raw)
    assert manifest["counts"]["total"]["records"] == sum(len(v) for v in records.values())


def test_schema_valid(bc, records):
    seen = set()
    for name in SPLITS:
        assert records[name], name + " is empty"
        for r in records[name]:
            assert bc.validate_record(r)
            assert r["id"] not in seen
            seen.add(r["id"])
            # schema fields from the task statement
            assert r["family"] in bc.FAMILIES
            assert r["evidence_type"] in ("observed", "hypothesis", "synthetic")
            assert set(r["provenance"]) >= {"repo", "commit", "path", "lines_or_section"}


def test_every_family_is_represented(bc, records):
    fams = {r["family"] for v in records.values() for r in v}
    assert fams == set(bc.FAMILIES)


def test_no_lineage_shared_across_splits(records):
    lin = {n: {r["lineage_id"] for r in records[n]} for n in SPLITS}
    assert not (lin["train"] & lin["val"])
    assert not (lin["train"] & lin["heldout"])
    assert not (lin["val"] & lin["heldout"])


def test_heldout_families_absent_from_train_and_val(bc, manifest, records):
    held = set(manifest["heldout_families"])
    assert held == set(bc.HELDOUT_FAMILIES) and len(held) == 2 and "transfer_across_families" in held
    assert {r["family"] for r in records["heldout"]} == held
    for name in ("train", "val"):
        assert not ({r["family"] for r in records[name]} & held)
    # no lineage that appears in heldout appears anywhere else, whatever the family
    held_lineages = {r["lineage_id"] for r in records["heldout"]}
    for name in ("train", "val"):
        assert not ({r["lineage_id"] for r in records[name]} & held_lineages)
    assert manifest["quarantined_records"] == 0 or manifest["quarantined_lineages"]


def test_val_is_about_ten_percent_of_remaining_lineages(bc, records):
    rest = {r["lineage_id"] for n in ("train", "val") for r in records[n]}
    val = {r["lineage_id"] for r in records["val"]}
    assert len(val) == max(1, int(round(bc.VAL_FRACTION * len(rest))))


def test_synthetic_is_marked_and_capped(bc, records):
    allrec = [r for v in records.values() for r in v]
    synth = [r for r in allrec if r["evidence_type"] == "synthetic"]
    assert len(synth) <= bc.MAX_SYNTHETIC_FRACTION * len(allrec)
    for r in synth:
        assert r["provenance"]["repo"] == "synthetic-template"
        assert r["observed_outcome"] is None
        assert "invented" in r["messages"][2]["content"] or "templated" in r["messages"][2]["content"]
    # synthetic records are never held out
    assert not [r for r in records["heldout"] if r["evidence_type"] == "synthetic"]


def test_only_observed_records_carry_outcomes(records):
    for v in records.values():
        for r in v:
            assert (r["observed_outcome"] is not None) == (r["evidence_type"] == "observed")


def test_observed_records_cite_a_real_file(bc, records):
    _require_sources(bc)
    by_sha = {(s["repo"], s["commit"]): s for s in bc.SOURCES.values()}
    checked = 0
    for v in records.values():
        for r in v:
            if r["evidence_type"] != "observed":
                continue
            p = r["provenance"]
            src = by_sha[(p["repo"], p["commit_sha"])]
            res = subprocess.run([bc.GIT, "-C", src["local_path"], "show", "%s:%s" % (p["commit_sha"], p["path"])], capture_output=True)
            assert res.returncode == 0, "%s cites missing file %s@%s" % (r["id"], p["path"], p["commit_sha"][:10])
            nlines = res.stdout.decode("utf-8").count("\n") + 1
            lo, hi = [int(x) for x in p["lines_or_section"].rsplit("lines ", 1)[1].split("-")]
            assert 1 <= lo <= hi <= nlines
            assert p["path"] in src["files_used"]
            checked += 1
    assert checked > 0


def test_observed_outcome_numbers_come_from_the_cited_quotes(bc, records):
    """Every evidence quote registered for a spec is still present in its pinned file."""
    _require_sources(bc)
    n = 0
    for sp in bc.SPECS:
        for q in sp["q"]:
            path, text = q if isinstance(q, tuple) else (sp["path"], q)
            bc.locate(sp["src"], path, text)
            n += 1
    assert n >= 300


def test_rebuild_is_byte_identical(bc):
    _require_sources(bc)
    with tempfile.TemporaryDirectory() as td:
        bc.write_outputs(td)
        for name in ("train.jsonl", "val.jsonl", "heldout.jsonl", "manifest.json"):
            a = hashlib.sha256((Path(td) / name).read_bytes()).hexdigest()
            b = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            assert a == b, name + " differs after rebuild"


def test_sources_config_pins_full_shas(bc):
    for s in bc.SOURCES.values():
        assert len(s["commit"]) == 40 and all(c in "0123456789abcdef" for c in s["commit"]), s["key"]
        assert s["files_used"], s["key"]
        assert s["commit"][:7] in s["ref_label"]
