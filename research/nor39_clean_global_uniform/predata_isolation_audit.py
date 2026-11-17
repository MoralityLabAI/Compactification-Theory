"""NOR-39 pre-data source-isolation guard.

Deliberately uses AST/static inspection for pre-data tests. It never imports
the NOR39 evaluator, invokes its source generator, or reconstructs a private
NOR39 native target. Optionally verifies a *preexisting* archived public-only
fixture, after the independent fixture-creation stage has completed.

The audit guards the exact source-preview failure seen in NOR38. It certifies
the inspected tree and fixture format, not the full history of every external
process on a machine.
"""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIRST_REGISTRATION="4a5f3750d240d46c031e0cb432fb7b1a884cfbe9"
PREVIOUS_SHA="37fa2aeedeb5ab54bc1082ec2d50f8a084c4f4a7d6279acd5d122396d760e504"
PREVIOUS_GRADE_SHA="9c69d663d4d1511e664e930c0f8ed42d8d6f59c4d10557323fb5ee54478c036d"
TESTING_BANNED={"new_source","make_fixture","restore_native_target",
                "regenerate_native","evaluate"}
SCHEMA="nor39.clean_global_uniform_public_only_reps68to75.v1"


def _forbidden_calls(source: str, filename: str):
    tree=ast.parse(source,filename=filename)
    found=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            f=node.func
            name=f.attr if isinstance(f,ast.Attribute) else (
                f.id if isinstance(f,ast.Name) else None)
            if name in TESTING_BANNED:
                # Archived NOR37/NOR38 native-target tests are permitted;
                # those identities were frozen and graded before NOR39.
                receiver=f.value if isinstance(f,ast.Attribute) else None
                explicitly_historical=(
                    isinstance(receiver,ast.Attribute) and
                    receiver.attr in ("nor37","nor38"))
                if not explicitly_historical:
                    found.append((node.lineno,name))
            if isinstance(f,ast.Name) and f.id in ("eval","exec"):
                found.append((node.lineno,f.id))
        if isinstance(node,ast.ImportFrom):
            for part in node.names:
                if part.name in TESTING_BANNED:
                    found.append((node.lineno,part.name))
    return found


def assert_no_new_source_from_tests():
    """Checks all pre-data test modules without executing them."""
    files=sorted(HERE.glob("test*.py"))
    assert files, "NOR39 needs a historical-source-only unit test module"
    checked=[]
    for path in files:
        body=path.read_text(encoding="utf-8")
        bad=_forbidden_calls(body,path.as_posix())
        assert not bad,(path.as_posix(),bad)
        checked.append({"file":str(path.relative_to(ROOT)),
                        "sha256":sha256(body.encode()).hexdigest()})
    # The scanner must reject a preview of a NOR39 source/program.
    for trap in ("x.new_source(68,512,'wide_dense_small',set())",
                 "source=make_fixture()",
                 "x.restore_native_target(row,pred)",
                 "native=regenerate_native(row)",
                 "score=evaluate(fixture)"):
        assert _forbidden_calls(trap,"synthetic_trap.py")
    assert not _forbidden_calls("x.history()\nx.components('global7_mix')\n",
                                "historical_only.py")
    assert not _forbidden_calls("x.nor38.restore_native_target(row,pred)",
                                "archived_history.py")
    return checked


def historical_source_shas():
    """Read previous source IDs only, without referencing any NOR39 source."""
    from research.nor38_global_uniform import experiment as old
    prior=old.history()
    archived=ROOT/"research/nor38_global_uniform/results/nor38_original_first_hosted.zip"
    with ZipFile(archived) as z:
        assert z.testzip() is None
        original=z.read("nor38_fixture.json")
        grade=z.read("nor38_full.json")
    assert len(original)==9319507 and sha256(original).hexdigest()==PREVIOUS_SHA
    assert len(grade)==929756 and sha256(grade).hexdigest()==PREVIOUS_GRADE_SHA
    fixture=json.loads(original)
    assert len(prior)==180 and len(fixture["cases"])==48
    all_ids=prior+[r["full_public_input_sha256"] for r in fixture["cases"]]
    assert len(set(all_ids))==len(all_ids)==228
    return set(all_ids)


def check_frozen_public_only(path: Path):
    """Verify a prior, already-created fixture; no source synthesis."""
    raw=path.read_bytes()
    x=json.loads(raw)
    assert x["schema"]==SCHEMA
    assert x["registered_first_commit"]==FIRST_REGISTRATION
    assert x["replicates"]==list(range(68,76))
    assert x["degrees"]==[512,1024]
    assert x["source_offset"]==130039000
    assert x["excluded_historical_sources"]==228
    assert x["public_only"] and x["planted_and_private_target_absent"]
    assert x["attack_witness_and_audit_absent"]
    ids=[row["full_public_input_sha256"] for row in x["cases"]]
    assert len(ids)==len(set(ids))==48
    assert set(ids).isdisjoint(historical_source_shas())
    assert sorted((r["replicate"],r["degree"],r["family"]) for r in x["cases"])==sorted(
        (rep,d,fam) for rep in range(68,76) for d in (512,1024)
        for fam in ("wide_dense_small","wide_sparse_small","wide_rational_short"))
    assert all(row["split"]=="TEST" and
               not any(s in row for s in
                       ("program","future","target","attack","witness","audit"))
               for row in x["cases"])
    return {"bytes":len(raw),"sha256":sha256(raw).hexdigest(),"source_units":len(ids)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=Path,
                        help="Existing source-only fixture AFTER freeze; never creates data")
    parser.add_argument("--no-tests",action="store_true",
                        help="Used only by post-freeze fixture-only stage")
    args=parser.parse_args()
    if args.no_tests:
        assert args.fixture, "fixture path required for fixture-only check"
        scanned=[]
    else:
        scanned=assert_no_new_source_from_tests()
    out={"schema":"nor39.predata_source_isolation_audit.v1",
         "registration_sha":FIRST_REGISTRATION,
         "predata_tests":scanned,
         "historical_source_units_excluded":len(historical_source_shas())}
    if args.fixture:
        out["frozen_fixture"]=check_frozen_public_only(args.fixture)
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
