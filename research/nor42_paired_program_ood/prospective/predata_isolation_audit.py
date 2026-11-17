"""NOR-42 pre-data guard: all tests historical-source-only or synthetic.

No NEW NOR42 public source, native program, hidden true future, CAL/EVAL
world or graded outcome may be generated before the original preregistration
and full public-source-only fixture freeze. After fixture generation this
guard validates bytes without constructing or restoring the private truth.
"""
from __future__ import annotations
import argparse,ast,json,sys
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
FIRST="9bf353b7c76535e581ff026098cd682914a90ef0"
SCHEMA="nor42.paired_grammar_ood_public_only_reps100to115.v1"
PRIOR_SOURCE_SHA="4073090e5888ed10283d82dc5153758d299ed7cfe9c753be39ba2c2bd4c7119d"
PRIOR_GRADE_SHA="4efa6bdb1dd3656d0fbb6639065a8f2d934bd4a1acb952c128748cea9918de58"
PRIOR_ZIP_SHA="0752c8c736ebe4460d02d655d7513662093d5c9dd2ea6e632a0ea76705469d12"
BANNED={"new_source","make_fixture","restore_worlds","evaluate"}


def canonical(obj):
    return (json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=True)+"\n").encode("ascii")


def source_test_calls(body,path):
    tree=ast.parse(body,filename=str(path))
    found=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            f=node.func
            attr=(f.attr if isinstance(f,ast.Attribute) else
                  f.id if isinstance(f,ast.Name) else None)
            if attr in BANNED or attr in ("eval","exec"):
                found.append((node.lineno,attr))
        if isinstance(node,ast.ImportFrom):
            for part in node.names:
                if part.name in BANNED:found.append((node.lineno,part.name))
    return found


def inspect_predata_tests():
    files=sorted(HERE.glob("test*.py"))
    assert files,"at least one historical-only unit test required"
    results=[]
    for path in files:
        raw=path.read_bytes()
        bad=source_test_calls(raw.decode("utf-8"),path)
        assert not bad,(path.name,bad)
        results.append({"path":str(path.relative_to(ROOT)),
                        "sha256":sha256(raw).hexdigest()})
    for bad in ("x.new_source(100,512,'wide_dense_small',set())",
                "m=make_fixture()",
                "x.restore_worlds(row,pred)",
                "evaluate(new_fixture)"):
        assert source_test_calls(bad,"synthetic_bad.py")
    for good in ("x.history()","x.hidden_future(old_source,prior,512,1,'I')",
                 "x.nor41.restore_native_target(old_row,pred)"):
        assert not source_test_calls(good,"synthetic_good.py")
    return results


def old_source_ids():
    from research.nor41_scale_ood import experiment as nor41
    prior=nor41.history()
    assert len(prior)==324
    archive=ROOT/"research/nor41_scale_ood/results/nor41_original_first_hosted.zip"
    raw=archive.read_bytes()
    assert len(raw)==3388660 and sha256(raw).hexdigest()==PRIOR_ZIP_SHA
    with ZipFile(archive) as z:
        assert z.testzip() is None
        src=z.read("nor41_fixture.json")
        grade=z.read("nor41_full.json")
    assert len(src)==9618649 and sha256(src).hexdigest()==PRIOR_SOURCE_SHA
    assert len(grade)==5007355 and sha256(grade).hexdigest()==PRIOR_GRADE_SHA
    x=json.loads(src)
    assert x["schema"]=="nor41.scale_ood_public_only_reps84to99.v1"
    assert len(x["cases"])==64
    ids=sorted(prior+[r["full_public_input_sha256"] for r in x["cases"]])
    assert len(ids)==len(set(ids))==388
    return ids


def check_frozen_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    old=old_source_ids()
    assert x["schema"]==SCHEMA
    assert x["registered_first_commit"]==FIRST
    assert x["replicates"]==list(range(100,116))
    assert x["degrees"]==[512,1024]
    assert x["families"]==["wide_dense_small","wide_regime_blocks"]
    assert x["source_offset"]==130042000
    assert x["public_only"] and x["planted_and_private_target_absent"]
    assert x["hidden_beta_transform_absent"] and x["attack_witness_and_audit_absent"]
    assert x["source_units"]==64 and x["private_truth_worlds"]==128
    assert x["excluded_historical_sources"]==388
    assert x["historical_source_ids_sha256"]==sha256(canonical(old)).hexdigest()
    rows=x["cases"]
    ids=[r["full_public_input_sha256"] for r in rows]
    assert len(ids)==len(set(ids))==64 and set(ids).isdisjoint(old)
    assert {(r["pair_replicate"],r["degree"],r["family"]) for r in rows}=={
        (rep,d,fam) for rep in range(100,116) for d in (512,1024)
        for fam in ("wide_dense_small","wide_regime_blocks")}
    assert all(r["split"]=="TEST" and
               r["paired_world_labels"]==["clean","hidden"] and
               r["paired_world_prior"]==[1,2] and
               not set(r).intersection(
                   ("program","future","target","attack","witness",
                    "audit","beta","hidden_transform"))
               for r in rows)
    return {"bytes":len(raw),"sha256":sha256(raw).hexdigest(),
            "source_pairs":len(rows),"historical_excluded":len(old)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path)
    p.add_argument("--no-tests",action="store_true")
    args=p.parse_args()
    scanned=[] if args.no_tests else inspect_predata_tests()
    out={
        "schema":"nor42.predata_source_isolation_audit.v1",
        "registered_first_commit":FIRST,
        "historical_sha_excluded":len(old_source_ids()),
        "predata_tests":scanned
    }
    if args.fixture:
        out["frozen_public_fixture"]=check_frozen_fixture(args.fixture)
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
