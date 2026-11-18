"""NOR50 public-only fresh-source generation; private exact world restored later.

All first source IDs use new seed offset, 16 new blocks, and are SHA-disjoint
from the original 452 NOR01-42 historical public inputs. This module never
constructs a future while generating or freezing the public fixture.
"""
from __future__ import annotations
import argparse,json,random,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21
from research.nor41_scale_ood import experiment as nor41
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor45_authorized_program import historical_smoke as old

REPS=tuple(range(200,216))
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_regime_blocks")
OFFSET=140050000
SUFFIX_SALT=202611500091
SCHEMA="nor50.first_64_public_only_memoization_sources.v1"
REGISTRATION="a3b7a211e312235f8e9363612aff4173f50a3a4e"
NOR42_ORIGINAL_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"

def canonical(obj):
    return (json.dumps(obj,sort_keys=True,ensure_ascii=True,indent=2)+"\n").encode("ascii")

def original_ids():
    earlier=set(nor42.history())
    assert len(earlier)==388
    fixture,grade=old.load_original()
    assert len(fixture["cases"])==64
    original={row["full_public_input_sha256"] for row in fixture["cases"]}
    assert len(original)==64 and earlier.isdisjoint(original)
    return earlier|original

def source_seed(rep,d,fam):
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(d)+10007*FAMILIES.index(fam))

def new_source(rep,d,fam,seen):
    """Replicate NOR42 public-source grammar with disjoint source seed."""
    q=3*d//2
    rng=random.Random(source_seed(rep,d,fam))
    for attempt in range(100):
        program=nor21.new_header_program(rng)
        demos=tuple((x:=nor41.public_input(rng,fam,q),
                     corpus.execute(x,program)) for _ in range(2))
        query=nor41.public_input(rng,fam,q)
        c=corpus.Case("nor50-public-source",fam,demos,query,(),None)
        headers=corpus.possible_headers(c)
        if len(headers)!=1:continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=program[:4]:continue
        source_sha=nor21.public_digest(c,d)
        if source_sha in seen:continue
        mask=nor21.native_mask(c,h,d)
        if not mask:continue
        predictions=nor21.candidate_predictions(query,h,q)
        if len({predictions[j] for j in range(16) if mask&(1<<j)})!=1:
            continue
        original_class=4*corpus.OPS.index(program[4])+corpus.OPS.index(program[5])
        assert mask&(1<<original_class)
        seen.add(source_sha)
        return {
            "split":"TEST","pair_replicate":rep,"degree":d,
            "query_degree":q,"family":fam,
            "source_seed":source_seed(rep,d,fam),
            "selected_attempt":attempt,
            "full_public_input_sha256":source_sha,
            "public_header":{"base":h.base,"delay":h.delay,
                             "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(x[:d+1]),list(y[:d+1])] for x,y in demos],
            "public_query_input":list(query),
            "original_native_public_mask":mask
        }
    raise RuntimeError(f"NOR50 source exhausted 100 attempts: {rep} D{d} {fam}")

def make_fixture():
    old_ids=original_ids()
    seen=set(old_ids)
    rows=[new_source(rep,d,fam,seen)
          for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(rows)==64 and len(seen)==len(old_ids)+64
    return {
        "schema":SCHEMA,"registration_first_commit":REGISTRATION,
        "source_offset":OFFSET,"replicates":list(REPS),
        "degrees":list(DEGREES),"families":list(FAMILIES),
        "public_source_units":64,"excluded_original_source_count":len(old_ids),
        "no_private_suffix_or_true_future":True,
        "no_trace_or_benchmark_timings":True,
        "prior_NOR42_original_source_sha256":NOR42_ORIGINAL_SHA,
        "cases":rows
    }

def read_fixture(path):
    raw=Path(path).read_bytes()
    data=json.loads(raw)
    assert data["schema"]==SCHEMA and data["registration_first_commit"]==REGISTRATION
    assert data["source_offset"]==OFFSET
    assert data["replicates"]==list(REPS) and data["degrees"]==list(DEGREES)
    assert data["families"]==list(FAMILIES)
    assert data["public_source_units"]==64 and data["excluded_original_source_count"]==452
    assert data["no_private_suffix_or_true_future"] and data["no_trace_or_benchmark_timings"]
    assert data["prior_NOR42_original_source_sha256"]==NOR42_ORIGINAL_SHA
    rows=data["cases"]
    ids=[r["full_public_input_sha256"] for r in rows]
    assert len(rows)==len(set(ids))==64
    assert set(ids).isdisjoint(original_ids())
    assert Counter(r["pair_replicate"] for r in rows)=={rep:4 for rep in REPS}
    assert {(r["pair_replicate"],r["degree"],r["family"]) for r in rows}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    assert all(r["source_seed"]==source_seed(
        r["pair_replicate"],r["degree"],r["family"]) for r in rows)
    assert all(not set(r).intersection(("program","future","target",
       "hidden_beta","hidden_transform","private","trace","timings")) for r in rows)
    return data,sha256(raw).hexdigest()

def restore_worlds(row):
    """Reconstruct first private clean/hidden exact worlds only after SHA freeze."""
    rep,d,fam=row["pair_replicate"],row["degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor41.public_input(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor41.public_input(rng,fam,3*d//2)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    clean=corpus.execute(query,planted)
    random_suffix=random.Random(nor42.seed(SUFFIX_SALT,row["full_public_input_sha256"]))
    beta=random_suffix.choice(nor42.BETAS)
    op=random_suffix.choice(corpus.OPS)
    hidden=nor42.hidden_future(query,clean,d,beta,op)
    return tuple(clean),tuple(hidden),beta,op

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only-out",required=True,type=Path)
    a=p.parse_args()
    raw=canonical(make_fixture())
    a.source_only_out.parent.mkdir(parents=True,exist_ok=True)
    a.source_only_out.write_bytes(raw)
    print("NOR50 FIRST 64 SOURCE-ONLY FIXTURE BYTES SHA",len(raw),sha256(raw).hexdigest())

if __name__=="__main__":main()
