"""NOR-21 independent source-locked grader and complete timing/memory audit.

Imports only the ORIGINAL RRC25 source interpreter, never the NOR21
benchmark or source generator. Independently regenerate the selected
private source programs from the registered public source seeds, and
rederive all exact native acceptance masks at degrees 512/1024.

Reference program identity and future truth are reconstructed only for
auditing the frozen original post-selection results. They are never
available to the benchmark's public candidate ranking functions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus

FIXTURE_SCHEMA="nor21.public_only_source_36_degree512_1024.v1"
SCHEMA="nor21.degree_512_1024_out_of_grammar_cpu_memory.v1"
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
KINDS=("in_grammar","public_corrupt","future_shift")
MODES=("native16","cached16","stream16","chunk4")
SEED_OFFSET=94210000
BOOTSTRAPS=4096
BOOT_SEED=202611210521
RECEIPT=HERE/"results"/"nor21_first_frozen_receipt.json"


def close(x,y):
    return math.isclose(float(x),float(y),rel_tol=1e-12,abs_tol=1e-7)


def public_case(entry):
    d=entry["observation_degree"]
    h=entry["public_header"]
    assert d in DEGREES and entry["query_degree"]==d*3//2
    header=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    demos=tuple((tuple(src),tuple(y)) for src,y in entry["demos"])
    query=tuple(entry["public_query_input"])
    assert len(demos)==2 and all(len(src)==len(y)==d+1 for src,y in demos)
    assert len(query)==entry["query_degree"]+1
    assert query[0]==1
    c=corpus.Case("nor21-source-only",entry["family"],demos,query,(),None)
    assert c.program is None and c.future==()
    return c,header


def public_source_hash(c,degree):
    return sha256(json.dumps(
        [[list(src[:degree+1]),list(y[:degree+1])]
         for src,y in c.demos]+[list(c.query)],
        separators=(",",":")).encode()).hexdigest()


def candidate_program(header,j):
    return (header.base,header.delay,header.spacing,header.amplitude,
            corpus.OPS[j//4],corpus.OPS[j%4])


def independent_native_mask(c,header,degree):
    return sum(1<<j for j in range(16)
               if all(corpus.execute(src[:degree+1],
                      candidate_program(header,j))==y[:degree+1]
                      for src,y in c.demos))


def inverse_operands(f):
    assert f[0]==1
    result=[1]
    for i in range(1,len(f)):
        result.append(-sum(f[k]*result[i-k] for k in range(1,i+1)))
    assert tuple(result)==corpus.d.operate(f,"R")
    return max(abs(z).bit_length() for z in result)


def nonzero(rng):
    return rng.choice((-2,-1,1,2))


def independent_source(rng,family,degree):
    if family=="wide_dense_small":
        return (1,)+tuple(nonzero(rng) for _ in range(degree))
    if family=="wide_sparse_small":
        return (1,)+tuple(nonzero(rng) if rng.random()<.12 else 0
                          for _ in range(degree))
    if family=="wide_rational_short":
        polynomial=[1]+[0]*degree
        polynomial[1]=rng.choice((-1,1))
        for _ in range(3):
            polynomial[rng.randint(2,12)]+=nonzero(rng)
        return corpus.d.operate(tuple(polynomial),"R")
    raise ValueError("unknown registered family")


def reconstruct_sealed_program(entry):
    """Independent generator replay, not importing NOR21 code."""
    seed=entry["source_seed"]
    r=entry["replicate"];d=entry["observation_degree"]
    family=entry["family"];kind=entry["case_kind"]
    expected=(corpus.SEED+SEED_OFFSET+1000003*r+
              100009*DEGREES.index(d)+10007*FAMILIES.index(family)+
              331*KINDS.index(kind))
    assert seed==expected
    rng=random.Random(seed)
    q=entry["query_degree"]
    program=None
    demos=None
    query=None
    for attempt in range(entry["selected_attempt"]+1):
        program=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((f:=independent_source(rng,family,q),
                     corpus.execute(f,program)) for _ in range(2))
        query=independent_source(rng,family,q)
    assert len(demos)==2 and query is not None
    if kind=="public_corrupt":
        src,y=demos[1]
        changed=list(y)
        changed[d]+=1
        demos=(demos[0],(src,tuple(changed)))
    original=[[
        list(src[:d+1]),list(y[:d+1])]
        for src,y in demos]
    assert original==entry["demos"]
    assert list(query)==entry["public_query_input"]
    c=corpus.Case("nor21-independent-replay",family,demos,query,(),None)
    assert public_source_hash(c,d)==entry["full_public_input_sha256"]
    return program,query


def grade_replay(entry,choice):
    prog,query=reconstruct_sealed_program(entry)
    kind=entry["case_kind"]
    target=corpus.execute(query,prog)
    if kind=="future_shift":
        changed=list(target)
        changed[-1]+=1
        target=tuple(changed)
    chosen=choice["chosen_class"]
    if chosen is None:
        correct=False
    else:
        h=entry["public_header"]
        hdr=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
        predicted=corpus.execute(query,candidate_program(hdr,chosen))
        correct=(predicted==target)
    return {
        "future_correct":correct,
        "public_certificate_issued":chosen is not None,
        "false_accept_outside_family":chosen is not None and not correct,
        "public_no_match_abstention":(choice["mask"]==0 and chosen is None),
        "private_truth_used_only_for_postchoice_grade":True,
    }


def block_bootstrap(rows,key):
    blocks=defaultdict(list)
    for r in rows:
        blocks[r["replicate"]].append(r)
    assert set(blocks)=={0,1}
    assert len(blocks[0])==len(blocks[1])==9
    rng=random.Random(BOOT_SEED+77)
    values=[]
    for _ in range(BOOTSTRAPS):
        sample=[x for rep in (rng.randrange(2) for _ in range(2))
                for x in blocks[rep]]
        values.append(statistics.median(x[key] for x in sample))
    values.sort()
    return values[int(.025*(BOOTSTRAPS-1))],values[int(.975*(BOOTSTRAPS-1))]


def audit(fixture_bytes,result_bytes):
    fixture=json.loads(fixture_bytes)
    result=json.loads(result_bytes)
    assert fixture["schema"]==FIXTURE_SCHEMA
    assert fixture["source_seed_offset"]==SEED_OFFSET
    assert fixture["observation_degrees"]==list(DEGREES)
    assert fixture["query_degrees"]==[768,1536]
    assert fixture["families"]==list(FAMILIES)
    assert fixture["case_kinds"]==list(KINDS)
    assert fixture["public_only"]
    assert fixture["private_programs_and_true_future_outputs_absent"]
    assert result["schema"]==SCHEMA
    assert result["public_fixture_sha256"]==sha256(fixture_bytes).hexdigest()
    assert result["source_cases"]==len(result["records"])==len(fixture["cases"])==36
    assert result["modes"]==list(MODES)
    assert result["warmups"]==1 and result["timed_repetitions"]==3
    assert len({s["full_public_input_sha256"] for s in fixture["cases"]})==36
    assert Counter(x["observation_degree"] for x in fixture["cases"])=={
        512:18,1024:18}
    assert Counter(x["case_kind"] for x in fixture["cases"])=={
        "in_grammar":12,"public_corrupt":12,"future_shift":12}
    assert Counter(x["replicate"] for x in fixture["cases"])=={0:18,1:18}
    assert result["scope"]["future_not_in_public_fixture"]
    assert result["scope"]["all_public_candidate_decisions_frozen_before_sealed_grading"]
    for entry,r in zip(fixture["cases"],result["records"],strict=True):
        degree=entry["observation_degree"]
        c,h=public_case(entry)
        assert public_source_hash(c,degree)==entry["full_public_input_sha256"]
        assert r["full_public_input_sha256"]==entry["full_public_input_sha256"]
        assert r["observation_degree"]==degree
        assert r["query_degree"]==entry["query_degree"]
        assert r["family"]==entry["family"]
        assert r["case_kind"]==entry["case_kind"]
        mask=independent_native_mask(c,h,degree)
        assert mask==entry["exact_public_accept_mask"]==r["original_reference_mask"]
        pred={corpus.execute(c.query,candidate_program(h,j))
              for j in range(16) if mask&(1<<j)}
        expected_unanimity=(len(pred)==1) if mask else False
        assert expected_unanimity==entry["hypothetical_query_future_unanimous"]
        assert r["reference_unanimity"]==expected_unanimity
        chosen=next((j for j in range(16) if mask&(1<<j)),None) if expected_unanimity else None
        assert chosen==r["chosen_class"]
        assert r["all_four_public_modes_exactly_equal"]
        assert r["sealed_grade"]["private_truth_used_only_for_postchoice_grade"]
        reference_grade=grade_replay(entry,{"chosen_class":chosen,"mask":mask})
        for key,val in reference_grade.items():
            assert val==r["sealed_grade"][key],(degree,entry["case_kind"],key)
        if entry["case_kind"]=="future_shift":
            witness=r["sealed_grade"]["counterfactual_public_same_future_different"]
            assert witness["same_public_sha256"]==entry["full_public_input_sha256"]
            assert witness["hidden_future_targets_differ"]
            assert witness["valid_world_future_sha256"]!=witness[
                "mutant_world_future_sha256"]
        else:
            assert r["sealed_grade"]["counterfactual_public_same_future_different"] is None
        assert set(r["times_ns"])==set(MODES)
        assert set(r["medians_ns"])==set(MODES)
        assert set(r["full_tracemalloc"])==set(MODES)
        assert set(r["verifier_tracemalloc"])==set(MODES)
        for mode in MODES:
            samples=r["times_ns"][mode]
            assert len(samples)==3
            for row in samples:
                assert set(row)=={
                    "query_ns","verification_ns","certificate_ns","total_ns"}
                assert all(type(v) is int and v>=0 for v in row.values())
                assert row["total_ns"]==row["query_ns"]+row[
                    "verification_ns"]+row["certificate_ns"]
                assert row["total_ns"]>0
            for metric in ("query_ns","verification_ns","certificate_ns","total_ns"):
                assert close(statistics.median(x[metric] for x in samples),
                             r["medians_ns"][mode][metric])
            for key in ("full_tracemalloc","verifier_tracemalloc"):
                rec=r[key][mode]
                assert rec["mask"]==mask
                assert rec["peak_bytes"]>=rec["current_bytes"]>=0
            assert r["reciprocal_R_invocations_per_demo"][mode]==(
                8+16*int(h.base=="R") if mode=="native16" else 1)
        actual_ratio=r["full_tracemalloc"]["stream16"]["peak_bytes"]/r[
            "full_tracemalloc"]["cached16"]["peak_bytes"]
        actual_speed=r["medians_ns"]["native16"]["total_ns"]/max(
            1,r["medians_ns"]["stream16"]["total_ns"])
        assert close(actual_ratio,r["stream_over_cached_complete_peak_ratio"])
        assert close(actual_speed,r["native_over_stream_complete_cpu_ratio"])
        actual_inverse=max(
            inverse_operands(src[:degree+1]) for src,y in c.demos)
        actual_inverse=max(actual_inverse,inverse_operands(c.query))
        assert actual_inverse==r["max_reciprocal_inverse_output_bits"]
    for degree in DEGREES:
        rows=[x for x in result["records"] if x["observation_degree"]==degree]
        assert len(rows)==18
        stats=result["by_degree"][str(degree)]
        assert stats["cases"]==18
        speed=statistics.median(x["native_over_stream_complete_cpu_ratio"] for x in rows)
        mem=statistics.median(x["stream_over_cached_complete_peak_ratio"] for x in rows)
        wins=sum(x["stream_over_cached_complete_peak_ratio"]<1 for x in rows)
        assert close(stats["median_cpu_native_over_stream_ratio"],speed)
        assert close(stats["median_memory_stream_over_cached_ratio"],mem)
        assert stats["stream_peak_lower_cases"]==wins
        assert stats["primary_degree_pass"]==(
            speed>=1.5 and mem<=.80 and wins>=14)
        for key,ci in (
            ("stream_over_cached_complete_peak_ratio","memory_block_bootstrap"),
            ("native_over_stream_complete_cpu_ratio","cpu_block_bootstrap")):
            lo,hi=block_bootstrap(rows,key)
            assert stats[ci]["draws"]==4096
            assert close(lo,stats[ci]["p2_5"])
            assert close(hi,stats[ci]["p97_5"])
        assert stats["grade_by_kind"]["in_grammar"]["future_correct"]==6
        assert stats["grade_by_kind"]["public_corrupt"]["certified"]==0
        assert stats["grade_by_kind"]["future_shift"]["out_of_family_false_accept"]==6
    assert result["primary_all_degrees_pass"]==all(
        z["primary_degree_pass"] for z in result["by_degree"].values())
    assert set(result["separate_process_RSS"])==set(MODES)
    return {
        "public_source_cases":36,
        "fresh_unique_complete_public_inputs":36,
        "exact_native_512_1024_masks":36,
        "sealed_true_futures_replayed":36,
        "timing_samples_checked":36*4*3,
        "separate_memory_records_checked":36*4*2,
        "bootstrap_draws_recomputed":4096*2*2,
        "in_scope_correct_certificates":12,
        "public_corrupt_zero_mask_abstentions":12,
        "hidden_future_counterfactual_false_accepts_outside_family":12,
        "degree512_and_1024_engineering_pass":result[
            "primary_all_degrees_pass"],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--full",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    if args.archive:
        with ZipFile(args.archive) as z:
            assert sorted(z.namelist())==["nor21_fixture.json","nor21_full.json"]
            assert z.testzip() is None
            fixture=z.read("nor21_fixture.json")
            full=z.read("nor21_full.json")
    elif args.fixture and args.full:
        fixture=args.fixture.read_bytes()
        full=args.full.read_bytes()
    elif args.fixture or args.full:
        p.error("both fixture and full required")
    else:
        print("NOR21 INDEPENDENT AUDIT AWAITS ORIGINAL PUBLIC BENCHMARK")
        return
    if RECEIPT.exists():
        pinned=json.loads(RECEIPT.read_text())
        assert len(fixture)==pinned["original_public_fixture_bytes"]
        assert len(full)==pinned["original_benchmark_bytes"]
        assert sha256(fixture).hexdigest()==pinned["original_public_fixture_sha256"]
        assert sha256(full).hexdigest()==pinned["original_benchmark_sha256"]
    r=audit(fixture,full)
    print("NOR21 INDEPENDENT DEGREE SCALE AND OUTSIDE FAMILY AUDIT PASSED",r)
    if args.check:
        assert r["timing_samples_checked"]==432
        assert r["hidden_future_counterfactual_false_accepts_outside_family"]==12


if __name__=="__main__":
    main()
