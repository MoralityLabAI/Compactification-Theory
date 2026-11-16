"""NOR-21: exact degree scaling and explicit out-of-family counterfactuals.

The public fixture contains two degree-512/1024 demonstrations, public
query INPUT of degree-768/1536, and the frozen public header. The private
generating program and hidden query OUTPUT are NEVER part of a public
fixture. Sealed correctness is evaluated separately *after* all policy
choices are frozen. Every policy is an exact finite interpreter.

Scientific boundaries:
 - Publicly corrupted demonstration => all 16 candidate checks fail;
   responsible verifier abstains.
 - Hidden future override => same complete public data as a valid world,
   but different target, so public-only certainty can fail outside the
   enumerated generating family even for exact mask unanimity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor17_prefix_certificates import experiment as nor17
from research.nor19_streamed_verifier import benchmark as nor19
from research.nor18r_runtime import benchmark as nor18r

SCHEMA="nor21.degree_512_1024_out_of_grammar_cpu_memory.v1"
FIXTURE_SCHEMA="nor21.public_only_source_36_degree512_1024.v1"
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
KINDS=("in_grammar","public_corrupt","future_shift")
MODES=("native16","cached16","stream16","chunk4")
REPLICATES=2
SAMPLE_COUNT=len(DEGREES)*len(FAMILIES)*len(KINDS)*REPLICATES
SEED_OFFSET=94210000
WARMUPS=1
TRIALS=3
BOOTSTRAPS=4096
BOOT_SEED=202611210521
SPEED_THRESHOLD=1.5
MEMORY_RATIO_THRESHOLD=.80
MIN_MEMORY_WINS_PER_DEGREE=14


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def random_coeff(rng):
    return rng.choice((-2,-1,1,2))


def public_source(rng,family,degree):
    if family not in FAMILIES or degree<512:
        raise ValueError("unsupported source family or degree")
    if family=="wide_dense_small":
        return (1,)+tuple(random_coeff(rng) for _ in range(degree))
    if family=="wide_sparse_small":
        return (1,)+tuple(random_coeff(rng) if rng.random()<.12 else 0
                          for _ in range(degree))
    # Rational-series input generated from a short polynomial. Its actual
    # coefficients may be far wider than ±2 source polynomial coefficients.
    p=[1]+[0]*degree
    p[1]=rng.choice((-1,1))
    for _ in range(3):
        p[rng.randint(2,12)]+=random_coeff(rng)
    return corpus.d.operate(tuple(p),"R")


def seed(rep,degree,family,kind):
    if not (0<=rep<REPLICATES and degree in DEGREES
            and family in FAMILIES and kind in KINDS):
        raise ValueError("outside registered NOR21 source grid")
    return (corpus.SEED+SEED_OFFSET+1000003*rep+
            100009*DEGREES.index(degree)+10007*FAMILIES.index(family)+
            331*KINDS.index(kind))


def new_header_program(rng):
    return (rng.choice(corpus.OPS),
            rng.choice(corpus.DELAYS),
            rng.choice(corpus.SPACINGS),
            rng.choice(corpus.AMPLITUDES),
            rng.choice(corpus.OPS),
            rng.choice(corpus.OPS))


def public_digest(c,public_degree):
    return sha256(json.dumps(
        [[list(src[:public_degree+1]),list(y[:public_degree+1])]
         for src,y in c.demos]+[list(c.query)],
        separators=(",",":")).encode()).hexdigest()


def program(h,j):
    if not 0<=j<16:
        raise ValueError("invalid candidate")
    return (h.base,h.delay,h.spacing,h.amplitude,
            corpus.OPS[j//4],corpus.OPS[j%4])


def candidate_predictions(src,h,degree):
    """Exact common candidate query outputs via four public transforms."""
    assert len(src)>=degree+1 and degree>=max(corpus.DELAYS)
    return nor17.candidate_outputs(src,h,degree)


def native_mask(c,h,public_degree):
    """Original RRC25 exact execute semantics with explicit degree."""
    bits=0
    for j in range(16):
        p=program(h,j)
        if all(corpus.execute(src[:public_degree+1],p)==
               y[:public_degree+1] for src,y in c.demos):
            bits|=1<<j
    return bits


def cached_mask(c,h,degree):
    """Original eager all-16 candidate materialization, now at degree n."""
    mask=65535
    for src,y in c.demos:
        outputs=candidate_predictions(src,h,degree)
        mask &= sum(1<<j for j in range(16)
                    if outputs[j]==y[:degree+1])
    return mask


def stream_mask(c,h,degree):
    """Single-demo cache, exact coefficientwise comparisons and early exit."""
    mask=65535
    k=h.delay
    delta=h.spacing
    amp=h.amplitude
    for src,y in c.demos:
        f=src[:degree+1]
        observed=y[:degree+1]
        transforms={op:corpus.d.operate(f,op) for op in corpus.OPS}
        base=transforms[h.base]
        for j in range(16):
            if not(mask&(1<<j)):
                continue
            u=transforms[corpus.OPS[j//4]]
            v=transforms[corpus.OPS[j%4]]
            for n in range(degree+1):
                value=base[n]
                if n==k:
                    value+=amp
                if n>=k+delta:
                    value+=amp*u[n-k-delta]
                if n>=k+2*delta:
                    value+=amp*v[n-k-2*delta]
                if value!=observed[n]:
                    mask &= ~(1<<j)
                    break
        del base,u,v,transforms
        if mask==0:
            break
    return mask


def chunk4_mask(c,h,degree):
    mask=65535
    for src,y in c.demos:
        f=src[:degree+1]
        observed=y[:degree+1]
        transforms={op:corpus.d.operate(f,op) for op in corpus.OPS}
        for start in range(0,16,4):
            batch=[(j,tuple(nor19.build_one_candidate(
                h,transforms,j,degree)))
                for j in range(start,start+4) if mask&(1<<j)]
            for j,out in batch:
                if out!=observed:
                    mask &= ~(1<<j)
            if batch:
                del out
            del batch
        del transforms
        if mask==0:
            break
    return mask


def public_verifier(c,h,degree,mode):
    if mode=="native16":
        return native_mask(c,h,degree)
    if mode=="cached16":
        return cached_mask(c,h,degree)
    if mode=="stream16":
        return stream_mask(c,h,degree)
    if mode=="chunk4":
        return chunk4_mask(c,h,degree)
    raise ValueError("unregistered NOR21 comparator")


def certify(c,h,degree,query_degree,mode,measure=False):
    """Public selection only. Mask 0 yields safe abstention, no assertion."""
    if mode not in MODES:
        raise ValueError("unregistered mode")
    start=time.perf_counter_ns() if measure else None
    outputs=candidate_predictions(c.query,h,query_degree)
    mid=time.perf_counter_ns() if measure else None
    mask=public_verifier(c,h,degree,mode)
    end=time.perf_counter_ns() if measure else None
    if mask:
        futures={outputs[j] for j in range(16) if mask&(1<<j)}
        unanimous=len(futures)==1
        chosen=next((j for j in range(16) if mask&(1<<j)),None) if unanimous else None
        status="public_unanimous_family_conditional" if unanimous else "public_ambiguous_abstain"
    else:
        unanimous=False
        chosen=None
        status="no_public_candidate_abstain"
    if measure:
        finished=time.perf_counter_ns()
        timings={
            "query_ns":mid-start,
            "verification_ns":end-mid,
            "certificate_ns":finished-end,
            "total_ns":finished-start,
        }
        assert timings["total_ns"]==sum(timings[k] for k in (
            "query_ns","verification_ns","certificate_ns"))
        return {"mask":mask,"unanimous":unanimous,"chosen_class":chosen,
                "status":status,"times":timings}
    return {"mask":mask,"unanimous":unanimous,
            "chosen_class":chosen,"status":status}


def sample_source(rep,degree,family,kind,used=None):
    if used is None:
        used=set()
    rng=random.Random(seed(rep,degree,family,kind))
    query_degree=degree*3//2
    for attempt in range(80):
        p=new_header_program(rng)
        demos=tuple((f:=public_source(rng,family,query_degree),
                     corpus.execute(f,p)) for _ in range(2))
        query=public_source(rng,family,query_degree)
        c=corpus.Case(f"nor21-r{rep}-d{degree}-{family}-{kind}-{attempt}",
                      family,demos,query,(),None)
        headers=corpus.possible_headers(c)
        if len(headers)!=1:
            continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=p[:4]:
            continue
        if kind=="public_corrupt":
            src,y=c.demos[1]
            damaged=list(y)
            damaged[degree]+=1
            c=corpus.Case(c.identifier,c.family,
                (c.demos[0],(src,tuple(damaged))),c.query,(),None)
            assert len(corpus.possible_headers(c))==1
        digest=public_digest(c,degree)
        if digest in used:
            continue
        mask=native_mask(c,h,degree)
        if kind=="public_corrupt":
            if mask:
                continue
            unique=False
        else:
            if not mask:
                continue
            preds=candidate_predictions(c.query,h,query_degree)
            unique=len({preds[j] for j in range(16) if mask&(1<<j)})==1
            if not unique:
                continue
            planted_index=corpus.OPS.index(p[4])*4+corpus.OPS.index(p[5])
            assert mask&(1<<planted_index)
        used.add(digest)
        return c,h,p,attempt,mask,unique,digest
    raise RuntimeError("NOR21 source exhausted "+str((rep,degree,family,kind)))


def fixture():
    seen=set()
    records=[]
    for rep in range(REPLICATES):
        for degree in DEGREES:
            for family in FAMILIES:
                for kind in KINDS:
                    c,h,p,attempt,mask,unique,digest=sample_source(
                        rep,degree,family,kind,seen)
                    records.append({
                        "replicate":rep,
                        "observation_degree":degree,
                        "query_degree":degree*3//2,
                        "family":family,
                        "case_kind":kind,
                        "source_seed":seed(rep,degree,family,kind),
                        "selected_attempt":attempt,
                        "public_header":{"base":h.base,"delay":h.delay,
                              "spacing":h.spacing,"amplitude":h.amplitude},
                        "demos":[[list(src[:degree+1]),list(y[:degree+1])]
                                 for src,y in c.demos],
                        "public_query_input":list(c.query),
                        "full_public_input_sha256":digest,
                        "exact_public_accept_mask":mask,
                        "hypothetical_query_future_unanimous":unique,
                        "max_public_input_coefficient_bits":max(
                            abs(v).bit_length()
                            for src,y in c.demos for v in src[:degree+1]),
                        "max_public_observed_output_bits":max(
                            abs(v).bit_length()
                            for src,y in c.demos for v in y[:degree+1]),
                    })
    assert len(records)==SAMPLE_COUNT
    assert len({r["full_public_input_sha256"] for r in records})==SAMPLE_COUNT
    assert Counter(x["observation_degree"] for x in records)=={
        512:18,1024:18}
    assert Counter(x["case_kind"] for x in records)=={
        kind:12 for kind in KINDS}
    return {
        "schema":FIXTURE_SCHEMA,
        "source_seed_offset":SEED_OFFSET,
        "observation_degrees":list(DEGREES),
        "query_degrees":[768,1536],
        "families":list(FAMILIES),
        "case_kinds":list(KINDS),
        "source_replicates":REPLICATES,
        "public_only":True,
        "private_programs_and_true_future_outputs_absent":True,
        "cases":records,
    }


def public_case(entry):
    d=entry["observation_degree"]
    q=entry["query_degree"]
    assert d in DEGREES and q==3*d//2
    h=entry["public_header"]
    header=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    demos=tuple((tuple(f),tuple(y)) for f,y in entry["demos"])
    src=tuple(entry["public_query_input"])
    assert len(demos)==2
    assert all(len(f)==len(y)==d+1 for f,y in demos)
    assert len(src)==q+1
    c=corpus.Case("nor21-public",entry["family"],demos,src,(),None)
    assert c.program is None and c.future==()
    assert public_digest(c,d)==entry["full_public_input_sha256"]
    return c,header


def frozen_fixture(path):
    raw=Path(path).read_bytes()
    f=json.loads(raw)
    assert f["schema"]==FIXTURE_SCHEMA
    assert f["source_seed_offset"]==SEED_OFFSET
    assert f["observation_degrees"]==list(DEGREES)
    assert f["query_degrees"]==[768,1536]
    assert f["families"]==list(FAMILIES)
    assert f["case_kinds"]==list(KINDS)
    assert f["source_replicates"]==REPLICATES
    assert f["public_only"] and f["private_programs_and_true_future_outputs_absent"]
    assert len(f["cases"])==SAMPLE_COUNT
    assert len({r["full_public_input_sha256"] for r in f["cases"]})==SAMPLE_COUNT
    return f,sha256(raw).hexdigest()


def allocation(c,h,d,q,mode,verifier_only=False):
    gc.collect()
    tracemalloc.start()
    try:
        if verifier_only:
            actual=public_verifier(c,h,d,mode)
        else:
            actual=certify(c,h,d,q,mode)["mask"]
        current,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"peak_bytes":peak,"current_bytes":current,"mask":actual}


def rss_worker(fixture_path,mode):
    f,_=frozen_fixture(fixture_path)
    initial=nor19.get_rss_kib()
    for entry in f["cases"]:
        c,h=public_case(entry)
        certify(c,h,entry["observation_degree"],entry["query_degree"],mode)
    final=nor19.get_rss_kib()
    return {
        "mode":mode,"cases":SAMPLE_COUNT,
        "baseline_peak_rss_KiB":initial,
        "after_peak_rss_KiB":final,
        "additional_peak_rss_KiB":(max(0,final-initial)
            if initial is not None and final is not None else None),
    }


def isolated_rss(fixture_path):
    outcomes={}
    for mode in MODES:
        worker=subprocess.run([sys.executable,str(Path(__file__).resolve()),
                "--fixture",str(fixture_path),"--rss-worker",mode],
            check=True,capture_output=True,text=True)
        res=json.loads(worker.stdout.strip().splitlines()[-1])
        assert res["mode"]==mode and res["cases"]==SAMPLE_COUNT
        outcomes[mode]=res
    return outcomes


def bootstrap(rows,key):
    groups=defaultdict(list)
    for r in rows:
        groups[r["replicate"]].append(r)
    assert set(groups)=={0,1}
    assert len(groups[0])==len(groups[1])
    rng=random.Random(BOOT_SEED+77)
    values=[]
    for _ in range(BOOTSTRAPS):
        sample=[x for rep in (rng.randrange(2) for _ in range(2))
                for x in groups[rep]]
        values.append(statistics.median(z[key] for z in sample))
    values.sort()
    return {
        "draws":BOOTSTRAPS,
        "unit":"one complete 9-case source replicate per public degree",
        "p2_5":values[int(.025*(BOOTSTRAPS-1))],
        "p97_5":values[int(.975*(BOOTSTRAPS-1))],
    }


def graded_after_freeze(entry,choice):
    """The ONLY function regenerating true program and hidden future.

    Must be called strictly after every benchmark policy's decision is
    frozen. Its p/future are never passed to any public selection routine.
    """
    rep=entry["replicate"];d=entry["observation_degree"]
    family=entry["family"];kind=entry["case_kind"]
    source,h,planted,attempt,mask,unanimous,digest=sample_source(
        rep,d,family,kind)
    assert digest==entry["full_public_input_sha256"]
    assert attempt==entry["selected_attempt"]
    assert mask==entry["exact_public_accept_mask"]
    assert unanimous==entry["hypothetical_query_future_unanimous"]
    valid_future=corpus.execute(source.query,planted)
    if kind=="future_shift":
        changed=list(valid_future)
        changed[-1]+=1
        hidden_target=tuple(changed)
        assert hidden_target!=valid_future
        # These two generative worlds have identical complete observed
        # public data and query input; only future target differs.
        counterfactual={
            "same_public_sha256":digest,
            "valid_world_future_sha256":sha256(json.dumps(
                valid_future,separators=(",",":")).encode()).hexdigest(),
            "mutant_world_future_sha256":sha256(json.dumps(
                hidden_target,separators=(",",":")).encode()).hexdigest(),
            "hidden_future_targets_differ":True,
        }
    else:
        hidden_target=valid_future
        counterfactual=None
    chosen=choice["chosen_class"]
    selected=corpus.execute(source.query,program(h,chosen)) if chosen is not None else None
    if chosen is None:
        correct=False
    else:
        correct=(selected==hidden_target)
    if kind=="in_grammar":
        assert choice["unanimous"] and correct
    elif kind=="public_corrupt":
        assert choice["mask"]==0 and chosen is None
    else:
        assert choice["unanimous"] and chosen is not None and not correct
    return {
        "future_correct":correct,
        "public_certificate_issued":chosen is not None,
        "false_accept_outside_family":chosen is not None and not correct,
        "public_no_match_abstention":choice["mask"]==0 and chosen is None,
        "counterfactual_public_same_future_different":counterfactual,
        "private_truth_used_only_for_postchoice_grade":True,
    }


def summarize(rows):
    ratios=[x["stream_over_cached_complete_peak_ratio"] for x in rows]
    speed=[x["native_over_stream_complete_cpu_ratio"] for x in rows]
    return {
        "cases":len(rows),
        "median_memory_stream_over_cached_ratio":statistics.median(ratios),
        "stream_peak_lower_cases":sum(x<1 for x in ratios),
        "median_cpu_native_over_stream_ratio":statistics.median(speed),
        "median_complete_ms":{mode:statistics.median(
            x["medians_ns"][mode]["total_ns"] for x in rows)/1e6
            for mode in MODES},
        "median_complete_peak_KiB":{mode:statistics.median(
            x["full_tracemalloc"][mode]["peak_bytes"] for x in rows)/1024
            for mode in MODES},
        "median_verifier_only_peak_KiB":{mode:statistics.median(
            x["verifier_tracemalloc"][mode]["peak_bytes"] for x in rows)/1024
            for mode in MODES},
        "largest_observed_source_bits":max(
            x["max_public_source_bits"] for x in rows),
        "largest_observed_inverse_bits":max(
            x["max_reciprocal_inverse_output_bits"] for x in rows),
        "grade_by_kind":{kind:{
            "cases":sum(x["case_kind"]==kind for x in rows),
            "certified":sum(x["case_kind"]==kind and x["sealed_grade"][
                "public_certificate_issued"] for x in rows),
            "future_correct":sum(x["case_kind"]==kind and x["sealed_grade"][
                "future_correct"] for x in rows),
            "out_of_family_false_accept":sum(x["case_kind"]==kind and x[
                "sealed_grade"]["false_accept_outside_family"] for x in rows),
        } for kind in KINDS},
    }


def benchmark(fixture_path,warmups=WARMUPS,trials=TRIALS):
    if warmups!=WARMUPS or trials!=TRIALS:
        raise ValueError("registered warmups/trials changed")
    fixture,fixture_sha=frozen_fixture(fixture_path)
    observations=[]
    for pos,entry in enumerate(fixture["cases"]):
        c,h=public_case(entry)
        d=entry["observation_degree"];q=entry["query_degree"]
        expected=(entry["exact_public_accept_mask"],
                  entry["hypothetical_query_future_unanimous"])
        predictions={mode:certify(c,h,d,q,mode) for mode in MODES}
        assert all((pred["mask"],pred["unanimous"])==expected
                   for pred in predictions.values())
        assert len({pred["chosen_class"] for pred in predictions.values()})==1
        trials_by_mode={name:[] for name in MODES}
        rng=random.Random(BOOT_SEED+pos*100003)
        for run_index in range(WARMUPS+TRIALS):
            order=list(MODES)
            rng.shuffle(order)
            for mode in order:
                gc.collect()
                was_enabled=gc.isenabled()
                gc.disable()
                try:
                    measured=certify(c,h,d,q,mode,measure=True)
                finally:
                    if was_enabled:
                        gc.enable()
                assert (measured["mask"],measured["unanimous"])==expected
                if run_index>=WARMUPS:
                    trials_by_mode[mode].append(measured["times"])
        assert all(len(s)==TRIALS for s in trials_by_mode.values())
        medians={mode:{
            label:float(statistics.median(z[label] for z in ts))
            for label in ("query_ns","verification_ns","certificate_ns","total_ns")}
            for mode,ts in trials_by_mode.items()}
        complete_memory={mode:allocation(c,h,d,q,mode) for mode in MODES}
        verifier_memory={mode:allocation(c,h,d,q,mode,True) for mode in MODES}
        assert all(z["mask"]==expected[0] for z in complete_memory.values())
        assert all(z["mask"]==expected[0] for z in verifier_memory.values())
        # Profile actual normalized R inverse operands outside timing.
        observed_inverses=[nor18r.exact_inverse_operand_ledger(
            src[:d+1]) for src,y in c.demos]
        observed_inverses.append(
            nor18r.exact_inverse_operand_ledger(c.query))
        max_inverse_bits=max(x["max_inverse_coefficient_bits"]
                             for x in observed_inverses)
        ratio=complete_memory["stream16"]["peak_bytes"]/max(1,
              complete_memory["cached16"]["peak_bytes"])
        cpu=medians["native16"]["total_ns"]/max(
            1,medians["stream16"]["total_ns"])
        # Seal the decision vector before accessing future-generation code.
        frozen_selection=dict(predictions["stream16"])
        sealed_grade=graded_after_freeze(entry,frozen_selection)
        observations.append({
            "replicate":entry["replicate"],"observation_degree":d,
            "query_degree":q,"family":entry["family"],
            "case_kind":entry["case_kind"],
            "full_public_input_sha256":entry["full_public_input_sha256"],
            "original_reference_mask":expected[0],
            "reference_unanimity":expected[1],
            "chosen_class":frozen_selection["chosen_class"],
            "public_certificate_status":frozen_selection["status"],
            "all_four_public_modes_exactly_equal":True,
            "times_ns":trials_by_mode,"medians_ns":medians,
            "full_tracemalloc":complete_memory,
            "verifier_tracemalloc":verifier_memory,
            "stream_over_cached_complete_peak_ratio":ratio,
            "native_over_stream_complete_cpu_ratio":cpu,
            "max_public_source_bits":entry["max_public_input_coefficient_bits"],
            "max_reciprocal_inverse_output_bits":max_inverse_bits,
            "reciprocal_R_invocations_per_demo":{
                "native16":8+16*int(h.base=="R"),
                "cached16":1,"stream16":1,"chunk4":1},
            "sealed_grade":sealed_grade,
        })
    assert len(observations)==SAMPLE_COUNT
    by_degree={}
    for d in DEGREES:
        rows=[r for r in observations if r["observation_degree"]==d]
        assert len(rows)==18
        stats=summarize(rows)
        stats["memory_block_bootstrap"]=bootstrap(rows,
            "stream_over_cached_complete_peak_ratio")
        stats["cpu_block_bootstrap"]=bootstrap(rows,
            "native_over_stream_complete_cpu_ratio")
        stats["primary_degree_pass"]=(
            stats["median_memory_stream_over_cached_ratio"]<=
            MEMORY_RATIO_THRESHOLD and
            stats["stream_peak_lower_cases"]>=MIN_MEMORY_WINS_PER_DEGREE and
            stats["median_cpu_native_over_stream_ratio"]>=SPEED_THRESHOLD)
        assert stats["grade_by_kind"]["in_grammar"]["future_correct"]==6
        assert stats["grade_by_kind"]["public_corrupt"]["certified"]==0
        assert stats["grade_by_kind"]["future_shift"][
            "out_of_family_false_accept"]==6
        by_degree[str(d)]=stats
    return {
        "schema":SCHEMA,
        "fixture_schema":FIXTURE_SCHEMA,
        "public_fixture_sha256":fixture_sha,
        "host":{"os":platform.platform(),"python":sys.version,
                "machine":platform.machine(),"compiler":platform.python_compiler(),
                "cpus":os.cpu_count()},
        "source_cases":SAMPLE_COUNT,
        "registered_observation_degrees":list(DEGREES),
        "registered_query_degrees":[768,1536],
        "source_replicates":REPLICATES,
        "families":list(FAMILIES),
        "case_kinds":list(KINDS),
        "modes":list(MODES),
        "warmups":WARMUPS,"timed_repetitions":TRIALS,
        "records":observations,
        "by_degree":by_degree,
        "overall":summarize(observations),
        "separate_process_RSS":isolated_rss(fixture_path),
        "primary_all_degrees_pass":all(x["primary_degree_pass"]
            for x in by_degree.values()),
        "scope":{
            "future_not_in_public_fixture":True,
            "all_public_candidate_decisions_frozen_before_sealed_grading":True,
            "generator_membership_scope_tested_with_outside_family_future":True,
            "zero_public_mask_requires_abstention":True,
            "all_36_complete_public_inputs_distinct":True,
            "same_query_first_order_in_all_four_modes":True,
            "tracemalloc_separate_untimed_execution":True,
            "ratio_ci_based_on_only_two_9case_replicates":True,
            "no_claim_of_arbitrary_theorem_prover_soundness":True,
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--warmups",type=int,default=WARMUPS)
    p.add_argument("--trials",type=int,default=TRIALS)
    p.add_argument("--rss-worker",choices=MODES)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    if args.make_fixture:
        result=fixture()
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(canonical(result))
        print("NOR21 PUBLIC FIXTURE SEALED",len(result["cases"]),
              sha256(args.make_fixture.read_bytes()).hexdigest())
        return
    if args.fixture is None:
        p.error("must pass --fixture or --make-fixture")
    if args.rss_worker:
        print(json.dumps(rss_worker(args.fixture,args.rss_worker),
                         sort_keys=True))
        return
    result=benchmark(args.fixture,args.warmups,args.trials)
    if args.check:
        assert len(result["records"])==36
        assert set(result["by_degree"])=={"512","1024"}
        assert all(r["all_four_public_modes_exactly_equal"]
                   for r in result["records"])
    raw=canonical(result)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(raw)
    else:
        print(raw.decode(),end="")


if __name__=="__main__":
    main()
