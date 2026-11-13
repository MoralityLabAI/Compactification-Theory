"""DMB-01B: exact deletion monitor for RRC25's source-locked trained ranker.

Each generated case maps to its two top-ranked candidate class IDs (8 bits).
A B-bit deterministic streaming monitor observes iid deletion traces of
that word. The best *terminal decoder for the finite empirical prior* is
computed exactly using Fraction. Original RRC25 model weights are fixed.

The 8-bit diagnostic is a ranking output, NOT hidden chain of thought.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import bench25 as source_gate
from research.rrc_compactification import compound25 as task
from research.rrc_compactification import model25 as model
from research.deletion_memory_barrier_v1.reference import (
    deletion_law, finite_prior_ceiling, optimal_terminal_success_weighted,
    streaming_laws_for_words,
)

STATUS="FROZEN_TRAINED_RRC25_RANKING_DELETION_DIAGNOSTIC"
SCHEMA="dmb01b.rrc25_ranker_deletion.v1"
DELETIONS=(Fraction(1,2),Fraction(4,5))
COUNTS=(0,1,2,3)
BITS=(2,4,6)


def ranked_word(case, checkpoint:dict)->dict:
    heads=task.possible_headers(case)
    if not heads:
        raise RuntimeError("public header scan has no candidates")
    selected=heads[0]
    scores=model.logits(case,selected,checkpoint)
    if len(scores)!=16 or not all(math.isfinite(x) for x in scores):
        raise RuntimeError("native RRC25 head invalid")
    ranked=sorted(range(16),key=lambda j:(-scores[j],j))
    classes=ranked[:2]
    word="".join(f"{v:04b}" for v in classes)
    if len(word)!=8:
        raise AssertionError("two source classes must encode exactly eight bits")
    return {"case_id":case.identifier,"family":case.family,
            "public_candidate_headers":len(heads),
            "ranked_class_ids":classes,"ranked_trace_word":word}


def empirical_prior(records:list[dict])->dict[str,Fraction]:
    if not records:
        raise ValueError("empty source roster")
    counts=Counter(x["ranked_trace_word"] for x in records)
    return {w:Fraction(count,len(records)) for w,count in sorted(counts.items())}


def full_one_trace_success(prior:dict[str,Fraction],q:Fraction)->Fraction:
    laws={word:deletion_law(word,q) for word in prior}
    observations=set().union(*(set(d) for d in laws.values()))
    return sum((max(prior[x]*laws[x].get(t,Fraction())
                    for x in prior) for t in observations),Fraction())


def one_prior_report(prior:dict[str,Fraction],q:Fraction,
                     counts=COUNTS,bits=BITS)->dict:
    if not counts or not bits or any(m<0 for m in counts) or any(b<1 for b in bits):
        raise ValueError("invalid monitoring grid")
    rows=[]
    base=max(prior.values())
    for m in counts:
        for B in bits:
            laws=streaming_laws_for_words(prior,q,m,B)
            value=optimal_terminal_success_weighted(laws,prior)
            limit=finite_prior_ceiling(prior,B)
            if not base<=value<=limit:
                raise AssertionError("monitor prior ceiling violated")
            rows.append({"m":m,"B":B,"success":str(value),
                         "success_float":float(value),
                         "prior_topK_ceiling":str(limit),
                         "prior_topK_ceiling_float":float(limit),
                         "terminal_state_alphabet":1<<B,
                         "decoder_static_codebook_source_atoms":len(prior)})
    unrestricted=full_one_trace_success(prior,q)
    if any(row["success_float"] > float(unrestricted)+1e-12
           for row in rows if row["m"]==1):
        raise AssertionError("B-bit state beats full one-trace observation")
    return {"deletion_q":str(q),"full_one_trace_MAP_success":str(unrestricted),
            "one_trace_MAP_float":float(unrestricted),"monitor_grid":rows}


def run(test_per:int=6,deletions=DELETIONS,counts=COUNTS,bits=BITS)->dict:
    if type(test_per) is not int or not 1<=test_per<=16:
        raise ValueError("test_per must be 1..16")
    if len(set(deletions))!=len(deletions):
        raise ValueError("duplicate deletion rate")
    trained,digests=source_gate.checkpoints()
    checkpoint=trained["recurrent"]
    if checkpoint["kind"]!="recurrent" or checkpoint["parameters"]!=1000:
        raise AssertionError("source recurrent checkpoint contract changed")
    roster=task.roster(include_test=True,train_per=1,dev_per=1,test_per=test_per)
    cases=[x for fam in task.TEST_FAMILIES for x in roster[fam]]
    if len(cases)!=len(task.TEST_FAMILIES)*test_per:
        raise AssertionError("unexpected original source cohort size")
    records=[ranked_word(case,checkpoint) for case in cases]
    prior=empirical_prior(records)
    if sum(prior.values())!=1:
        raise AssertionError("empirical prior normalization failed")
    entropy=-sum(float(v)*math.log2(float(v)) for v in prior.values())
    metrics=[one_prior_report(prior,Fraction(q),counts,bits) for q in deletions]
    return {
        "status":STATUS,
        "schema":SCHEMA,
        "original_model":"RRC25 trained recurrent GRU template ranker",
        "checkpoint_sha256":digests["recurrent"],
        "source_model_hashes":checkpoint["source_sha256"],
        "model_parameters":checkpoint["parameters"],
        "source_roster_sha256":task.fingerprint(cases),
        "source_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_per_family":test_per,
        "source_cases":len(records),
        "observed_symbol_count":2,
        "native_ranked_candidate_bits":8,
        "source_prior_support":len(prior),
        "source_prior_shannon_entropy_bits":entropy,
        "ranked_source_word_probabilities":{w:str(p) for w,p in prior.items()},
        "source_case_records":records,
        "native_model_rankings_not_hidden_cot":True,
        "ranker_trained_on_dmb01_data":False,
        "posterior_table_from_same_finite_prior":True,
        "dmb01_monitor_codebook_description_bits_nominal":
            len(prior)*8,
        "counts":list(counts),
        "memory_bit_budgets":list(bits),
        "results":metrics,
        "deletion_mask_positions_revealed":False,
        "trained_weights_updated":False,
        "external_theorem_122_140_reproduced":False,
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-per",type=int,default=6)
    p.add_argument("--deletions",type=Fraction,nargs="+",
                   default=list(DELETIONS))
    p.add_argument("--counts",type=int,nargs="+",default=list(COUNTS))
    p.add_argument("--bits",type=int,nargs="+",default=list(BITS))
    p.add_argument("--out",type=Path)
    args=p.parse_args(argv)
    result=run(args.test_per,tuple(args.deletions),
               tuple(args.counts),tuple(args.bits))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,sort_keys=True,indent=2,
                                        allow_nan=False)+"\n")
    compact=[]
    for q in result["results"]:
        compact.append({
            "q":q["deletion_q"],
            "full_one_trace_MAP":q["one_trace_MAP_float"],
            "monitor":[{"m":x["m"],"B":x["B"],
                        "success":round(x["success_float"],6),
                        "upper":round(x["prior_topK_ceiling_float"],6)}
                       for x in q["monitor_grid"] if
                       x["m"] in (0,1,3) and x["B"] in (2,4,6)]
        })
    print(json.dumps({
        "status":result["status"],
        "checkpoint_sha256":result["checkpoint_sha256"],
        "cases":result["source_cases"],
        "distinct_source_words":result["source_prior_support"],
        "prior_entropy_bits":result["source_prior_shannon_entropy_bits"],
        "results":compact},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
