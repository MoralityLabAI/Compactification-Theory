"""NOR-30 TRAIN-only 16-quantum constrained minimax on frozen NOR-26 evidence.

This module contains NO NOR-30 heldout source generation, witness answers,
or test input. Registered before any NOR-30 TEST data are produced.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/"research/nor26_policy_aware_audits/results"
ORIGINAL_SHA="a451897b780b024aac49f543833bc5e3ec32a734489d3f7a9b14a929bd11669a"
ORIGINAL_SOURCE_SHA="e3f5a3df07d062d8681a39a7ddcba9a981b58af6f149f13bd372daf08ac6606f"
ARMS=("uniform","static_prior","posterior","shuffled_prior",
      "hybrid_25","hybrid_50","hybrid_75")
QUANTA=16
SCHEMA="nor30.nor26_only_frozen_grid_minimax.v1"


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def historical_matrix():
    receipt=json.loads((OLD/"nor26_first_original_receipt.json").read_text())
    assert receipt["outcome"]["sha256"]==ORIGINAL_SHA
    assert receipt["fixture"]["sha256"]==ORIGINAL_SOURCE_SHA
    with ZipFile(OLD/"nor26_original_hosted.zip") as z:
        assert sorted(z.namelist())==["nor26_fixture.json","nor26_full.json"]
        assert z.testzip() is None
        rb=z.read("nor26_full.json")
        sb=z.read("nor26_fixture.json")
    assert len(rb)==217954 and sha256(rb).hexdigest()==ORIGINAL_SHA
    assert len(sb)==2838554 and sha256(sb).hexdigest()==ORIGINAL_SOURCE_SHA
    full=json.loads(rb)
    src=json.loads(sb)
    assert full["schema"]=="nor26.policy_aware_blind_attack_crossplay.v1"
    assert len(full["source_records"])==len(src["cases"])==12
    assert full["registered_primary_pass"]
    counts={(a,b):0 for a in ARMS for b in ARMS}
    for s in full["source_records"]:
        assert len(s["crossplay"])==49
        assert all(o["total"]==4096 for o in s["crossplay"])
        for o in s["crossplay"]:
            key=(o["attacker_policy"],o["auditor_policy"])
            assert key in counts
            counts[key]+=o["detected"]
    matrix=[[counts[(att,aud)] for aud in ARMS] for att in ARMS]
    assert all(0<=n<=12*4096 for row in matrix for n in row)
    oldhashes=sorted(x["full_public_input_sha256"] for x in src["cases"])
    assert len(set(oldhashes))==12
    return matrix,oldhashes


def compositions(n,k):
    if k==1:
        yield (n,)
    else:
        for first in range(n+1):
            for tail in compositions(n-first,k-1):
                yield (first,)+tail


def optimize(matrix,constrained=True):
    assert len(matrix)==7 and all(len(r)==7 for r in matrix)
    winning=None
    best_key=None
    examined=0
    for w in compositions(QUANTA,len(ARMS)):
        if constrained and not 2<=w[0]<=12:
            continue
        scores=tuple(sum(v*wi for v,wi in zip(row,w)) for row in matrix)
        key=(min(scores),sum(scores),tuple(-x for x in w))
        examined+=1
        if best_key is None or key>best_key:
            winning=w
            best_key=key
    assert winning is not None and sum(winning)==QUANTA
    assert examined>0
    return {"weights":list(winning),"minimum_weighted_hits":best_key[0],
            "sum_weighted_hits":best_key[1],
            "number_of_grid_candidates":examined,
            "maximum_uniform_quanta":12 if constrained else 16,
            "minimum_uniform_quanta":2 if constrained else 0}


def freeze_training():
    m,old=historical_matrix()
    chosen=optimize(m,True)
    unconstrained=optimize(m,False)
    matrix_hash=sha256(canonical({"auditors":ARMS,"attacker_types":ARMS,
                                  "hits":m,"denominator_per_cell":49152})).hexdigest()
    return {"schema":SCHEMA,"registration":"NOR-30 2026-11-17",
            "historical_fixture_sha256":ORIGINAL_SOURCE_SHA,
            "historical_result_sha256":ORIGINAL_SHA,
            "historical_source_count":12,
            "historical_public_hashes":old,
            "grid_quanta":QUANTA,
            "attacker_types":list(ARMS),"auditors":list(ARMS),
            "historical_denominator_per_cell":49152,
            "historical_integer_hit_matrix":m,
            "historical_matrix_sha256":matrix_hash,
            "tie_break":"max min row hits, max total row hits, lexicographically smallest weights",
            "constrained_solution":chosen,
            "unconstrained_diagnostic_solution":unconstrained,
            "test_information_used_for_training":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    receipt=freeze_training()
    payload=canonical(receipt)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(payload)
    print("NOR30 FROZEN TRAIN-ONLY MIXTURE",receipt["constrained_solution"])
    print("NOR30 HISTORIC 7x7 MATRIX SHA",receipt["historical_matrix_sha256"])
    print("NOR30 FROZEN TRAIN RECEIPT",len(payload),sha256(payload).hexdigest())


if __name__=="__main__":
    main()
