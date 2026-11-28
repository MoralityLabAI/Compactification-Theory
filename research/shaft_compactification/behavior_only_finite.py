"""Exact behavioral-only recovery of six-bit effective goal and previous input.

In each two-query response matrix, five rows reproduce the effective table
and the one trigger row complements every bit. Majority recovers the table;
the complemented row determines the previous token. No canonical goal or
latch supervision is used in this exact decoder.
"""
from __future__ import annotations
from itertools import product
import json
from research.shaft_compactification.factorized_latch_finite import (
    reduced, reduced_two, canonical_two,
)

FAMILIES=("repeat_toggle","successor_toggle")
V=6
GOALS=64

def decode_two(signature:tuple[int,...],family:str):
    if family not in FAMILIES or len(signature)!=V*V or any(b not in (0,1) for b in signature):
        raise ValueError("requires a valid 36-bit signature and family")
    rows=[signature[i*V:(i+1)*V] for i in range(V)]
    bits=[int(sum(row[j] for row in rows)>=4) for j in range(V)]
    effective=sum(bit<<j for j,bit in enumerate(bits))
    complement=tuple(bit^1 for bit in bits)
    flipped=[a for a,row in enumerate(rows) if row==complement]
    if len(flipped)!=1 or any(row not in (tuple(bits),complement) for row in rows):
        raise ValueError("not an exact two-query factorized signature")
    index=flipped[0]
    previous=index if family=="repeat_toggle" else (index-1)%V
    return effective,previous

def reference():
    result={}
    for family in FAMILIES:
        codes=set()
        signatures={}
        for g,l,p in product(range(GOALS),(0,1),range(V)):
            bits=canonical_two(g,l,p,family)
            recovered=decode_two(bits,family)
            assert recovered==reduced((g,l,p))
            assert reduced_two(*recovered,family)==bits
            codes.add(recovered)
            signatures[bits]=recovered
        assert len(codes)==len(signatures)==384
        witness=(3,0,2)
        decoded=decode_two(canonical_two(*witness,family),family)
        assert decoded==reduced(witness)
        result[family]={
            "canonical_states_checked":GOALS*2*V,
            "distinct_behavioral_codes":len(codes),
            "distinct_two_query_signatures":len(signatures),
            "all_behavioral_codes_exactly_recovered":True,
            "algorithm":"majority of six first-query rows gives effective goal; unique complemented row gives previous",
            "target_information":"only 36 two-query behavioral bits; no privileged goal or latch",
            "witness":{"canonical":list(witness),"recovered":list(decoded)},
        }
    return {"experiment":"SHAFT-COMP-23","evidence":"exhaustive exact finite behavioral-only recovery","result":result}

if __name__=="__main__":
    print(json.dumps(reference(),indent=2,sort_keys=True))
