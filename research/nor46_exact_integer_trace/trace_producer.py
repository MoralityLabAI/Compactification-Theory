"""NOR-46 historical-only prover for transparent full exact-integer traces.

A full-trace certificate is NOT a STARK, SNARK, zero-knowledge transcript,
or succinct proof. It contains all transform coefficients and all output
coefficients. The separately signed NOR45 program statement and root are
outside this untrusted trace and must be checked by the independent verifier.
"""
from __future__ import annotations
import json
from hashlib import sha256
from typing import Sequence

from research.nor45_authorized_program import independent_execution_verifier as baseline

PROTOCOL="NOR46/full-exact-integer-trace/v1"


def canonical(value)->bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True).encode("ascii")


def transcript(program:dict,query:Sequence[int],observed_degree:int)->dict:
    """Construct a complete witness using the OLD independent exact executor.

    The new NOR46 trace verifier does not call this executor or derive
    expected transform arrays from its output.
    """
    source=tuple(query)
    baseline._validate_program(program,observed_degree,len(source))
    if not source or source[0]!=1:
        raise ValueError("normalized source required")
    ops={program["base"],program["u"],program["v"]}
    if program["hidden_beta"]:
        ops.add(program["hidden_transform"])
    transformed={op:baseline.unary(source,op) for op in sorted(ops)}
    base=transformed[program["base"]]
    left=transformed[program["u"]]
    right=transformed[program["v"]]
    c=program["amplitude"]
    K=program["delay"]
    spacing=program["spacing"]
    beta=program["hidden_beta"]
    late=transformed.get(program["hidden_transform"]) if beta else None
    suffix_start=observed_degree+32
    output=[]
    for i in range(len(source)):
        x=base[i]
        if i==K:x+=c
        if i>=K+spacing:x+=c*left[i-K-spacing]
        if i>=K+2*spacing:x+=c*right[i-K-2*spacing]
        if beta and i>=suffix_start:x+=beta*late[i-suffix_start]
        output.append(x)
    return {
        "protocol":PROTOCOL,
        "query_sha256":baseline.query_digest(source),
        "coefficient_count":len(source),
        "observed_degree":observed_degree,
        "unary_transforms":{op:[str(v) for v in transformed[op]]
                            for op in sorted(transformed)},
        "output_coefficients":[str(v) for v in output]
    }


def certificate_sha256(trace:dict)->str:
    return sha256(canonical(trace)).hexdigest()


def trace_json_bytes(trace:dict)->int:
    return len(canonical(trace))
