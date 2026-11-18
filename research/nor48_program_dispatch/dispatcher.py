"""NOR-48 signed-program R-aware exact execution dispatch, arithmetic core.

Backend selection uses the separately authorized native program ONLY.
This reentrant GMP trace verifier passes convolution explicitly, never
reassigns nor46 globals. The signing/session/provider-root validations
are tested separately with the original NOR45/46 authority harness;
this routine alone is not a signed-certificate verifier.
"""
from __future__ import annotations
from typing import Sequence

from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as trace
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp

class DispatchError(trace.TraceError):
    pass

R_FIELDS=("base","u","v")

def authorized_routing_key(program:dict,observed_degree:int,m:int)->str:
    """Closed exact program is the sole routing input, never an unsigned hint."""
    native._validate_program(program,observed_degree,m)
    ops=(program[k] for k in R_FIELDS)
    used="R" in ops
    if program["hidden_beta"]:
        used=used or program["hidden_transform"]=="R"
    return "gmp_full_trace" if used else "nor45_direct"

def _checked_transform(source:tuple[int,...],op:str,witness:tuple[int,...],
                       convolve)->None:
    if len(source)!=len(witness) or witness[0]!=1:
        raise DispatchError("wrong transform length or unit")
    if op=="R":
        product=convolve(source,witness)
        if len(product)!=len(source) or product[0]!=1 or any(
                x!=0 for x in product[1:]):
            raise DispatchError("invalid signed exact inverse witness")
    else:
        trace._check_op(source,op,witness)

def verified_gmp_trace_values(witness:dict,query:Sequence[int],program:dict,
                              observed_degree:int,convolve=gmp.full_gmp)->tuple[int,...]:
    """Verify original full NOR46 exact-integer proof with injected backend."""
    source=tuple(query)
    n=len(source)
    native._validate_program(program,observed_degree,n)
    if not source or source[0]!=1 or any(type(v) is not int for v in source):
        raise DispatchError("source must be exact normalized integer series")
    if not isinstance(witness,dict) or set(witness)!={
        "protocol","query_sha256","coefficient_count","observed_degree",
        "unary_transforms","output_coefficients"}:
        raise DispatchError("noncanonical trace schema")
    if witness["protocol"]!=trace.PROTOCOL or witness["query_sha256"]!=native.query_digest(source):
        raise DispatchError("wrong trace protocol or query")
    if (type(witness["coefficient_count"]) is not int or
        witness["coefficient_count"]!=n or
        type(witness["observed_degree"]) is not int or
        witness["observed_degree"]!=observed_degree):
        raise DispatchError("trace degree/length mismatch")
    needed={program["base"],program["u"],program["v"]}
    if program["hidden_beta"]:needed.add(program["hidden_transform"])
    raw=witness["unary_transforms"]
    if not isinstance(raw,dict) or set(raw)!=needed:
        raise DispatchError("missing or excess trace transform")
    transforms={}
    for op in sorted(needed):
        values=raw[op]
        if not isinstance(values,list) or len(values)!=n:
            raise DispatchError("incorrect transform array length")
        coefficients=tuple(trace._integer(v) for v in values)
        _checked_transform(source,op,coefficients,convolve)
        transforms[op]=coefficients
    output=witness["output_coefficients"]
    if not isinstance(output,list) or len(output)!=n:
        raise DispatchError("incorrect output array length")
    claimed=tuple(trace._integer(v) for v in output)
    base=transforms[program["base"]]
    u=transforms[program["u"]]
    v=transforms[program["v"]]
    beta=program["hidden_beta"]
    hidden=transforms[program["hidden_transform"]] if beta else None
    k,d,c=program["delay"],program["spacing"],program["amplitude"]
    onset=observed_degree+32
    for i,have in enumerate(claimed):
        want=base[i]
        if i==k:want+=c
        if i>=k+d:want+=c*u[i-k-d]
        if i>=k+2*d:want+=c*v[i-k-2*d]
        if beta and i>=onset:want+=beta*hidden[i-onset]
        if want!=have:
            raise DispatchError(f"wrong exact output at coefficient {i}")
    return claimed

def dispatch_verified_arithmetic(program:dict,query:Sequence[int],degree:int,
                                 untrusted_trace:dict|None=None,
                                 unsigned_backend_hint:str|None=None):
    """Compute full future and root. The unsigned backend hint is ignored.

    Program must originate from a previously independently authenticated
    signed statement; authentication is outside this core benchmark.
    """
    route=authorized_routing_key(program,degree,len(query))
    if route=="nor45_direct":
        values=native.execute_authorized_program(program,query,degree)
    else:
        if untrusted_trace is None:
            raise DispatchError("R program requires full exact trace witness")
        values=verified_gmp_trace_values(untrusted_trace,query,program,degree)
    root=native.independently_computed_root(values)
    return {"route":route,"output":values,"merkle_root":root}
