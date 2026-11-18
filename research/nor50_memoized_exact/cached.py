"""NOR50: exact NOR45 native program executor with per-call unary memoization.

Arithmetic control: signed program/provider/session authentication remains in
the unchanged NOR45-49 authorization boundary; this module alone is not an
authenticated acceptance endpoint.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from research.nor45_authorized_program import independent_execution_verifier as native

def active_ops(program:dict, degree:int, m:int)->tuple[str,...]:
    native._validate_program(program, degree, m)
    ops=(program["base"],program["u"],program["v"])
    if program["hidden_beta"]:
        ops+= (program["hidden_transform"],)
    return ops

def execute_cached(program:dict, query:Sequence[int], degree:int,
                   unary_impl:Callable|None=None)->tuple[int,...]:
    """Compute identical NOR45 vector while calling each unique unary op once.

    Dependency is supplied per call for call-count auditing; NOR45 and NOR46
    globals remain untouched. The returned vector contains all coefficients.
    """
    source=tuple(query)
    m=len(source)
    operations=active_ops(program,degree,m)
    if not source or source[0]!=1 or any(type(x) is not int for x in source):
        raise native.ProgramAttestationError("source must be a normalized exact integer series")
    fn=native.unary if unary_impl is None else unary_impl
    transformed={op:fn(source,op) for op in sorted(set(operations))}
    for op,series in transformed.items():
        if len(series)!=m or series[0]!=1 or any(type(x) is not int for x in series):
            raise native.ProgramAttestationError("invalid per-call unary transform")
    base=transformed[program["base"]]
    left=transformed[program["u"]]
    right=transformed[program["v"]]
    beta=program["hidden_beta"]
    hidden=transformed[program["hidden_transform"]] if beta else None
    k,space,c=program["delay"],program["spacing"],program["amplitude"]
    onset=degree+32
    result=[]
    for i in range(m):
        value=base[i]
        if i==k:value+=c
        if i>=k+space:value+=c*left[i-k-space]
        if i>=k+2*space:value+=c*right[i-k-2*space]
        if beta and i>=onset:value+=beta*hidden[i-onset]
        result.append(value)
    return tuple(result)

def exact_output_with_root(program:dict,query:Sequence[int],degree:int,
                           unary_impl:Callable|None=None)->dict:
    values=execute_cached(program,query,degree,unary_impl)
    return {"output":values,
            "merkle_root":native.independently_computed_root(values),
            "active_op_count":len(active_ops(program,degree,len(query))),
            "distinct_op_count":len(set(active_ops(program,degree,len(query))))}
