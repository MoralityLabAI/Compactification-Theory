"""NOR59 source-certified finite support for ORIGINAL NOR45 authorized programs.

There is deliberately no arbitrary root callback and no user-provided cache
stride. The verifier has already authenticated the original NOR45 program,
NOR44 verifier precommit and provider signature before invoking this helper.
The only cached values depend on original full length and source-derived
support stride. All authenticated output-dependent nodes are freshly hashed.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Sequence

from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor58_static_merkle import template

ALLOWED_SUPPORT_STRIDES=(1,4,8)
ROOT_MODES=("warm","cold")

class AuthorizedSupportError(nor45.ProgramAttestationError):
    pass

def _checked_exact_full_source(query:Sequence[int],degree:int)->tuple[int,...]:
    if not isinstance(query,(tuple,list)) or type(degree) is not int:
        raise AuthorizedSupportError("full source/observed degree is not independently typed")
    f=tuple(query)
    if degree not in (512,1024) or len(f)!=3*degree//2+1:
        raise AuthorizedSupportError("outside authorized NOR45 source geometry")
    if not f or type(f[0]) is not int or f[0]!=1:
        raise AuthorizedSupportError("normalized NOR45 exact integer unit required")
    if any(type(x) is not int for x in f):
        raise AuthorizedSupportError("non-builtin signed exact input integer")
    return f

def trusted_public_support_stride(query:Sequence[int],degree:int)->int:
    """Read ONLY independently trusted public source coefficients and length."""
    f=_checked_exact_full_source(query,degree)
    for s in (8,4):
        if all(f[i]==0 for i in range(1,len(f)) if i%s):
            return s
    return 1

def certify_actual_NOR45_program_support(*,program:dict,
                                          query:Sequence[int],
                                          observed_degree:int,
                                          output:Sequence[int])->int:
    """Finite source/grammar/output closure certificate, executable Python.

    Applied AFTER original NOR45 signed authority/session/query checks.
    This checks exact real original shifts and all off-support coefficients,
    and never infers any stride from program/provider/model metadata.
    """
    f=_checked_exact_full_source(query,observed_degree)
    m=len(f)
    nor45._validate_program(program,observed_degree,m)
    s=trusted_public_support_stride(f,observed_degree)
    if any(program[k] not in nor45.OPS for k in ("base","u","v")):
        raise AuthorizedSupportError("unregistered signed NOR45 unary op")
    if program["hidden_beta"] and program["hidden_transform"] not in nor45.OPS:
        raise AuthorizedSupportError("unregistered NOR45 hidden unary suffix")
    k=program["delay"]
    space=program["spacing"]
    shifts=(k,k+space,k+2*space)
    if program["hidden_beta"]:
        shifts+= (observed_degree+32,)
    if any(shift%s for shift in shifts):
        raise AuthorizedSupportError("authenticated program shift violates source support")
    if not isinstance(output,(tuple,list)) or len(output)!=m:
        raise AuthorizedSupportError("full authorized program output length changed")
    v=tuple(output)
    if any(type(x) is not int for x in v) or type(v[0]) is not int or v[0]!=1:
        raise AuthorizedSupportError("nonunit or noninteger authorized output")
    if any(v[i]!=0 for i in range(1,m) if i%s):
        raise AuthorizedSupportError("recomputed authorized output violates support closure")
    return s

@lru_cache(maxsize=6)
def _geometry_only_warm_template(m:int,s:int)->template.StaticGeometryTemplate:
    return template.build_template(m,s)

def warm_geometry_only_template(m:int,s:int)->template.StaticGeometryTemplate:
    """Prepare only trusted geometry. The root path independently certifies s."""
    if type(m) is not int or m not in (769,1537):
        raise AuthorizedSupportError("warm NOR45 source length outside signed grammar")
    if type(s) is not int or s not in ALLOWED_SUPPORT_STRIDES:
        raise AuthorizedSupportError("unsupported canonical public source stride")
    return _geometry_only_warm_template(m,s)

def authorized_root(*,program:dict,query:Sequence[int],
                    observed_degree:int,output:Sequence[int],mode:str)->str:
    if mode not in ROOT_MODES:
        raise AuthorizedSupportError("unknown exact NOR45 static root policy")
    s=certify_actual_NOR45_program_support(
        program=program,query=query,observed_degree=observed_degree,
        output=output)
    m=len(query)
    geometry=(warm_geometry_only_template(m,s) if mode=="warm" else
              template.build_template(m,s))
    return geometry.root(output)

def source_free_closure_explanation()->dict:
    """Public, auditable closure constraints, not a Lean theorem."""
    return {
        "NOR45_ops":list(nor45.OPS),
        "support_strides":list(ALLOWED_SUPPORT_STRIDES),
        "delays":list(nor45.DELAYS),
        "spacings":list(nor45.SPACINGS),
        "observed_degrees":[512,1024],
        "all_allowed_delay_and_spacing_sums_divisible_by_8":
            all(k%8==0 and (k+t)%8==0 and (k+2*t)%8==0
                for k in nor45.DELAYS for t in nor45.SPACINGS),
        "all_hidden_suffix_onsets_divisible_by_8":
            all((degree+32)%8==0 for degree in (512,1024)),
        "S_maps_supported_stride_s_to_multiple_2s":True,
        "R_inverse_uniqueness_requires_normalized_constant_one":True,
        "formal_Lean_kernel_proof_available":False
    }
