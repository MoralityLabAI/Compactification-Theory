"""Exact signed reciprocal kernel and Newton producer, NOR51.

This module intentionally measures a single formal-series inverse (including
length 3073), outside NOR45's currently authorized-program grammar. It does
not authenticate an executable program/session or establish semantic truth.
"""
from __future__ import annotations
from collections.abc import Callable,Sequence
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp

class ExactReciprocalError(ValueError):
    pass

def normalized(source:Sequence[int])->tuple[int,...]:
    if not isinstance(source,(tuple,list)):
        raise ExactReciprocalError("exact source must be a list/tuple")
    x=tuple(source)
    if not x or type(x[0]) is not int or x[0]!=1:
        raise ExactReciprocalError("normalized exact source must begin with integer 1")
    if any(type(v) is not int or abs(v).bit_length()>nor46.MAX_COEFFICIENT_BITS
           for v in x):
        raise ExactReciprocalError("source contains noninteger or overcap coefficient")
    return x

def checked_inverse_witness(source:Sequence[int],witness:Sequence[int],
                            convolve:Callable|None=None)->tuple[int,...]:
    """Require all coefficients of F*R to equal (1,0,...,0) as signed ints."""
    f=normalized(source)
    if not isinstance(witness,(tuple,list)) or len(witness)!=len(f):
        raise ExactReciprocalError("wrong inverse witness length/type")
    r=tuple(witness)
    if type(r[0]) is not int or r[0]!=1 or any(
        type(v) is not int or abs(v).bit_length()>nor46.MAX_COEFFICIENT_BITS
        for v in r):
        raise ExactReciprocalError("invalid exact signed witness coefficient")
    kernel=gmp.full_gmp if convolve is None else convolve
    p=kernel(f,r)
    if len(p)!=len(f) or p[0]!=1 or any(v!=0 for v in p[1:]):
        raise ExactReciprocalError("claimed inverse fails full exact no-carry identity")
    return r

def native_single_inverse(source:Sequence[int])->tuple[int,...]:
    f=normalized(source)
    x=native.unary(f,"R")
    if len(x)!=len(f) or x[0]!=1:
        raise ExactReciprocalError("native output was not normalized")
    return x

def newton_inverse(source:Sequence[int],convolve:Callable|None=None)->tuple[int,...]:
    """Formal exact Newton doubling G <- G*(2-F*G) mod x^next.

    Every stage uses two independently checked, signed Kronecker GMP
    convolution calls by default; intermediate integer bounds stay in force.
    """
    f=normalized(source)
    kernel=gmp.full_gmp if convolve is None else convolve
    current=(1,)
    while len(current)<len(f):
        upto=min(len(current)*2,len(f))
        padded=current+(0,)*(upto-len(current))
        partial=kernel(f[:upto],padded)
        assert len(partial)==upto and partial[0]==1
        correction=(2-partial[0],)+tuple(-v for v in partial[1:])
        expanded=kernel(padded,correction)
        if (len(expanded)!=upto or expanded[0]!=1 or
            expanded[:len(current)]!=current):
            raise ExactReciprocalError("Newton stage lost exact lower-degree coefficients")
        current=tuple(expanded)
    return current

def root(values:Sequence[int])->str:
    return native.independently_computed_root(values)

def direct_with_root(source:Sequence[int]):
    inv=native_single_inverse(source)
    return inv,root(inv)

def gmp_witness_with_root(source:Sequence[int],witness:Sequence[int]):
    inv=checked_inverse_witness(source,witness)
    return inv,root(inv)

def newton_verified_with_root(source:Sequence[int]):
    inv=newton_inverse(source)
    inv=checked_inverse_witness(source,inv)
    return inv,root(inv)

def backend_metadata():
    return {
        "gmp":gmp.backend_metadata(),
        "native_program_grammar_extended":False,
        "signed_program_authentication_in_kernel":False,
        "full_merkle_root_in_all_timed_arms":True,
        "signed_integer_field_reduction_used":False,
        "inverse_resource_coefficient_bits":nor46.MAX_COEFFICIENT_BITS,
        "inverse_resource_packed_bits":nor46.MAX_PACKED_BITS
    }
