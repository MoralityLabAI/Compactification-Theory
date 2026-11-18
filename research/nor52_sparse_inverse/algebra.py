"""NOR52 exact integer sparse-support inverse and checked GMP witness.

Every path returns the full signed vector and rebuilds the independent full
NOR45-format Merkle root. A public stride is validated, never trusted.
"""
from __future__ import annotations
from collections.abc import Sequence
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp
from research.nor51_exact_r_crossover import algebra as prior

class SparseInverseError(ValueError):
    pass

REGISTERED_STRIDES=(1,4,8)

def stride_checked(stride:int)->int:
    if type(stride) is not int or stride not in REGISTERED_STRIDES:
        raise SparseInverseError("unsupported public exact source stride")
    return stride

def checked_source(source:Sequence[int],stride:int)->tuple[int,...]:
    s=stride_checked(stride)
    f=prior.normalized(source)
    if any(f[i]!=0 for i in range(1,len(f)) if i%s):
        raise SparseInverseError("source violates claimed public support")
    return f

def checked_full_witness(source:Sequence[int],witness:Sequence[int],
                         stride:int):
    s=stride_checked(stride)
    f=checked_source(source,s)
    if not isinstance(witness,(tuple,list)) or len(witness)!=len(f):
        raise SparseInverseError("wrong full-length witness")
    w=tuple(witness)
    if type(w[0]) is not int or w[0]!=1:
        raise SparseInverseError("witness must begin with builtin exact unit")
    if any(type(v) is not int or abs(v).bit_length()>nor46.MAX_COEFFICIENT_BITS
           for v in w):
        raise SparseInverseError("non-integer or overcap signed witness")
    if any(w[i]!=0 for i in range(1,len(w)) if i%s):
        raise SparseInverseError("witness violates full public support")
    # Enforce the SAME original full-size signed-integer and packed-bit cap,
    # even though the actual convolution below uses fewer coefficients.
    gmp._parameters(f,w)
    return f,w

def compressed_length(m:int,s:int)->int:
    return (m-1)//s+1

def compress(f:Sequence[int],s:int)->tuple[int,...]:
    return tuple(f[i] for i in range(0,len(f),s))

def expand(coeffs:Sequence[int],m:int,s:int)->tuple[int,...]:
    if len(coeffs)!=compressed_length(m,s):
        raise SparseInverseError("wrong compressed inverse length")
    out=[0]*m
    for j,value in enumerate(coeffs):
        out[j*s]=value
    return tuple(out)

def native_full_root(source:Sequence[int]):
    return prior.direct_with_root(source)

def native_compressed_root(source:Sequence[int],stride:int):
    f=checked_source(source,stride)
    small=compress(f,stride)
    inverse_small=native.unary(small,"R")
    full=expand(inverse_small,len(f),stride)
    # Original full signed coefficient and total packed-bit resource caps.
    gmp._parameters(f,full)
    return full,native.independently_computed_root(full)

def gmp_full_root(source:Sequence[int],witness:Sequence[int]):
    return prior.gmp_witness_with_root(source,witness)

def gmp_compressed_root(source:Sequence[int],witness:Sequence[int],stride:int):
    f,w=checked_full_witness(source,witness,stride)
    small_f=compress(f,stride)
    small_w=compress(w,stride)
    product=gmp.full_gmp(small_f,small_w)
    if len(product)!=len(small_f) or product[0]!=1 or any(
            value!=0 for value in product[1:]):
        raise SparseInverseError("full witness fails exact compressed inverse")
    # Return THE ORIGINAL full-length, untrusted but independently validated
    # witness, rather than rebuilding a different output from it.
    return w,native.independently_computed_root(w)

def original_and_compressed_resource_sizes(source,witness,stride):
    f,w=checked_full_witness(source,witness,stride)
    n,k,full_width=gmp._parameters(f,w)
    small_f=compress(f,stride)
    small_w=compress(w,stride)
    small_n,small_k,small_width=gmp._parameters(small_f,small_w)
    return {
        "original_coefficient_count":n,
        "compressed_coefficient_count":small_n,
        "original_selected_width_bits":full_width,
        "compressed_selected_width_bits":small_width,
        "original_selected_packed_bits":n*full_width,
        "compressed_selected_packed_bits":small_n*small_width,
        "original_source_max_bits":max(abs(v).bit_length() for v in f),
        "original_witness_max_bits":max(abs(v).bit_length() for v in w)
    }
