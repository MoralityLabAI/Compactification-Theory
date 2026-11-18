"""NOR-47: experimental exact-signed GMP convolution backends.

The historical NOR46 full-trace verifier used CPython arbitrary-precision int
to pack two signed coefficient vectors and multiply them once. Here we keep
its exact no-carry bound, signed offset corrections, and resource caps while
substituting GMP mpz for large integer multiplication. The two GMP arms
differ ONLY in whether packing uses Python int or mpz.

This is research CPU acceleration, not a proof system and not a replacement
for the independently pinned source/program/session/provider authorities.
"""
from __future__ import annotations
from typing import Sequence

import gmpy2

from research.nor46_exact_integer_trace import trace_verifier as original


class GMPTraceError(original.TraceError):
    pass


def _parameters(a:Sequence[int],b:Sequence[int]):
    n=len(a)
    if not n or len(b)!=n:
        raise GMPTraceError("convolution vectors need identical nonzero length")
    if any(type(x) is not int for x in a) or any(type(x) is not int for x in b):
        raise GMPTraceError("convolution inputs must be exact integers")
    maximum=max(max(abs(x).bit_length() for x in a),
                max(abs(x).bit_length() for x in b))
    if maximum>original.MAX_COEFFICIENT_BITS:
        raise GMPTraceError("unbounded trace coefficient")
    offset=1<<maximum
    width=(4*n*offset*offset).bit_length()+1
    if n*width>original.MAX_PACKED_BITS:
        raise GMPTraceError("packed integer exceeds verifier resource cap")
    assert 2**width>4*n*offset*offset
    return n,offset,width


def _python_pack(a,offset,width):
    acc=0
    for x in reversed(a):
        acc=(acc<<width)+(x+offset)
    return acc


def _gmp_pack(a,offset,width):
    acc=gmpy2.mpz(0)
    for x in reversed(a):
        acc=(acc<<width)+(x+offset)
    return acc


def exact_convolution_prefix_gmp(a:Sequence[int],b:Sequence[int],
                                 packing:str="gmp")->tuple[int,...]:
    """One GMP multiplication, exact signed convolution to degree <m.

    With K=2^maxbits, shifted digits x+K lie in [0,2K) and the product's
    coefficient sum has magnitude <4*m*K^2. The chosen power-of-two base L
    is strictly greater; all low product digits are carry-free. Subtract
    the cross terms to recover the exact signed coefficient per index.
    """
    n,offset,width=_parameters(a,b)
    if packing not in ("gmp","hybrid_python_pack"):
        raise GMPTraceError("unsupported exact packing backend")
    if packing=="gmp":
        packed_a=_gmp_pack(a,offset,width)
        packed_b=_gmp_pack(b,offset,width)
    else:
        packed_a=gmpy2.mpz(_python_pack(a,offset,width))
        packed_b=gmpy2.mpz(_python_pack(b,offset,width))
    product=packed_a*packed_b
    mask=(gmpy2.mpz(1)<<width)-1
    prefix_a=prefix_b=0
    out=[]
    for i in range(n):
        prefix_a+=a[i]
        prefix_b+=b[i]
        current=int(product&mask)
        out.append(current-offset*(prefix_a+prefix_b)-(i+1)*offset*offset)
        product>>=width
    return tuple(out)


def full_gmp(a,b):
    return exact_convolution_prefix_gmp(a,b,"gmp")


def hybrid_gmp(a,b):
    return exact_convolution_prefix_gmp(a,b,"hybrid_python_pack")


def backend_metadata():
    return {
        "gmpy2_version":gmpy2.version(),
        "gmp_version":gmpy2.mp_version(),
        "max_coefficient_bits":original.MAX_COEFFICIENT_BITS,
        "max_packed_bits":original.MAX_PACKED_BITS,
        "algorithm":"exact carry-free Kronecker, original NOR46 no-carry bound",
        "claimed_cryptographic_soundness":False,
        "claimed_proof_succinctness":False
    }
