"""NOR53: exact native gmpy2.pack/unpack signed Kronecker convolution.

Retains NOR47's integer/carry/resource bounds. The only arithmetic change is
delegating packing and digit unpacking to the GMP C extension. Full exact
witness source/length/normalization checks remain mandatory.
"""
from __future__ import annotations
from collections.abc import Sequence
import gmpy2
from research.nor47_gmp_exact_convolution import gmp_kernel as old
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor51_exact_r_crossover import algebra as nor51
from research.nor52_sparse_inverse import algebra as nor52

class NativePackError(ValueError):
    pass

def pack_orientation():
    """Determine a pinned GMP pack order with exact independent small integers."""
    little=0x21
    packed=int(gmpy2.pack([1,2],4))
    if packed==little:
        return "least_significant_first"
    if packed==0x12:
        return "most_significant_first"
    raise NativePackError("unsupported gmpy2.pack digit order")

def unpack_is_little_endian():
    decoded=[int(v) for v in gmpy2.unpack(gmpy2.mpz(0x321),4)]
    return decoded[:3]==[1,2,3]

def _pack_low_first(digits:list[int],width:int):
    orientation=pack_orientation()
    if not digits:
        return gmpy2.mpz(0)
    if any(type(d) is not int or d<0 or d.bit_length()>width for d in digits):
        raise NativePackError("invalid nonnegative exact packed digit")
    if orientation=="least_significant_first":
        return gmpy2.pack(digits,width)
    return gmpy2.pack(list(reversed(digits)),width)

def exact_convolution_prefix_fast(a:Sequence[int],b:Sequence[int])->tuple[int,...]:
    """Exact signed truncated convolution with original NOR47 safety caps.

    The two shifted arrays are packed by gmpy2.pack, multiplied once, then
    native-unpacked. The low n digits implement the identical NOR47 correction.
    """
    n,offset,width=old._parameters(a,b)
    if not unpack_is_little_endian():
        raise NativePackError("gmpy2.unpack is not low-digit-first")
    left=_pack_low_first([x+offset for x in a],width)
    right=_pack_low_first([x+offset for x in b],width)
    product=left*right
    blocks=gmpy2.unpack(product,width)
    if len(blocks)<n:
        blocks=list(blocks)+[0]*(n-len(blocks))
    prefix_a=prefix_b=0
    out=[]
    for i in range(n):
        prefix_a+=a[i]
        prefix_b+=b[i]
        digit=int(blocks[i])
        out.append(digit-offset*(prefix_a+prefix_b)-(i+1)*offset*offset)
    return tuple(out)

def _checked_full_source_witness(source,witness):
    f=nor51.normalized(source)
    if not isinstance(witness,(tuple,list)) or len(witness)!=len(f):
        raise NativePackError("untrusted full witness has wrong length/type")
    r=tuple(witness)
    if type(r[0]) is not int or r[0]!=1:
        raise NativePackError("untrusted full witness needs builtin integer unit")
    if any(type(v) is not int or abs(v).bit_length()>nor46.MAX_COEFFICIENT_BITS
           for v in r):
        raise NativePackError("untrusted witness noninteger or overcap")
    old._parameters(f,r)
    return f,r

def full_verified_root(source,witness):
    f,r=_checked_full_source_witness(source,witness)
    product=exact_convolution_prefix_fast(f,r)
    if product!=(1,)+(0,)*(len(f)-1):
        raise NativePackError("fast full witness violates exact integer inverse identity")
    return r,native.independently_computed_root(r)

def compressed_verified_root(source,witness,stride:int):
    # Original NOR52 checks source, full witness, all off-support coefficients
    # and original FULL packed-integer cap before shortening the series.
    f,r=nor52.checked_full_witness(source,witness,stride)
    small_f=nor52.compress(f,stride)
    small_r=nor52.compress(r,stride)
    product=exact_convolution_prefix_fast(small_f,small_r)
    if product!=(1,)+(0,)*(len(small_f)-1):
        raise NativePackError("fast compressed witness violates exact inverse identity")
    return r,native.independently_computed_root(r)

def backend_metadata():
    return {
        "gmpy2_version":gmpy2.version(),
        "gmp_version":gmpy2.mp_version(),
        "pack_orientation":pack_orientation(),
        "unpack_little_endian":unpack_is_little_endian(),
        "original_caps":old.backend_metadata(),
        "process_global_override":False,
        "exact_signed_field_reduction":False,
        "complete_merkle_root_built":True,
        "signed_NOR45_grammar_extended":False
    }
