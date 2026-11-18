"""NOR55 canonical exact support-only signed integer witness payload.

Wire format:
   b'N55C1' + minimal ULEB128(coefficient count) +
   repeated (sign_octet [0/1] + minimal ULEB128(magnitude_byte_count)
             + minimal unsigned big-endian magnitude).
A zero uses sign=0 and magnitude-byte-count=0. Negative zero, overlong
varints, leading zero magnitude bytes, overcaps and trailing bytes reject.
"""
from __future__ import annotations
from collections.abc import Sequence
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor52_sparse_inverse import algebra as support

MAGIC=b"N55C1"
MAX_PAYLOAD_BYTES=8_000_000
MAX_MAGNITUDE_BYTES=(nor46.MAX_COEFFICIENT_BITS+7)//8
MAX_VARINT_BYTES=5
MAX_COEFFICIENTS=65536

class CompactWitnessError(ValueError):
    pass

def encode_uvarint(value:int)->bytes:
    if type(value) is not int or not 0<=value<2**32:
        raise CompactWitnessError("unsigned varint value out of range")
    result=bytearray()
    while True:
        b=value&0x7f
        value>>=7
        result.append(b|(0x80 if value else 0))
        if not value:break
    return bytes(result)

def decode_uvarint(raw:bytes,offset:int)->tuple[int,int]:
    if type(raw) is not bytes or type(offset) is not int or offset<0:
        raise CompactWitnessError("invalid varint input or offset")
    value=0
    start=offset
    for i in range(MAX_VARINT_BYTES):
        if offset>=len(raw):
            raise CompactWitnessError("truncated unsigned varint")
        digit=raw[offset]
        offset+=1
        value|=(digit&0x7f)<<(7*i)
        if not digit&0x80:
            if value>=2**32 or raw[start:offset]!=encode_uvarint(value):
                raise CompactWitnessError("overlong/noncanonical unsigned varint")
            return value,offset
    raise CompactWitnessError("unsigned varint exceeds five bytes")

def encode_reduced(values:Sequence[int])->bytes:
    if not isinstance(values,(tuple,list)) or not 1<=len(values)<=MAX_COEFFICIENTS:
        raise CompactWitnessError("invalid reduced witness count")
    if values[0]!=1 or type(values[0]) is not int:
        raise CompactWitnessError("reduced witness is not a builtin integer unit")
    out=bytearray(MAGIC)
    out.extend(encode_uvarint(len(values)))
    for x in values:
        if type(x) is not int or abs(x).bit_length()>nor46.MAX_COEFFICIENT_BITS:
            raise CompactWitnessError("noninteger or oversized signed coefficient")
        magnitude=abs(x)
        width=(magnitude.bit_length()+7)//8
        out.append(1 if x<0 else 0)
        out.extend(encode_uvarint(width))
        if width:
            out.extend(magnitude.to_bytes(width,"big"))
    if len(out)>MAX_PAYLOAD_BYTES:
        raise CompactWitnessError("compact source witness exceeds byte cap")
    return bytes(out)

def decode_reduced(raw:bytes,expected_count:int)->tuple[int,...]:
    if type(raw) is not bytes or not len(MAGIC)<=len(raw)<=MAX_PAYLOAD_BYTES:
        raise CompactWitnessError("invalid compact exact witness bytes")
    if not raw.startswith(MAGIC):
        raise CompactWitnessError("invalid compact source witness magic")
    if type(expected_count) is not int or not 1<=expected_count<=MAX_COEFFICIENTS:
        raise CompactWitnessError("invalid trusted expected reduced count")
    count,offset=decode_uvarint(raw,len(MAGIC))
    if count!=expected_count:
        raise CompactWitnessError("reduced witness coefficient count mismatch")
    result=[]
    for _ in range(count):
        if offset>=len(raw):
            raise CompactWitnessError("truncated witness coefficient sign")
        sign=raw[offset]
        offset+=1
        if sign not in (0,1):
            raise CompactWitnessError("noncanonical witness coefficient sign")
        length,offset=decode_uvarint(raw,offset)
        if length>MAX_MAGNITUDE_BYTES:
            raise CompactWitnessError("witness magnitude exceeds 32768-bit cap")
        if offset+length>len(raw):
            raise CompactWitnessError("truncated signed integer magnitude")
        magnitude_bytes=raw[offset:offset+length]
        offset+=length
        if length and magnitude_bytes[0]==0:
            raise CompactWitnessError("nonminimal big-endian magnitude")
        magnitude=int.from_bytes(magnitude_bytes,"big")
        if sign==1 and magnitude==0:
            raise CompactWitnessError("negative zero not canonical")
        if magnitude.bit_length()>nor46.MAX_COEFFICIENT_BITS:
            raise CompactWitnessError("overcap exact integer magnitude")
        result.append(-magnitude if sign else magnitude)
    if offset!=len(raw):
        raise CompactWitnessError("trailing unsigned witness bytes")
    if not result or result[0]!=1:
        raise CompactWitnessError("witness not unit normalized")
    return tuple(result)

def encode_full_supported(witness:Sequence[int],stride:int)->bytes:
    s=support.stride_checked(stride)
    if not isinstance(witness,(tuple,list)) or len(witness)<1:
        raise CompactWitnessError("invalid full witness container")
    w=tuple(witness)
    if type(w[0]) is not int or w[0]!=1 or any(
            type(v) is not int or abs(v).bit_length()>nor46.MAX_COEFFICIENT_BITS
            for v in w):
        raise CompactWitnessError("invalid complete signed witness")
    if any(w[i]!=0 for i in range(1,len(w)) if i%s):
        raise CompactWitnessError("nonzero off-support witness coefficient")
    return encode_reduced(support.compress(w,s))

def decode_full_supported(raw:bytes,full_length:int,stride:int)->tuple[int,...]:
    s=support.stride_checked(stride)
    if type(full_length) is not int or not 1<=full_length<=MAX_COEFFICIENTS:
        raise CompactWitnessError("invalid trusted full witness length")
    reduced=decode_reduced(raw,support.compressed_length(full_length,s))
    return support.expand(reduced,full_length,s)
