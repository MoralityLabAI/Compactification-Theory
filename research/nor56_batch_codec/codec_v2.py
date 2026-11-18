"""NOR56 exact signed compact-v2 table decoder.

Canonical wire:
 b'N56C2' + minimal ULEB128(n) + n big-endian unsigned 16-bit entries
 + tightly concatenated big-endian minimal unsigned magnitude bytes.

Table's high bit is sign, low 15 bits are magnitude byte count; zero is
sign=0/count=0. 32768-bit original NOR46 cap implies <=4096 bytes.
The whole n-entry length table is decoded in one struct.unpack_from call.
"""
from __future__ import annotations
import struct
from collections.abc import Sequence
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor52_sparse_inverse import algebra as support
from research.nor55_compact_witness import codec as v1

MAGIC=b"N56C2"
MAX_PAYLOAD_BYTES=v1.MAX_PAYLOAD_BYTES
MAX_COEFFICIENTS=v1.MAX_COEFFICIENTS
MAX_MAGNITUDE_BYTES=(nor46.MAX_COEFFICIENT_BITS+7)//8
SIGN_MASK=0x8000
LENGTH_MASK=0x7fff

class BatchCodecError(ValueError):
    pass

def encode_reduced(values:Sequence[int])->bytes:
    if not isinstance(values,(tuple,list)) or not 1<=len(values)<=MAX_COEFFICIENTS:
        raise BatchCodecError("invalid reduced signed witness count")
    if type(values[0]) is not int or values[0]!=1:
        raise BatchCodecError("reduced witness requires exact builtin-integer unit")
    metadata=[]
    magnitude_parts=[]
    width_total=0
    for x in values:
        if type(x) is not int or abs(x).bit_length()>nor46.MAX_COEFFICIENT_BITS:
            raise BatchCodecError("noninteger/overcap exact signed coefficient")
        magnitude=abs(x)
        nbytes=(magnitude.bit_length()+7)//8
        if nbytes>MAX_MAGNITUDE_BYTES or nbytes>LENGTH_MASK:
            raise BatchCodecError("magnitude byte count exceeds table cap")
        sign=SIGN_MASK if x<0 else 0
        if nbytes==0 and sign:
            raise BatchCodecError("negative zero forbidden")
        metadata.append(sign|nbytes)
        magnitude_parts.append(magnitude.to_bytes(nbytes,"big"))
        width_total+=nbytes
    fixed=MAGIC+v1.encode_uvarint(len(values))
    size=len(fixed)+2*len(values)+width_total
    if size>MAX_PAYLOAD_BYTES:
        raise BatchCodecError("overlong v2 exact signed payload")
    return fixed+struct.pack(f">{len(values)}H",*metadata)+b"".join(magnitude_parts)

def decode_reduced(raw:bytes,expected_count:int)->tuple[int,...]:
    if type(raw) is not bytes or not len(MAGIC)<=len(raw)<=MAX_PAYLOAD_BYTES:
        raise BatchCodecError("invalid batch signed payload input")
    if not raw.startswith(MAGIC):
        raise BatchCodecError("wrong signed compact-v2 magic")
    if type(expected_count) is not int or not 1<=expected_count<=MAX_COEFFICIENTS:
        raise BatchCodecError("invalid trusted source reduced coefficient count")
    count,offset=v1.decode_uvarint(raw,len(MAGIC))
    if count!=expected_count:
        raise BatchCodecError("reduced source witness count mismatches trusted input")
    end_table=offset+2*count
    if end_table>len(raw):
        raise BatchCodecError("truncated signed 16-bit length table")
    # One native/C-backed table read replaces Python-per-entry varint reads.
    packed=struct.unpack_from(f">{count}H",raw,offset)
    lengths=tuple(entry & LENGTH_MASK for entry in packed)
    if any(n>MAX_MAGNITUDE_BYTES for n in lengths):
        raise BatchCodecError("signed magnitude exceeds original 32768-bit cap")
    if end_table+sum(lengths)!=len(raw):
        raise BatchCodecError("signed table length disagrees with exact payload bytes")
    out=[]
    offset=end_table
    for entry,nbytes in zip(packed,lengths):
        sign=bool(entry&SIGN_MASK)
        if not nbytes:
            if sign:
                raise BatchCodecError("negative zero is noncanonical")
            out.append(0)
            continue
        part=raw[offset:offset+nbytes]
        offset+=nbytes
        if len(part)!=nbytes or part[0]==0:
            raise BatchCodecError("truncated/nonminimal big-endian signed magnitude")
        magnitude=int.from_bytes(part,"big")
        if magnitude.bit_length()>nor46.MAX_COEFFICIENT_BITS:
            raise BatchCodecError("signed coefficient outside original full cap")
        out.append(-magnitude if sign else magnitude)
    if offset!=len(raw) or not out or out[0]!=1:
        raise BatchCodecError("trailing bytes or missing exact unit")
    return tuple(out)

def encode_full_supported(witness:Sequence[int],stride:int)->bytes:
    s=support.stride_checked(stride)
    if not isinstance(witness,(tuple,list)) or not witness:
        raise BatchCodecError("invalid full signed witness container")
    r=tuple(witness)
    if type(r[0]) is not int or r[0]!=1 or any(
            type(x) is not int or abs(x).bit_length()>nor46.MAX_COEFFICIENT_BITS
            for x in r):
        raise BatchCodecError("wrong full signed exact integer unit/caps")
    if any(r[i]!=0 for i in range(1,len(r)) if i%s):
        raise BatchCodecError("off-support full witness coefficient nonzero")
    return encode_reduced(support.compress(r,s))

def decode_full_supported(raw:bytes,full_length:int,stride:int)->tuple[int,...]:
    s=support.stride_checked(stride)
    if type(full_length) is not int or not 1<=full_length<=MAX_COEFFICIENTS:
        raise BatchCodecError("invalid trusted full source length")
    count=support.compressed_length(full_length,s)
    return support.expand(decode_reduced(raw,count),full_length,s)
