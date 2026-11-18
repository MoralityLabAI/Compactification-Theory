"""NOR58 immutable trusted-geometry indexed-zero/pad NOR45 Merkle template.

The ONLY cache key is (full original length m, independently trusted exact
series support stride s). All original full-index leaf/pad hash bytes and
internal NODE tags are unchanged. Every supported witness coefficient is
hashed freshly per verification, and no result/dynamic hashes are cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor52_sparse_inverse import algebra as support

SCHEMA=b"NOR58/TRUSTED-STATIC-MERKLE-TEMPLATE/v1\x00"
MAX_FULL_LENGTH=4096
SUPPORTED_STRIDES=(1,4,8)

class StaticMerkleError(ValueError):
    pass

@dataclass(frozen=True)
class StaticGeometryTemplate:
    m:int
    stride:int
    size:int
    static_nodes:tuple[bytes|None,...]
    dynamic_leaf_indices:tuple[int,...]
    dynamic_internal_indices_desc:tuple[int,...]
    static_node_count:int
    static_zero_leaf_count:int
    static_padding_leaf_count:int
    template_sha256:str
    canonical_template_bytes:int

    def root(self,values:Sequence[int])->str:
        """Recompute full ORIGINAL NOR45 Merkle root, checking entire support.

        The caller's signed verifier has already validated source and complete
        witness under original full resource caps; this extra local check
        prevents accidental use on mismatched full vectors.
        """
        if not isinstance(values,(tuple,list)) or len(values)!=self.m:
            raise StaticMerkleError("template original full result length mismatch")
        if (type(values[0]) is not int or values[0]!=1 or
            any(type(v) is not int for v in values)):
            raise StaticMerkleError("original full witness requires builtin integer unit")
        if any(values[i]!=0 for i in range(1,self.m) if i%self.stride):
            raise StaticMerkleError("signed witness violates trusted static zero support")
        # Per-call private dynamic hash workspace. No reuse of any computed
        # leaf/internal dependent on signed witness values across sessions.
        dynamic=[None]*(2*self.size)
        framed_m=nor45._word(self.m)
        for i in self.dynamic_leaf_indices:
            dynamic[self.size+i]=sha256(
                nor45.LEAF+framed_m+nor45._word(i)+nor45._integer_bytes(values[i])
            ).digest()
        for k in self.dynamic_internal_indices_desc:
            left=dynamic[2*k]
            right=dynamic[2*k+1]
            if left is None:left=self.static_nodes[2*k]
            if right is None:right=self.static_nodes[2*k+1]
            if left is None or right is None:
                raise StaticMerkleError("inconsistent complete static subtree plan")
            dynamic[k]=sha256(nor45.NODE+left+right).digest()
        root=dynamic[1] if self.size>1 else dynamic[self.size]
        if root is None or len(root)!=32:
            raise StaticMerkleError("missing complete dynamic original Merkle root")
        return root.hex()


def build_template(m:int,stride:int)->StaticGeometryTemplate:
    if (type(m) is not int or not 1<=m<=MAX_FULL_LENGTH or
        type(stride) is not int or stride not in SUPPORTED_STRIDES):
        raise StaticMerkleError("unsupported independently trusted full Merkle geometry")
    size=1<<(m-1).bit_length()
    nodes:[bytes|None]=[None]*(2*size)
    active=[]
    zeros=0
    pads=0
    framed_m=nor45._word(m)
    canonical_zero=nor45._integer_bytes(0)
    for i in range(size):
        leaf=size+i
        if i>=m:
            nodes[leaf]=sha256(nor45.PAD+framed_m+nor45._word(i)).digest()
            pads+=1
        elif i%stride:
            nodes[leaf]=sha256(
                nor45.LEAF+framed_m+nor45._word(i)+canonical_zero
            ).digest()
            zeros+=1
        else:
            active.append(i)
    for k in range(size-1,0,-1):
        left=nodes[2*k]
        right=nodes[2*k+1]
        if left is not None and right is not None:
            nodes[k]=sha256(nor45.NODE+left+right).digest()
    dynamic_desc=tuple(k for k in range(size-1,0,-1) if nodes[k] is None)
    static_count=sum(x is not None for x in nodes[1:])
    # SHA-lock only the original independently trusted geometry and all static
    # indexed canonical hashes. This digest never includes a source value,
    # a dynamic supported coefficient, signed payload or result root.
    h=sha256()
    h.update(SCHEMA+nor45._word(m)+nor45._word(stride)+nor45._word(size))
    record_bytes=len(SCHEMA)+12
    for k in range(1,2*size):
        x=nodes[k]
        if x is not None:
            record=nor45._word(k)+x
            h.update(record)
            record_bytes+=len(record)
    template=StaticGeometryTemplate(
        m=m,stride=stride,size=size,
        static_nodes=tuple(nodes),
        dynamic_leaf_indices=tuple(active),
        dynamic_internal_indices_desc=dynamic_desc,
        static_node_count=static_count,
        static_zero_leaf_count=zeros,
        static_padding_leaf_count=pads,
        template_sha256=h.hexdigest(),
        canonical_template_bytes=record_bytes
    )
    if not 1<=len(active)<=m or len(dynamic_desc)>=size:
        raise StaticMerkleError("malformed static full tree geometry plan")
    return template
