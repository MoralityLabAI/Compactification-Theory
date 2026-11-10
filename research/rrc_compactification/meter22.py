"""RRC-22 independent exact arithmetic meter for RRC-21 graph verification.

For each attempted graph, the audited integer interpreter implements the same
truncated series operations as topology16.execute. It records *actual operand*
bit-lengths and arithmetic op counts, not CPU bit instructions. Tests require
output equality with the pre-existing exact interpreter on all graph/topology
combinations and randomized source data.
"""
from __future__ import annotations
from dataclasses import dataclass

try:
    from . import topology16 as top, graph11 as g
except ImportError:
    import topology16 as top
    import graph11 as g


@dataclass
class Meter:
    multiplications: int = 0
    additions: int = 0
    sign_changes: int = 0
    copies: int = 0
    integer_comparisons: int = 0
    max_operand_bits: int = 0
    max_intermediate_bits: int = 0
    mul_operand_bit_product: int = 0
    add_operand_max_bits: int = 0
    comparison_max_bits: int = 0

    def see(self, *integers):
        for v in integers:
            if type(v) is not int:
                raise TypeError('integer coefficient required')
            self.max_intermediate_bits = max(self.max_intermediate_bits, v.bit_length())

    def mul(self, a, b):
        self.multiplications += 1
        ab, bb = a.bit_length(), b.bit_length()
        self.max_operand_bits = max(self.max_operand_bits, ab, bb)
        self.mul_operand_bit_product += ab * bb
        result = a*b
        self.see(result)
        return result

    def add(self, a, b):
        self.additions += 1
        ab, bb = a.bit_length(), b.bit_length()
        self.max_operand_bits = max(self.max_operand_bits, ab, bb)
        self.add_operand_max_bits += max(ab, bb)
        result = a+b
        self.see(result)
        return result

    def compare(self, a, b):
        self.integer_comparisons += 1
        self.comparison_max_bits = max(self.comparison_max_bits,
                                       a.bit_length(), b.bit_length())
        return a==b

    def summary(self):
        return {k: int(v) for k,v in vars(self).items()}


def unary(series, op, meter):
    if not series or series[0] != 1:
        raise ValueError('unit series required')
    if op=='I':
        meter.copies += len(series)
        return tuple(series)
    if op=='N':
        result=[]
        for i,v in enumerate(series):
            if i%2:
                v=-v
                meter.sign_changes += 1
                meter.see(v)
            result.append(v)
        return tuple(result)
    if op=='S':
        meter.copies += (len(series)+1)//2
        return tuple(series[i//2] if i%2 == 0 else 0 for i in range(len(series)))
    if op=='R':
        answer=[1]
        for n in range(1,len(series)):
            accum=0
            for k in range(1,n+1):
                accum=meter.add(accum,meter.mul(series[k],answer[n-k]))
            answer.append(-accum)
            meter.sign_changes += 1
            meter.see(answer[-1])
        return tuple(answer)
    raise ValueError(op)


def convolution(a,b,meter):
    if len(a)!=len(b):raise ValueError('unequal series lengths')
    out=[]
    for n in range(len(a)):
        accum=0
        for k in range(n+1):
            accum=meter.add(accum,meter.mul(a[k],b[n-k]))
        out.append(accum)
    return tuple(out)


def execute(f, graph, topology, meter=None):
    if topology not in top.TOPOLOGIES or graph not in g.GRAPHS:
        raise ValueError('bad graph or topology')
    if meter is None:meter=Meter()
    if not f or f[0]!=1 or any(type(v) is not int for v in f):
        raise ValueError('input must be integer normalized series')
    meter.see(*f)
    left=unary(f,graph[0],meter)
    right=unary(left if topology=='cascade' else f, graph[1],meter)
    if topology=='additive':
        mixed=(1,)+tuple(meter.add(x,y) for x,y in zip(left[1:],right[1:]))
    elif topology=='hadamard':
        mixed=(1,)+tuple(meter.mul(x,y) for x,y in zip(left[1:],right[1:]))
    else:
        mixed=convolution(left,right,meter)
    result=unary(mixed,graph[2],meter)
    return result


def metered_trace(case, trace):
    meter=Meter()
    detected=None
    for graph in trace:
        checks=[]
        for f,expected in case.demos:
            result=execute(f[:top.PREFIX+1],graph,case.topology,meter)
            expected=expected[:top.PREFIX+1]
            checks.append(all(meter.compare(a,b) for a,b in zip(result,expected)))
        if all(checks):
            detected=graph
            break
    if trace and detected is None and len(trace)<1:
        raise AssertionError('unreachable')
    return detected,meter
