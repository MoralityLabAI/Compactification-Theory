"""Untimed NOR52 phase-attribution ledger for unchanged NOR47 GMP packing.

Reimplements the same no-carry arithmetic with separately measured phases;
never replaces the registered whole-arm timings or mutates global functions.
"""
from __future__ import annotations
import time
import gmpy2
from research.nor47_gmp_exact_convolution import gmp_kernel as original
from research.nor52_sparse_inverse import algebra

PHASES=("parameters_ns","packing_ns","multiplication_ns",
        "extraction_ns","correction_ns")

def instrumented_exact_product(source,witness):
    f,w=algebra.checked_full_witness(source,witness,1)  # Type/cap validation, outside phase clock.
    t0=time.perf_counter_ns()
    n,k,width=original._parameters(f,w)
    t1=time.perf_counter_ns()
    left=original._gmp_pack(f,k,width)
    right=original._gmp_pack(w,k,width)
    t2=time.perf_counter_ns()
    product=left*right
    t3=time.perf_counter_ns()
    mask=(gmpy2.mpz(1)<<width)-1
    digits=[]
    for i in range(n):
        digits.append(int(product&mask))
        product>>=width
    t4=time.perf_counter_ns()
    pref_a=pref_b=0
    out=[]
    for i in range(n):
        pref_a+=f[i]
        pref_b+=w[i]
        out.append(digits[i]-k*(pref_a+pref_b)-(i+1)*k*k)
    t5=time.perf_counter_ns()
    result=tuple(out)
    reference=original.full_gmp(f,w)
    if result!=reference or result!=(1,)+(0,)*(n-1):
        raise ValueError("phase-ledger exact reconstruction differs from NOR47 or identity")
    duration={
        "parameters_ns":t1-t0,"packing_ns":t2-t1,
        "multiplication_ns":t3-t2,"extraction_ns":t4-t3,
        "correction_ns":t5-t4
    }
    assert all(type(v) is int and v>=0 for v in duration.values())
    return {
        **duration,
        "phase_total_ns":sum(duration.values()),
        "digit_width_bits":width,
        "full_packed_bits":n*width,
        "source_length":n,
        "exact_signed_identity":True,
        "reference_kernel_equal":True
    }
