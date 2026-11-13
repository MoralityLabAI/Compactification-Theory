"""FPB-01: exact finite Fourier/bit-flip oracle for a hidden parity-lift mode.

Uses only stdlib and Fraction. The finite Rubinstein-type example has
17 original input bits and a 34-bit XOR-lifted input; Fourier coefficients
are transported analytically instead of enumerating all 2**34 inputs.

The superquadratic and unbounded-Fourier claims of OpenAI Math #132/#192
are conditional source results. This finite control does not reproduce
their asymptotic witnesses.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from fractions import Fraction
import hashlib
import json
from pathlib import Path

BASE_N = 16
HIDDEN_N = 17
SENTINEL = 1 << BASE_N
BASE_MASK = (1 << BASE_N) - 1
SURFACE_N = 5
RUBINSTEIN_PATTERNS = (0b0011, 0b1100)


def rubinstein16(x: int) -> int:
    """Four 4-bit blocks, each accepted iff exactly one aligned pair is 1."""
    if type(x) is not int or not 0 <= x < (1 << BASE_N):
        raise ValueError("16-bit source must be in 0..65535")
    return int(any(((x >> (4 * block)) & 15) in RUBINSTEIN_PATTERNS
                   for block in range(4)))


def hidden17(x: int) -> int:
    """Sentinel OR Rubinstein16, with last bit controlling direct acceptance."""
    if type(x) is not int or not 0 <= x < (1 << HIDDEN_N):
        raise ValueError("hidden17 input out of range")
    return int(bool(x & SENTINEL) or bool(rubinstein16(x & BASE_MASK)))


def sign(value: int) -> int:
    if value not in (0, 1):
        raise ValueError("sign expects a binary label")
    return 1-2*value


def hidden_sign(x: int) -> int:
    return sign(hidden17(x))


def lifted_sign(x: int, y: int) -> int:
    if any(type(v) is not int or not 0 <= v < (1 << HIDDEN_N)
           for v in (x, y)):
        raise ValueError("two independent 17-bit input registers required")
    return hidden_sign(x ^ y)


@lru_cache(maxsize=1)
def base_truth() -> bytes:
    return bytes(rubinstein16(x) for x in range(1 << BASE_N))


def exact_base_rejection_count() -> int:
    return base_truth().count(0)


def base_sensitivity() -> dict:
    """Exhaustive global sensitivity over all 2**16 truth-table inputs."""
    table=base_truth()
    s0=s1=0
    witnesses={}
    for x,value in enumerate(table):
        count=sum(table[x^(1<<j)]!=value for j in range(BASE_N))
        if value==0 and count>s0:
            s0=count;witnesses["zero"]=[x,count]
        elif value==1 and count>s1:
            s1=count;witnesses["one"]=[x,count]
    return {"sensitivity_zero_max":s0,
            "sensitivity_one_max":s1,
            "sensitivity_global":max(s0,s1),
            "example_witnesses":witnesses,
            "states_exhausted":1<<BASE_N}


def sensitive_zero_blocks() -> list[int]:
    """Eight independent aligned pairs plus one sentinel bit."""
    blocks=[0b11 << (4*b + 2*j) for b in range(4) for j in range(2)]
    blocks.append(SENTINEL)
    union=0
    for block in blocks:
        if union & block or not block or hidden17(block)!=1:
            raise AssertionError("proposed zero-sensitivity block fails")
        union|=block
    if hidden17(0)!=0:
        raise AssertionError("hidden source at zero must reject")
    return blocks


def hidden_sensitivity() -> dict:
    """Exact 17-bit sentinel sensitivity derived from 16-bit exhaustive scan."""
    base=base_sensitivity()
    # For sentinel=0 and h16=0, flipping sentinel adds one sensitive
    # coordinate; for sentinel=1, only that bit can affect the output.
    s=max(base["sensitivity_zero_max"]+1,
          base["sensitivity_one_max"],1)
    blocks=sensitive_zero_blocks()
    return {"global_hidden17":s,
            "global_parity_lift":2*s,
            "zero_disjoint_blocks_lower":len(blocks),
            "zero_blocks":[f"{b:017b}" for b in blocks],
            "finite_ratio_lower":str(Fraction(len(blocks),(2*s)**2)),
            "base":base}


def character(mask: int, state: int) -> int:
    return -1 if ((mask & state).bit_count() & 1) else 1


def base_fourier_numerator(mask: int) -> int:
    """Integer numerator of sign(h16) Fourier coefficient on 2**16."""
    if type(mask) is not int or not 0 <= mask < (1<<BASE_N):
        raise ValueError("base Fourier mask out of range")
    table=base_truth()
    return sum(sign(table[x])*character(mask,x) for x in range(1<<BASE_N))


def hidden_fourier(mask: int) -> Fraction:
    """Coefficient of H=(-1)^hidden17 on the 17-dimensional sign cube."""
    if type(mask) is not int or not 0 <= mask < (1<<HIDDEN_N):
        raise ValueError("hidden Fourier mask out of range")
    low=mask & BASE_MASK
    has_sentinel=bool(mask & SENTINEL)
    if low==0:
        p0=Fraction(exact_base_rejection_count(),1<<BASE_N)
        return p0 if has_sentinel else p0-1
    return Fraction(base_fourier_numerator(low),1<<(BASE_N+1))


def lifted_fourier(left: int, right: int) -> Fraction:
    """Pair-parity Fourier transport: nonzero iff masks match."""
    if any(type(x) is not int or not 0<=x<(1<<HIDDEN_N)
           for x in (left,right)):
        raise ValueError("lifted Fourier masks out of range")
    return hidden_fourier(left) if left==right else Fraction(0)


def majority5_sign(x: int) -> int:
    """Odd strict majority of five independent input signs."""
    if type(x) is not int or not 0<=x<(1<<SURFACE_N):
        raise ValueError("majority input out of range")
    return 1 if (x.bit_count()<3) else -1


def majority5_fourier(mask: int) -> Fraction:
    if type(mask) is not int or not 0<=mask<(1<<SURFACE_N):
        raise ValueError("majority mask out of range")
    return Fraction(sum(majority5_sign(x)*character(mask,x)
                        for x in range(1<<SURFACE_N)),1<<SURFACE_N)


def selector_sign(q: int, surface: int, left: int, right: int) -> int:
    """q=0 selects f5 surface; q=1 selects parity-lifted hidden H17."""
    if q not in (0,1):
        raise ValueError("selector mode must be binary")
    return majority5_sign(surface) if not q else lifted_sign(left,right)


def selector_global_singleton_sum() -> Fraction:
    """All singleton Fourier coefficients of the one-selector demo."""
    surface=sum((majority5_fourier(1<<i) for i in range(SURFACE_N)),
                Fraction())
    hidden=sum((lifted_fourier(1<<i,0)+lifted_fourier(0,1<<i)
                for i in range(HIDDEN_N)),Fraction())
    selector=(majority5_fourier(0)-hidden_fourier(0))/2
    return (surface+hidden)/2+selector


def smallest_parity_lift_check() -> dict:
    """Exhaustively verify spectral transport for ALL 16 Boolean h on 2 bits.

    A 2-bit Boolean function is given by four truth values. Its XOR-lift
    lives on four variables, making complete 16-variable-point checking
    possible independently of our analytic large-function code.
    """
    for table in range(1<<4):
        def signed(x):
            return sign((table>>x)&1)
        for left in range(4):
            for right in range(4):
                full=sum((signed(x^y)*character(left,x)*
                          character(right,y)
                          for x in range(4) for y in range(4)))
                if left!=right:
                    if full!=0:
                        raise AssertionError("singleton transport cancellation failed")
                else:
                    source=sum(signed(z)*character(left,z) for z in range(4))
                    if full!=4*source:
                        raise AssertionError("paired transport gain mismatch")
    return {"base_boolean_functions":16,
            "lifted_input_pairs_each":16,
            "Fourier_mask_pairs_each":16,
            "equal_mask_transport_verified":True,
            "singleton_coefficients_zero":True}




def adaptive_unpaired_transcript(n: int, x: int, y: int) -> tuple:
    """One adaptive query in each matched pair, never both endpoints.

    The side of coordinate i depends on the previously read bit.
    The transcript records (side,index,observed bit). Side 0 is x.
    """
    if type(n) is not int or not 1<=n<=8:
        raise ValueError("probe dimension must be 1..8")
    if any(type(v) is not int or not 0<=v<(1<<n) for v in (x,y)):
        raise ValueError("probe registers outside declared width")
    previous=0
    events=[]
    for i in range(n):
        side=0 if i==0 or previous==0 else 1
        value=((x if side==0 else y)>>i)&1
        events.append((side,i,value))
        previous=value
    return tuple(events)


def conditional_unpaired_law(n: int, xor_state: int) -> dict[tuple,Fraction]:
    """Transcript law conditioned on a fixed hidden XOR source.

    Conditioned on z, x stays uniform and y=x xor z. Exact probabilities
    come from enumerating 2**n possible x registers.
    """
    if type(n) is not int or not 1<=n<=8 or (
            type(xor_state) is not int or not 0<=xor_state<(1<<n)):
        raise ValueError("n=1..8 and xor_state in 0..2**n-1 required")
    counts={}
    for x in range(1<<n):
        transcript=adaptive_unpaired_transcript(n,x,x^xor_state)
        counts[transcript]=counts.get(transcript,0)+1
    return {key:Fraction(count,1<<n) for key,count in counts.items()}


def conditional_completed_pair_law(n: int,xor_state: int,index: int=0):
    """Reveal both x_i and y_i, exposing the hidden z_i exactly."""
    if type(n) is not int or not 1<=n<=8 or (
            type(xor_state) is not int or not 0<=xor_state<(1<<n)) or (
            type(index) is not int or not 0<=index<n):
        raise ValueError("invalid paired probe")
    counts={}
    for x in range(1<<n):
        a=(x>>index)&1
        b=((x^xor_state)>>index)&1
        counts[(a,b)]=counts.get((a,b),0)+1
    return {key:Fraction(count,1<<n) for key,count in counts.items()}


def adaptive_probe_control(max_n: int=5) -> dict:
    """Exact finite conditional independence for an adaptive unpaired policy."""
    if type(max_n) is not int or not 1<=max_n<=8:
        raise ValueError("max_n must be in 1..8")
    cases=0
    for n in range(1,max_n+1):
        base=conditional_unpaired_law(n,0)
        if len(base)!=(1<<n) or set(base.values())!={Fraction(1,1<<n)}:
            raise AssertionError("unpaired transcript should be uniform")
        for z in range(1<<n):
            if conditional_unpaired_law(n,z)!=base:
                raise AssertionError("adaptive unmatched probe leaked hidden XOR")
            cases+=1
        # If the completed source parity bit differs, the paired
        # transcript supports are disjoint.
        a=conditional_completed_pair_law(n,0)
        b=conditional_completed_pair_law(n,1)
        if set(a)&set(b):
            raise AssertionError("completed pair failed to reveal hidden bit")
    return {
        "max_n":max_n,
        "conditional_source_states_checked":cases,
        "adaptive_unpaired_transcripts_source_independent":True,
        "one_completed_pair_reveals_one_hidden_bit":True,
        "adaptive_query_bound_per_source_dimension":"one per XOR pair",
    }

def exact_controls() -> dict:
    rejection=exact_base_rejection_count()
    if rejection!=38416:
        raise AssertionError("16-bit block acceptance count changed")
    sensitivity=hidden_sensitivity()
    if sensitivity["base"]["sensitivity_global"]!=4:
        raise AssertionError("global base sensitivity changed")
    if sensitivity["global_hidden17"]!=5 or (
            sensitivity["zero_disjoint_blocks_lower"]!=9):
        raise AssertionError("hidden block / sensitivity fixture changed")
    p=Fraction(rejection,1<<BASE_N)
    mu=hidden_fourier(0)
    sentinel=hidden_fourier(SENTINEL)
    if mu!=Fraction(-1695,4096) or sentinel!=Fraction(2401,4096):
        raise AssertionError("exact hidden mean/sentinel coefficient changed")
    if any(hidden_fourier(1<<i) for i in range(BASE_N)):
        raise AssertionError("unexpected base singleton signal")
    if hidden_fourier(BASE_MASK|SENTINEL)!=Fraction(1,4096):
        raise AssertionError("full hidden Fourier degree changed")
    fourth_order_masks=[
        (1 << (4*b+i)) | (1 << (4*b+j))
        for b in range(4) for i in range(4) for j in range(i+1,4)
    ]
    fourth_order_coeffs=[hidden_fourier(mask) for mask in fourth_order_masks]
    if len(fourth_order_coeffs)!=24 or set(map(abs,fourth_order_coeffs))!={Fraction(343,4096)}:
        raise AssertionError("expected 24 paired fourth-order coefficients")
    mu_full=hidden_fourier(0)
    sentinel_full=hidden_fourier(SENTINEL)
    fourth_order_mse=(1-mu_full**2-sentinel_full**2-
                      sum((c*c for c in fourth_order_coeffs),Fraction()))
    if lifted_fourier(SENTINEL,SENTINEL)!=sentinel:
        raise AssertionError("paired sentinel probe disappeared")
    if any(lifted_fourier(1<<i,0) or lifted_fourier(0,1<<i)
           for i in range(HIDDEN_N)):
        raise AssertionError("parity lift has a singleton Fourier coefficient")
    source_sum=sum((majority5_fourier(1<<i) for i in range(SURFACE_N)),
                   Fraction())
    if source_sum!=Fraction(15,8):
        raise AssertionError("majority5 singletons changed")
    full=selector_global_singleton_sum()
    if full!=Fraction(9375,8192):
        raise AssertionError("selector Fourier composition identity changed")
    small=smallest_parity_lift_check()
    adaptive=adaptive_probe_control(5)
    return {
        "status":"EXACT_FINITE_FOURIER_CONTROL",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_width":BASE_N,"hidden_width":HIDDEN_N,
        "parity_lift_width":2*HIDDEN_N,
        "surface_width":SURFACE_N,"selector_width":1,
        "base_rejection_count":rejection,
        "base_rejection_probability":str(p),
        "hidden_mean_sign":str(mu),
        "hidden_sentinel_fourier":str(sentinel),
        "hidden_base_pair_coeff_0_1":str(hidden_fourier(3)),
        "hidden_full_degree_certificate":str(hidden_fourier(BASE_MASK|SENTINEL)),
        "hidden_fourier_degree":17,
        "parity_lift_fourier_degree":34,
        "selector_fourier_degree":35,
        "surface_singleton_sum":str(source_sum),
        "selector_global_singleton_sum":str(full),
        "hidden_probe_mse_base":str(1-mu**2),
        "hidden_probe_mse_pair_sentinel":str(1-mu**2-sentinel**2),
        "hidden_probe_pair_mse_gain":str(sentinel**2),
        "hidden_block_fourth_order_count":len(fourth_order_coeffs),
        "hidden_block_fourth_order_abs_coeff":"343/4096",
        "hidden_probe_41_feature_mse_population":str(fourth_order_mse),
        "hidden_zero_witness":sensitivity,
        "exact_small_spectral_transport":small,
        "adaptive_unpaired_probe_control":adaptive,
        "openai_132_asymptotic_witness_reproduced":False,
        "openai_192_unbounded_ratio_witness_reproduced":False,
        "trained_transformer_checkpoint_tested":False,
    }


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args(argv)
    result=exact_controls()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
