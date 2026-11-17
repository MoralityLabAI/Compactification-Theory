"""SCM-05: fixed-range, saturate-free uniform chart quantization.

The decoded object is a *polygonal* loop of projected features. This module
certifies equivalence with the original unknown continuous loop conditional
on externally established Lipschitz upper bounds for the ORIGINAL loop.
Quantizing each theta independently produces discrete samples; the decoder
then interpolates linearly between quantized samples. The quantizer itself
is not asserted to preserve a continuous neural computation.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Mapping

from continuous import (_data, _closed, _gaps, _bounds, _relative, rat,
                        _sqrt_bounds, norm_squared, min_segment_distance_sq)


def uniform_quantize(frames, *, bits: int, range_abs) -> list[list[list[Fraction]]]:
    """2^bits center-of-cell reconstruction levels on the exact interval [-R,R]."""
    if not isinstance(bits, int) or isinstance(bits,bool) or not (1<=bits<=40):
        raise ValueError('bits must be an integer in 1..40')
    R=rat(range_abs)
    if R<=0:
        raise ValueError('range_abs must be strictly positive')
    frames=_data(frames)
    levels=1<<bits
    step=2*R/levels
    result=[]
    for row in frames:
        new=[]
        for point in row:
            coords=[]
            for x in point:
                if not (-R<=x<=R):
                    raise ValueError('Quantization saturation would invalidate error bound')
                index=min(levels-1,int((x+R)//step))
                coords.append(-R+(Fraction(index,1)+Fraction(1,2))*step)
            new.append(coords)
        result.append(new)
    return result


def continuous_to_quantized_polygon(frames, *, bits: int, range_abs,
                                    interval_lengths, pair_lipschitz_bounds,
                                    extra_margin='0', digits=12) -> dict:
    """Conditional braid preservation by sample quantization followed by PL decode.

    Original continuous q differs from PL(original) by <= L*dtheta/2.
    Quantized PL relative q differs from PL(original) by <= max endpoint
    relative quantization error. Collision-free straight homotopies through
    PL(original) give identical closed-braid/link classes when BOTH
    perturbations are smaller than original polygonal clearance.

    Uniform range R, b bits/coordinate ensures each component's absolute
    coordinate error <= R/2^b and pair relative vector error <=
    2 sqrt(2) R/2^b. This worst-case bound is independently checked.
    """
    a=_data(frames)
    if not _closed(a):
        return {'passes':False,'reason':'sampled_endpoints_not_exactly_equal',
                'scope':'uncertified', 'pairs':{}}
    gaps=_gaps(interval_lengths,len(a)-1)
    bounds=_bounds(pair_lipschitz_bounds,len(a[0]),len(gaps))
    margin=rat(extra_margin)
    if margin <0:
        raise ValueError('Extra margin must be nonnegative')
    decoded=uniform_quantize(a,bits=bits,range_abs=range_abs)
    if not _closed(decoded):
        raise AssertionError('Deterministic closed-loop quantizer failed to close')
    R=rat(range_abs)
    sqrt2_hi=_sqrt_bounds(Fraction(2),digits)[1]
    pair_bound=2*sqrt2_hi*R/(1<<bits)
    report={}
    for i,j in combinations(range(len(a[0])),2):
        details=[]
        for k,gap in enumerate(gaps):
            p0,p1=_relative(a[k],i,j),_relative(a[k+1],i,j)
            h0,h1=_relative(decoded[k],i,j),_relative(decoded[k+1],i,j)
            m=_sqrt_bounds(min_segment_distance_sq(p0,p1),digits)[0]
            alias=bounds[i,j][k]*gap/2
            quant_error=_sqrt_bounds(max(norm_squared((p0[0]-h0[0],p0[1]-h0[1])),
                                  norm_squared((p1[0]-h1[0],p1[1]-h1[1]))),digits)[1]
            details.append({
                'interval':k,
                'passes_actual_errors':bool(m>max(alias,quant_error)+margin),
                'passes_worst_case_bound':bool(m>max(alias,pair_bound)+margin),
                'PL_clearance_lower':float(m),
                'original_alias_upper':float(alias),
                'actual_relative_quant_error_upper':float(quant_error),
            })
        report[f'{i},{j}']={
            'passes_actual_errors':all(d['passes_actual_errors'] for d in details),
            'passes_worst_case_bound':all(d['passes_worst_case_bound'] for d in details),
            'intervals':details,
        }
    actual=all(x['passes_actual_errors'] for x in report.values())
    worst=all(x['passes_worst_case_bound'] for x in report.values())
    return {'passes': actual,'worst_case_passes':worst,'bits_per_coordinate':bits,
            'range_abs':str(R),'worst_case_pair_error_upper':float(pair_bound),
            'decoded_representation':'piecewise_linear_interpolation_of_quantized_samples',
            'scope':'conditional_on_external_original_path_lipschitz_proofs',
            'pairs':report,'reason':'sufficient_margin' if actual else 'insufficient_margin'}


def quantization_bit_knee(frames, *, range_abs, interval_lengths,
                          pair_lipschitz_bounds, max_bits=24,
                          extra_margin='0', digits=12) -> dict:
    """Smallest tested b yielding a worst-case uniform-error braid certificate."""
    if max_bits<1 or max_bits>40:
        raise ValueError('max_bits must lie in 1..40')
    for bits in range(1,max_bits+1):
        result=continuous_to_quantized_polygon(frames,bits=bits,
                    range_abs=range_abs,interval_lengths=interval_lengths,
                    pair_lipschitz_bounds=pair_lipschitz_bounds,
                    extra_margin=extra_margin,digits=digits)
        if result['worst_case_passes']:
            return {'certified':True,'bit_knee':bits,'scope':result['scope'],
                    'tested_max_bits':max_bits,'certificate':result}
    return {'certified':False,'bit_knee':None,'tested_max_bits':max_bits,
            'reason':'no_sufficient_worst_case_bound_within_budget'}
