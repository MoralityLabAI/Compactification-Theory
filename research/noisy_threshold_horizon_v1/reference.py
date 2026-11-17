"""NHB-01: exact finite noisy-cube monitoring with source-scoped bounds.

Probabilities, influences and spectra use Fraction. Shannon entropies and
inverse-entropy source envelopes are numerical and are labeled accordingly.
The terminal observer sees only the noisy state, not the original target,
a persistent copy of it, or the intervening trajectory.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path


def noise_rate(p):
    p=F(p)
    if not 0<=p<=F(1,2):
        raise ValueError('noise probability must be in [0,1/2]')
    return p


def width(values):
    size=len(values)
    if size<2 or size & (size-1) or size>1024:
        raise ValueError('truth/signal vector length must be 2**n, 1<=n<=10')
    return size.bit_length()-1


def signs(values):
    width(values)
    if any(type(x) is not int or x not in (-1,1) for x in values):
        raise ValueError('truth values must be integer signs')
    return tuple(values)


def char(mask,x):
    return -1 if (mask & x).bit_count()%2 else 1


def crossover(p,steps):
    p=noise_rate(p)
    if type(steps) is not int or steps<0:
        raise ValueError('nonnegative integer horizon required')
    return (1-(1-2*p)**steps)/2


def compose(p,q):
    p,q=noise_rate(p),noise_rate(q)
    return p+q-2*p*q


def spectrum(truth):
    data=list(signs(truth)); size=len(data); stride=1
    while stride<size:
        for start in range(0,size,2*stride):
            for j in range(stride):
                a,b=data[start+j],data[start+j+stride]
                data[start+j]=a+b; data[start+j+stride]=a-b
        stride*=2
    return tuple(F(v,size) for v in data)


def influence(truth):
    truth=signs(truth); n=width(truth); size=len(truth)
    return tuple(F(sum(truth[x]!=truth[x^(1<<j)] for x in range(size)),size)
                 for j in range(n))


def noise_signal(values,p):
    n=width(values); p=noise_rate(p); data=[F(v) for v in values]
    for j in range(n):
        bit=1<<j
        for x in range(len(data)):
            if not x & bit:
                a,b=data[x],data[x|bit]
                data[x]=(1-p)*a+p*b
                data[x|bit]=p*a+(1-p)*b
    return tuple(data)


def brute_signal(values,p):
    n=width(values); p=noise_rate(p)
    return tuple(sum((F(values[x])*p**((x^y).bit_count())*
                           (1-p)**(n-(x^y).bit_count())
                      for x in range(len(values))),F()) for y in range(len(values)))


def entropy(p):
    p=float(p)
    if not 0<=p<=1:
        raise ValueError('entropy argument outside [0,1]')
    return 0.0 if p in (0.0,1.0) else -p*math.log2(p)-(1-p)*math.log2(1-p)


def information_from_signal(values):
    vals=tuple(F(x) for x in values)
    if not vals or any(abs(v)>1 for v in vals):
        raise ValueError('conditional means must lie in [-1,1]')
    mean=sum(vals,F())/len(vals)
    return max(0.0,entropy((1+mean)/2)-math.fsum(entropy((1+x)/2) for x in vals)/len(vals))


def source_information_cap(initial_bits,p):
    """#119 envelope in bits: psi(rho*psi^{-1}(I0)), numerical only."""
    p=noise_rate(p)
    if not -1e-12<=initial_bits<=1+1e-12:
        raise ValueError('binary initial information must be in [0,1]')
    I=min(1.0,max(0.0,float(initial_bits)))
    if I==0: return 0.0
    lo,hi=0.0,1.0
    for _ in range(75):
        mid=(lo+hi)/2
        if 1-entropy((1-mid)/2)<I: lo=mid
        else: hi=mid
    amplitude=(lo+hi)/2
    return max(0.0,1-entropy((1-float(1-2*p)*amplitude)/2))


def threshold_truth(n,terms):
    if type(n) is not int or not 1<=n<=10 or not terms:
        raise ValueError('finite nonempty polynomial on n=1..10 required')
    if any(type(m) is not int or not 0<=m<(1<<n) for m in terms):
        raise ValueError('polynomial mask outside cube')
    return tuple(1 if sum((F(c)*char(m,x) for m,c in terms.items()),F())>=0 else -1
                 for x in range(1<<n))


def affine_projection(truth):
    coeff=spectrum(truth); n=width(truth)
    return threshold_truth(n,{0:coeff[0],**{1<<j:coeff[1<<j] for j in range(n)}})


def analyze(truth,p,eta=0,readout=None):
    truth=signs(truth); p=noise_rate(p); eta=noise_rate(eta)
    n=width(truth); size=len(truth); a=1-2*eta
    if readout is None: readout=truth
    readout=signs(readout)
    if len(readout)!=size: raise ValueError('readout dimension mismatch')
    coeff=spectrum(truth); inf=influence(truth); AS=sum(inf,F())
    u=tuple(a*v for v in truth); posterior=noise_signal(u,p)
    err=(1-sum((abs(v) for v in posterior),F())/size)/2
    same=(1-sum((F(f)*v for f,v in zip(truth,posterior)),F())/size)/2
    compressed=(1-sum((F(g)*v for g,v in zip(readout,posterior)),F())/size)/2
    alpha=F(sum(f!=g for f,g in zip(truth,readout)),size)
    ns=(same-eta)/a if a else (1-sum((F(f)*v for f,v in zip(truth,noise_signal(truth,p))),F())/size)/2
    rho=1-2*p
    spectral_ns=sum(((1-rho**m.bit_count())*c*c for m,c in enumerate(coeff)),F())/2
    energy=sum((v*v for v in posterior),F())/size
    energy_fourier=a*a*sum((rho**(2*m.bit_count())*c*c for m,c in enumerate(coeff)),F())
    if ns!=spectral_ns or energy!=energy_fourier or AS!=sum((m.bit_count()*c*c for m,c in enumerate(coeff)),F()):
        raise AssertionError('independent Fourier and direct noise identities disagree')
    if ns>p*AS or (1-2*err)**2>energy:
        raise AssertionError('influence or L2-Bayes certificate failed')
    if abs(compressed-same)>a*alpha or err>compressed:
        raise AssertionError('compression triangle or Bayes optimality failed')
    balanced=coeff[0]==0
    if balanced and err<compose(eta,p):
        raise AssertionError('balanced terminal-only lower bound failed')
    I0=information_from_signal(u); I=information_from_signal(posterior)
    cap=source_information_cap(I0,p)
    if I>cap+2e-11:
        raise AssertionError('finite instance violates claimed source #119 envelope')
    return {'width':n,'balanced':balanced,'mean_sign':str(coeff[0]),
            'target_positive_prior':str((1+a*coeff[0])/2),
            'noise':str(p),'target_flip_probability':str(eta),
            'influences':[str(v) for v in inf],'average_sensitivity':str(AS),
            'real_Fourier_degree':max((m.bit_count() for m,c in enumerate(coeff) if c),default=0),
            'first_level_mass':str(sum((coeff[1<<i]**2 for i in range(n)),F())),
            'same_function_error':str(same),'Bayes_terminal_error':str(err),
            'compressed_readout_error':str(compressed),'uniform_compression_error':str(alpha),
            'noise_sensitivity':str(ns),'posterior_squared_energy':str(energy),
            'balanced_error_floor':str(compose(eta,p)) if balanced else None,
            'initial_information_bits_numeric':I0,'terminal_information_bits_numeric':I,
            'source_119_cap_bits_numeric':cap,
            'source_119_full_proof_reproduced':False}


def examples():
    product={(1<<i)|(1<<j):1 for i in range(3) for j in range(3,6)}
    specs=[('coordinate7',7,{1:1}),('majority7',7,{1<<i:1 for i in range(7)}),
           ('parity7',7,{127:1}),('product_majorities6',6,product),
           ('rare_alert7',7,{0:-6,**{1<<i:1 for i in range(7)}})]
    return [(name,n,terms,threshold_truth(n,terms)) for name,n,terms in specs]


def monitor_receipt():
    rows=[]
    for name,n,terms,truth in examples():
        readout=affine_projection(truth)
        degree=max(m.bit_count() for m,c in terms.items() if c)
        for horizon in (0,1,2,4,5,8,16):
            p=crossover(F(1,20),horizon)
            row=analyze(truth,p,F(1,10),readout)
            row.update(name=name,horizon=horizon,verified_PTF_degree_upper=degree,
                       source_127_influence_upper_numeric=8*degree*math.sqrt(n))
            rows.append(row)
    rare=analyze(threshold_truth(7,{0:-6,**{1<<i:1 for i in range(7)}}),F(1,2),0)
    n=129
    large_AS=F(n*math.comb(n-1,(n-1)//2),2**(n-1))
    return {'schema':'nhb01.noisy_horizon.v1','status':'EXACT_FINITE_NOISY_CUBE_CONTROL',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'paper_families':[119,127],'step_noise':'1/20','target_flip':'1/10',
            'cases':rows,'fully_noisy_rare_alert_control':rare,
            'large_majority_control':{'n':n,'PTF_degree':1,'exact_average_sensitivity':str(large_AS),
                 'exact_average_sensitivity_float':float(large_AS),
                 'source_127_upper_numeric':8*math.sqrt(n),'trivial_upper':n},
            'entropy_is_numerical':True,'probabilities_are_exact_rationals':True,
            'trained_neural_checkpoint_evaluated':False,'history_memory_available':False,
            'source_asymptotic_proofs_reproduced':False}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',type=Path)
    args=ap.parse_args(argv); report=monitor_receipt()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    brief=[{'name':r['name'],'T':r['horizon'],
            'Bayes_error':float(F(r['Bayes_terminal_error'])),
            'same_error':float(F(r['same_function_error'])),
            'I_bits':r['terminal_information_bits_numeric'],
            'cap_bits':r['source_119_cap_bits_numeric']}
           for r in report['cases'] if r['horizon'] in (1,4,5,8)]
    print(json.dumps({'status':report['status'],'rows':brief,
                     'rare_at_half':report['fully_noisy_rare_alert_control'],
                     'majority129':report['large_majority_control']},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
