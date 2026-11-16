"""Exact implementation controls for Hamming-selected Taylor frame updates.

Compares the actual affine/ReLU compiler with a direct matrix recurrence, and
compares binomial matrices with polynomial substitution built by multiplication.
These finite checks do not replace the general Lean proofs or analytic bounds.
"""
from fractions import Fraction as F
from pathlib import Path
from math import comb
import hashlib
import json
import random
from spatial_circuit import Net, aff, read_bit, indices, transport, mv, snake, snake_rank

HERE=Path(__file__).resolve().parent
RNG=random.Random(1611202663)
counts={}
def check(name,value):
    assert value,name
    counts[name]=counts.get(name,0)+1
def digits(p,q):return [(q>>j)&1 for j in range(p)]
def pack(bits):
    z=F(1,4)
    for b in reversed(bits):z=(1+b+z)/4
    return z
def affine_mul(h,coefficients):
    out=[F(0)]*(len(coefficients)+1)
    for j,c in enumerate(coefficients):out[j]+=h*c;out[j+1]+=c
    return out
def power_shift(h,n):
    out=[F(1)]
    for _ in range(n):out=affine_mul(h,out)
    return out
def polynomial_shift(ix,v,axis,h):
    result={b:F(0) for b in ix}
    for g,c in zip(ix,v):
        for beta_axis,coefficient in enumerate(power_shift(h,g[axis])):
            b=tuple(beta_axis if j==axis else x for j,x in enumerate(g))
            result[b]+=c*coefficient
    return [result[b] for b in ix]

def frame_net(ix,D,p,a,axis,negative,C,centre,tau):
    J=len(ix)
    net=Net(['code']+[f'v{j}' for j in range(J)])
    for b in range(p):read_bit(net,f'q{b}')
    net.emit({f'choose{q}':aff({f'q{b}':1 if (q>>b)&1 else -1 for b in range(p)},
        1-sum(digits(p,q)),True) for q in range(D+1)},label='Hamming amplitude')
    net.drop(*[f'q{b}' for b in range(p)])
    for j in range(J):
        for b in range(a):read_bit(net,f'bit{b}')
        net.emit({f'cor{j}':aff({f'bit{b}':2**b for b in range(a)},-centre)},label='signed correction')
        net.drop(*[f'bit{b}' for b in range(a)])
    net.emit({f'pred{q}_{j}':aff({f'v{l}':T[j][l] for l in range(J)})
        for q in range(D+1) for T in [transport(ix,axis,(-1 if negative else 1)*q)] for j in range(J)},
        label='concrete Taylor alternatives')
    budgets=[sum(F(comb(g[axis],b[axis])*D**(g[axis]-b[axis]))
        for g in ix if g[axis]>=b[axis]) for b in ix]
    net.gate([(f'take{q}_{j}',f'choose{q}',f'pred{q}_{j}',C*budgets[j])
        for q in range(D+1) for j in range(J)])
    net.emit({f'v{j}':aff({f'take{q}_{j}':1 for q in range(D+1)}|{f'cor{j}':tau})
        for j in range(J)},label='selected transport and correction')
    net.clip([f'v{j}' for j in range(J)],-C,C)
    return net,budgets

def main():
    for d in [1,2,3]:
        for k in [0,1,2,3]:
            ix=indices(d,k)
            for axis in range(d):
                for h in range(-3,4):
                    v=[F(RNG.randrange(-9,10),8) for _ in ix]
                    check('polynomial_substitution',mv(transport(ix,axis,h),v)==polynomial_shift(ix,v,axis,h))
                    check('integral_transport_entries',all(t.denominator==1 for row in transport(ix,axis,h) for t in row))
    step_count=0
    stream_count=0
    maximum_weight=F(0)
    for d,k,D,p,a,centre in [(1,0,0,0,0,0),(1,3,2,2,4,7),(2,2,2,2,4,7),(3,2,2,2,4,7)]:
        ix=indices(d,k);J=len(ix);K=2;C=F(2);tau=F(1,32)
        sites=snake(d,K);B=len(sites)
        nets={(axis,negative):frame_net(ix,D,p,a,axis,negative,C,centre,tau)
            for axis in range(d) for negative in [False,True]}
        maximum_weight=max(maximum_weight,*(n.max_weight for n,_ in nets.values()))
        for trial in range(4):
            frames=[]
            for t in range(B-1):
                axis=next(j for j in range(d) if sites[t][j]!=sites[t+1][j])
                negative=sites[t+1][axis]<sites[t][axis]
                q=(t+trial)%(D+1)
                correction=[RNG.randrange(2*centre+1) for _ in ix]
                bits=digits(p,q)+[b for c in correction for b in digits(a,c)]
                frames.append((axis,negative,q,correction,bits))
            bits=[b for frame in frames for b in frame[-1]]
            code=pack(bits)
            initial=[F(RNG.randrange(-16,17),8) for _ in ix]
            states=[initial]
            v=initial
            consumed=0
            for axis,negative,q,correction,frame_bits in frames:
                net,budgets=nets[axis,negative]
                output=net.evaluate([code,*v]);compiled=net.evaluate_compiled([code,*v])
                check('signed_pair_compiler_agrees',output==compiled)
                predicted=mv(transport(ix,axis,(-1 if negative else 1)*q),v)
                for j in range(J):check('concrete_gate_row_bound',abs(predicted[j])<=C*budgets[j])
                target=[max(-C,min(C,x+tau*(c-centre))) for x,c in zip(predicted,correction)]
                check('frame_transport_and_correction',[output[f'v{j}'] for j in range(J)]==target)
                check('hamming_one_hot',[output[f'choose{r}'] for r in range(D+1)]==[int(r==q) for r in range(D+1)])
                consumed+=len(frame_bits)
                check('sentinel_suffix_exact',output['code']==pack(bits[consumed:]))
                code=output['code'];v=target;states.append(v);step_count+=1
            for selected,site in enumerate(sites):
                # Lean's rank list is highest-coordinate-first; reverse the
                # physical coordinate order used by spatial_circuit.snake.
                def rank(high_first):
                    if not high_first:return 0
                    i,*tail=high_first;size=K**len(tail);r=rank(tail)
                    return i*size+(r if i%2==0 else size-1-r)
                check('rank_coordinate_convention',rank(list(reversed(site)))==snake_rank(site,K)==selected)
                captured=[F(0)]*J
                for t,state in enumerate(states):
                    b=max(F(0),1-abs(F(t-selected)))
                    for j,z in enumerate(state):
                        gate=max(F(0),z+C*b)-max(F(0),z-C*b)-C*b
                        captured[j]=max(-C,min(C,captured[j]+gate))
                check('concrete_spatial_capture',captured==states[selected])
            stream_count+=1
    result={'status':'passed','assertions':counts,'total_assertions':sum(counts.values()),
        'streams':stream_count,'decoded_frames':step_count,'maximum_test_gate_weight':str(maximum_weight),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Exact finite implementation controls; general Lean proofs remain the mathematical evidence. No analytic remainder or dense-resource theorem asserted.'}
    (HERE/'transport_capture_controls_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
