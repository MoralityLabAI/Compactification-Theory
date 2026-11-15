"""Independent exact-rational controls; these are new, not recovered old tests.

Finite tests are regression evidence, not proofs of the general resource theorem.
All coefficients below are evaluated as Fraction; no floating point tolerances.
"""
from fractions import Fraction as F
from itertools import product
from math import comb, factorial
from pathlib import Path
import hashlib
import json
import random

rng = random.Random(15112026)
counts = {}

def check(name, predicate):
    assert predicate, name
    counts[name] = counts.get(name, 0) + 1

def relu(x):
    return max(F(0), x)

def nearest(x):
    return (x + F(1, 2)).numerator // (x + F(1, 2)).denominator

def multiindices(d, k):
    return [a for a in product(range(k + 1), repeat=d) if sum(a) <= k]

def trans(indices, v):
    return [[F(0) if any(g < b for b, g in zip(beta, gamma)) else
             F(prod(comb(g, b) * x ** (g-b) for b, g, x in zip(beta, gamma, v)))
             for gamma in indices] for beta in indices]

def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p

def mv(A, x):
    return [sum((a*b for a, b in zip(row, x)), F(0)) for row in A]

def mm(A, B):
    return [[sum((a*b for a, b in zip(row, col)), F(0)) for col in zip(*B)] for row in A]

for d in (1, 2, 3):
    for k in range(4):
        ix = multiindices(d, k)
        check("jet_dimension", len(ix) == comb(d+k, d))
        for _ in range(20):
            a = [rng.randrange(-3, 4) for _ in range(d)]
            b = [rng.randrange(-3, 4) for _ in range(d)]
            check("translation_group_identity", mm(trans(ix, a), trans(ix, b)) == trans(ix, [x+y for x, y in zip(a,b)]))
        e = [F(rng.randrange(-4, 5), 8) for _ in ix]
        for step in range(120):
            direction = [0]*d
            direction[step % d] = 1 if step % 2 else -1
            A = trans(ix, direction)
            transported = mv(A, e)
            r = [F(rng.randrange(-24, 25), 8) for _ in ix]
            c = [nearest(x-y) for x,y in zip(r, transported)]
            new_e = [x+y-z for x,y,z in zip(transported, c, r)]
            A_norm = max(sum(abs(x) for x in row) for row in A)
            check("corrected_jet_invariant", all(abs(x) <= F(1,2) for x in new_e))
            check("correction_alphabet", max(abs(x) for x in c) <= max(abs(x) for x in r) + A_norm/2 + F(1,2))
            check("integer_neighbor_transport", all(x.denominator == 1 for row in A for x in row))
            e = new_e

def pack(bits, sentinel=False):
    z = sum((F(1+b, 4**(i+1)) for i,b in enumerate(bits)), F(0))
    return z + (F(1,4**(len(bits)+1)) if sentinel else 0)

def digit(y):
    return 4*(relu(4*(y-F(7,16)))-relu(4*(y-F(1,2))))

for n in range(1, 11):
    for bits in product((0,1), repeat=n):
        for sentinel in (False, True):
            y = pack(bits, sentinel)
            for b in bits:
                q = digit(y)
                check("base4_digit_exact", q == b)
                y = 4*y-1-q
        y = pack(bits, True)
        perturbation = F(1, 192*4**(n-1))
        for sign in (-1, 1):
            noisy = y + sign*perturbation
            for b in bits:
                q = digit(noisy)
                check("sentinel_digit_robustness", q == b)
                noisy = 4*noisy-1-q

# This is an actual obstruction to a uniform two-sided robustness claim for
# the old finite code with no sentinel: the last leading-one state is 1/2.
eta = F(1, 2**20)
no_sentinel = {"code": "1/2", "perturbation": str(-eta), "decoded": str(digit(F(1,2)-eta))}
check("no_sentinel_counterexample", digit(F(1,2)-eta) != 1)
check("old_sign_counterexample", abs(-nearest(F(1))-1) == 2)

def lookup(values, x):
    n = len(values)
    if n == 1:
        return values[0]
    coeffs = [values[1]-values[0]]
    coeffs += [values[j+1]-2*values[j]+values[j-1] for j in range(1,n-1)]
    coeffs += [-(values[-1]-values[-2])]
    g = sum((c*relu(x-F(j,n)) for j,c in enumerate(coeffs)), F(0))
    # This implements a narrow signed pair followed by log2(n) doublings.
    pos, neg = relu(g), relu(-g)
    for _ in range(n.bit_length()-1):
        pos, neg = 2*pos, 2*neg
    return values[0]+pos-neg

for q in range(1, 7):
    n=2**q
    values=[F(rng.randrange(257),256) for _ in range(n)]
    for j in range(n):
        check("bounded_lookup_nodes", lookup(values,F(j,n)) == values[j])
    for j in range(n-1):
        for t in (F(1,4),F(1,2),F(3,4)):
            check("bounded_lookup_interpolation", lookup(values,F(j,n)+t/n) == (1-t)*values[j]+t*values[j+1])
    # All dense affine slots of the variable-width architecture are charged,
    # including zero slots: 1 -> n -> 2 -> [2]*q -> 1.
    widths=[1,n,2]+[2]*q+[1]
    P=sum((a+1)*b for a,b in zip(widths,widths[1:]))
    check("lookup_parameter_accounting", P == 4*n+6*q+5)

def tent(x):
    return 2*relu(x)-4*relu(x-F(1,2))+2*relu(x-1)

def square(x,m):
    y=x
    z=x
    for j in range(1,m+1):
        z=tent(z)
        y-=F(1,4**j)*z
    return y

for m in range(1,7):
    for j in range(129):
        x=F(j,128)
        check("triangular_square_error", 0 <= square(x,m)-x*x <= F(1,4**(m+1)))
    for _ in range(100):
        x,y=F(rng.randrange(129),128),F(rng.randrange(129),128)
        xy=2*square((x+y)/2,m)-(square(x,m)+square(y,m))/2
        check("multiplication_error", abs(xy-x*y) <= F(3,4**(m+1)))

def circle_distance(x):
    frac=x-(x.numerator//x.denominator)
    return min(frac,1-frac)

for d in range(1,7):
    Q=2*d+1
    D=2**((Q-1).bit_length())
    rho=F(1,4*D)
    for _ in range(180):
        x=[F(rng.randrange(-128,257),128) for _ in range(d)]
        good=[]
        for t in range(Q):
            safe=all(circle_distance(z-F(t,D)) > rho for z in x)
            if safe:
                good.append(t)
        check("shifted_grid_coverage", len(good)>=d+1)
        outputs=[F(1000*(-1)**t) for t in range(Q)]
        for t in good:
            outputs[t]=F(rng.randrange(-10,11),100)
        check("median_stitching", abs(sorted(outputs)[d])<=F(1,10))

# A fresh finite packed quadratic example. This is deliberately not labelled a
# reproduction of the unavailable 863-point historical test.
h=F(1,32)
tau=h*h
u=[F(0),F(0)]
v=[F(nearest(z)) for z in u]
corrections=[]
jets=[v[:]]
T=[[F(1),F(1)],[F(0),F(1)]]
for i in range(31):
    x=F(i,32)
    nxt=[(x+h)**2/tau,2*(x+h)*h/tau]
    r=[a-b for a,b in zip(nxt,mv(T,u))]
    e=[a-b for a,b in zip(v,u)]
    c=[nearest(a-b) for a,b in zip(r,mv(T,e))]
    corrections.append(c)
    v=[a+b for a,b in zip(mv(T,v),c)]
    u=nxt
    jets.append(v[:])
alphabet=4
bits=[]
for c in corrections:
    for z in c:
        assert -alphabet <= z <= alphabet
        bits += [(z+alphabet)>>j&1 for j in range(4)]
encoded=pack(bits,True)
decoded=[]
y=encoded
for _ in bits:
    q=digit(y)
    decoded.append(int(q))
    y=4*y-1-q
check("quadratic_packed_stream", decoded==bits)
max_error=F(0)
for j in range(1025):
    x=F(j,1024)
    i=min(int(x/h),31)
    local=(x-i*h)/h
    pred=tau*(jets[i][0]+jets[i][1]*local)
    max_error=max(max_error,abs(pred-x*x))
    check("quadratic_uniform_grid", abs(pred-x*x)<=h*h)

report={"status":"passed", "seed":15112026, "counts":counts,
        "total_assertions":sum(counts.values()),
        "counterexamples":{"no_sentinel_robustness":no_sentinel,"wrong_sign":{"t":1,"u":0,"v":0,"r":1,"old_error":-2,"correct_error":0}},
        "quadratic":{"points":1025,"h":str(h),"bits":len(bits),"stored_fraction_bits":2*(len(bits)+1),"maximum_error":str(max_error),"bound":str(h*h)},
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"New finite exact-arithmetic controls, not a universal theorem or historical reproduction"}
Path(__file__).with_name("quantization_control_results.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
