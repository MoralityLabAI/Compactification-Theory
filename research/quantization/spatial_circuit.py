"""Explicit layered rational ReLU circuits for the bounded spatial interface.

Each register is compiled into a positive/negative ReLU pair. All dense
affine slots, including zero coefficients and carried registers, are counted.
The evaluation API uses exact Fraction arithmetic. No primitive is a hidden
floor, multiplication, table lookup, discontinuity, or large scalar weight.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from math import comb, factorial
import hashlib
import json

def ceil_fraction(x):
    x=F(x)
    return -((-x.numerator)//x.denominator)

def nearest(x): return (F(x)+F(1,2)).__floor__()
def clip_value(x,lo=F(0),hi=F(1)): return max(lo,min(hi,x))
def is_power_two(n): return n>0 and n&(n-1)==0
def log2_int(n):
    assert is_power_two(n)
    return n.bit_length()-1
def indices(d,k): return list(b for b in product(range(k+1),repeat=d) if sum(b)<=k)

def snake(d,K):
    if d==0: return [()]
    low=snake(d-1,K)
    return [p+(j,) for j in range(K) for p in (low if j%2==0 else low[::-1])]

def snake_rank(p,K):
    rank=0
    for j,x in enumerate(p):
        rank=x*K**j+(rank if x%2==0 else K**j-1-rank)
    return rank

@dataclass
class Aff:
    terms: dict
    bias: F=F(0)
    relu: bool=False

def aff(terms=None,bias=0,relu=False):
    return Aff({k:F(v) for k,v in (terms or {}).items() if v},F(bias),relu)

class Net:
    def __init__(self,inputs):
        self.inputs=list(inputs)
        self.names=list(inputs)
        self.layers=[]
        self.events=[]
        self.max_weight=F(0)
        self.precision=0
        self.dense_parameters=0
        self.width=2*len(inputs)
    def emit(self,updates,keep=None,label=""):
        keep=list(self.names if keep is None else keep)
        out={n:aff({n:1}) for n in keep if n not in updates}
        out.update(updates)
        previous=set(self.names)
        for a in out.values():
            assert set(a.terms)<=previous,(label,set(a.terms)-previous)
            for x in [a.bias,*a.terms.values()]:
                self.max_weight=max(self.max_weight,abs(x))
                assert is_power_two(x.denominator),x
                self.precision=max(self.precision,log2_int(x.denominator))
        # Input pairs are introduced by an explicit first ReLU layer in the
        # compiled network. Its dense cost is accounted in certificate().
        self.dense_parameters+=(2*len(self.names)+1)*(2*len(out))
        self.width=max(self.width,2*len(out))
        self.layers.append(out)
        self.events.append((label,len(self.names),len(out)))
        self.names=list(out)
    def drop(self,*names):
        gone=set(names)
        self.emit({},keep=[n for n in self.names if n not in gone],label="release registers")
    def constant(self,name,x): self.emit({name:aff(bias=x)},label="constant")
    def clip(self,names,lo=0,hi=1):
        names=list(names); base=[n for n in self.names if n not in names]
        self.emit({"__clo_"+n:aff({n:1},-F(lo),True) for n in names}|
                  {"__chi_"+n:aff({n:1},-F(hi),True) for n in names},keep=base,label="clip hinges")
        self.emit({n:aff({"__clo_"+n:1,"__chi_"+n:-1},lo) for n in names},keep=base,label="clip combine")
    def amplify(self,names,exponent):
        for _ in range(exponent):
            self.emit({n:aff({n:2}) for n in names},label="shared doubling")
    def lookup(self,x,columns):
        n=len(next(iter(columns.values())))
        assert is_power_two(n) and all(len(v)==n for v in columns.values())
        if n==1:
            self.emit({name:aff(bias=values[0]) for name,values in columns.items()},label="one-entry table")
            return
        base=list(self.names)
        hinges=[f"__tab{j}" for j in range(n)]
        self.emit({h:aff({x:1},-F(j,n),True) for j,h in enumerate(hinges)},keep=base,label="table hinges")
        updates={}
        for name,values in columns.items():
            v=list(map(F,values))
            c=[v[1]-v[0]]+[v[j+1]-2*v[j]+v[j-1] for j in range(1,n-1)]+[-(v[-1]-v[-2])]
            updates[name]=aff(dict(zip(hinges,c)))
        self.emit(updates,keep=base,label="table aggregate")
        self.amplify(list(columns),log2_int(n))
        self.emit({name:aff({name:1},values[0]) for name,values in columns.items()},label="table offset")
    def gate(self,triples):
        """(output, bit register, bounded-value register, bound)."""
        base=list(self.names)
        updates={}
        for out,b,z,C in triples:
            updates["__gp_"+out]=aff({z:1,b:C},relu=True)
            updates["__gm_"+out]=aff({z:1,b:-C},relu=True)
        self.emit(updates,keep=base,label="binary gate hinges")
        self.emit({out:aff({"__gp_"+out:1,"__gm_"+out:-1,b:-C}) for out,b,z,C in triples},keep=base,label="binary gate combine")
    def evaluate(self,values):
        state={n:F(v) for n,v in zip(self.inputs,values)}
        for layer in self.layers:
            nxt={}
            for n,a in layer.items():
                v=a.bias+sum((c*state[t] for t,c in a.terms.items()),F(0))
                nxt[n]=max(F(0),v) if a.relu else v
            state=nxt
        return state
    def evaluate_compiled(self,values):
        """Independent evaluator of the actual signed-pair ReLU compilation."""
        z={n:(max(F(0),F(v)),max(F(0),-F(v))) for n,v in zip(self.inputs,values)}
        for layer in self.layers:
            out={}
            for n,a in layer.items():
                v=a.bias
                for t,c in a.terms.items(): v+=c*z[t][0]-c*z[t][1]
                out[n]=(max(F(0),v),F(0) if a.relu else max(F(0),-v))
            z=out
        return {n:p-m for n,(p,m) in z.items()}
    def certificate(self,output=None):
        nin=len(self.inputs)
        # First split and (when requested) scalar affine output reconstruction.
        P=self.dense_parameters+(nin+1)*2*nin
        if output is not None: P+=2*len(self.names)+1
        return {"hidden_layers":len(self.layers)+1,"depth_with_scalar_output":len(self.layers)+2,
                "dense_parameters":P,"maximum_width":self.width,
                "maximum_weight_or_bias":str(max(F(1),self.max_weight)),
                "fractional_bits":self.precision,"logical_layers":len(self.layers),
                "compilation":"Every register is a ReLU signed pair; all dense slots counted"}

def anchor(r,theta,h): return clip_value((r-1+theta)*h)

def transport(ix,axis,step):
    return [[F(comb(g[axis],b[axis])*step**(g[axis]-b[axis]))
             if g[axis]>=b[axis] and all(g[j]==b[j] for j in range(len(b)) if j!=axis) else F(0)
             for g in ix] for b in ix]

def mv(T,u): return [sum((a*b for a,b in zip(row,u)),F(0)) for row in T]

def encode_block(f_derivative,coarse,theta,params):
    d,K,D,ix,h,mu,tau,C,a,p=[params[n] for n in ("d","K","D","ix","h","mu","tau","C","a","p")]
    sites=snake(d,K)
    points=[tuple(anchor(c*K+i,theta,h) for c,i in zip(coarse,site)) for site in sites]
    def jet(y):
        return [F(f_derivative(beta,y))*mu**sum(beta)/prod_factorial(beta)/tau for beta in ix]
    states=[[nearest(v) for v in jet(y)] for y in points]
    bits=[]
    corrections=[]
    for t,(old,new) in enumerate(zip(sites,sites[1:])):
        axis=next(j for j in range(d) if old[j]!=new[j])
        sign=1 if new[axis]>old[axis] else -1
        step=(points[t+1][axis]-points[t][axis])/mu
        assert step.denominator==1 and 0<=sign*step<=D
        mag=int(sign*step)
        pred=mv(transport(ix,axis,int(step)),states[t])
        c=[int(v-w) for v,w in zip(states[t+1],pred)]
        assert max(map(abs,c),default=0)<=C,(coarse,theta,t,c,C)
        corrections.append(c)
        bits.extend((mag>>j)&1 for j in range(p))
        for z in c: bits.extend(((z+C)>>j)&1 for j in range(a))
    code=sum((F(1+b,4**(j+1)) for j,b in enumerate(bits)),F(0))+F(1,4**(len(bits)+1))
    return code,[tau*v for v in states[0]],states,points,bits

def prod_factorial(beta):
    z=1
    for b in beta: z*=factorial(b)
    return z

def parameters(d,s,n0,K):
    s=F(s); assert d>=1 and s>0 and is_power_two(n0) and is_power_two(K)
    k=ceil_fraction(s)-1; ix=indices(d,k); J=len(ix)
    Q=2*d+1; D=1<<(Q-1).bit_length(); rD=log2_int(D)
    h=F(1,n0*K); mu=h/D; q=log2_int(n0*K)
    tau=F(1,2**ceil_fraction(s*(q+rD)))
    AD=2**(k+1)*D**k
    C=2**(1+ceil_fraction(s*rD))+AD//2+1
    a=(2*C).bit_length(); p=D.bit_length()
    return dict(d=d,s=s,k=k,ix=ix,J=J,Q=Q,D=D,rD=rD,n0=n0,K=K,A=2*n0,N=(2*n0)**d,
                B=K**d,h=h,mu=mu,tau=tau,q=q,AD=AD,C=C,a=a,p=p,
                payload=(J*a+p)*(K**d-1))

def build_interface(f_derivative,params,phase):
    d,n0,K,A,N,D,h=[params[n] for n in ("d","n0","K","A","N","D","h")]
    theta=F(phase,D); delta=h/(8*D)
    net=Net([f"x{j}" for j in range(d)])
    base=list(net.names)
    hinges={}
    for j in range(d):
        for c in range(1,n0+1):
            t=(c*K-1+theta)*h
            hinges[f"cp{j}_{c}"]=aff({f"x{j}":1},-t+delta,True)
            hinges[f"cm{j}_{c}"]=aff({f"x{j}":1},-t-delta,True)
    net.emit(hinges,keep=base,label="coarse safe-cell hinges")
    net.emit({f"c{j}":aff({f"c{z}{j}_{c}":sgn for c in range(1,n0+1) for z,sgn in (("p",1),("m",-1))}) for j in range(d)},keep=base,label="coarse aggregate")
    net.amplify([f"c{j}" for j in range(d)],log2_int(2*D*K))
    net.emit({"address":aff({f"c{j}":F(A**(j+1),N) for j in range(d)})},label="lexicographic coarse address")
    codes=[]; initials=[[] for _ in params["ix"]]
    coarse_sites=[]
    for address in range(N):
        c=tuple(address//A**j%A for j in range(d))
        code,v,_,_,_=encode_block(f_derivative,c,theta,params)
        codes.append(code)
        for col,z in zip(initials,v): col.append(z)
        coarse_sites.append(c)
    net.lookup("address",{"code":codes}|{f"v{j}":v for j,v in enumerate(initials)})
    net.emit({f"u{j}":aff({f"x{j}":1,f"c{j}":-2},(1-theta)*h) for j in range(d)},label="normalized block coordinates before shared amplification")
    net.amplify([f"u{j}" for j in range(d)],log2_int(n0))
    net.clip([f"u{j}" for j in range(d)])
    if K==1:
        net.emit({f"i{j}":aff() for j in range(d)},label="one fine cell")
    else:
        base=list(net.names); delta_f=F(1,8*D*K)
        net.emit({f"f{z}{j}_{i}":aff({f"u{j}":1},-F(i,K)+sgn*delta_f,True)
                  for j in range(d) for i in range(1,K) for z,sgn in (("p",1),("m",-1))},keep=base,label="fine safe-cell hinges")
        net.emit({f"i{j}":aff({f"f{z}{j}_{i}":sgn for i in range(1,K) for z,sgn in (("p",1),("m",-1))}) for j in range(d)},keep=base,label="fine aggregate")
        net.amplify([f"i{j}" for j in range(d)],log2_int(4*D))
    net.emit({"rank":aff({"i0":1})},label="snake rank base")
    for j in range(1,d):
        net.lookup(f"i{j}",{"parity":[F(i%2) for i in range(K)]})
        net.emit({"reverse_delta":aff({"rank":-2},1-F(1,K**j))},label="snake reversal")
        net.gate([("reverse_gate","parity","reverse_delta",2)])
        net.emit({"rank":aff({f"i{j}":1,"rank":F(1,K),"reverse_gate":F(1,K)})},label="snake rank recurrence")
        net.clip(["rank"])
        net.drop("parity","reverse_delta","reverse_gate")
    net.emit({f"y{j}":aff({f"c{j}":2,f"i{j}":F(1,n0)},-(1-theta)*h) for j in range(d)},label="translated cell anchors")
    net.clip([f"y{j}" for j in range(d)])
    net.emit({f"xi{j}":aff({f"x{j}":1,f"y{j}":-1}) for j in range(d)},label="local coordinate differences")
    net.amplify([f"xi{j}" for j in range(d)],log2_int(n0*K))
    net.clip([f"xi{j}" for j in range(d)])
    return net

def read_bit(net,dest):
    base=list(net.names)
    net.emit({"__digp":aff({"code":4},-F(7,4),True),"__digm":aff({"code":4},-2,True)},keep=base,label="digit reader hinges")
    net.emit({dest:aff({"__digp":4,"__digm":-4}),
              "code":aff({"code":4,"__digp":-4,"__digm":4},-1)},keep=base,label="digit and suffix")
    net.clip(["code"])

def add_decoder_capture(net,params):
    d,K,D,ix,J,B,tau,C,a,p,AD=[params[n] for n in ("d","K","D","ix","J","B","tau","C","a","p","AD")]
    # Release all spatial intermediates before sequential decoding.
    keep=["rank","code"]+[f"v{j}" for j in range(J)]+[f"xi{j}" for j in range(d)]
    net.emit({f"capt{j}":aff() for j in range(J)},keep=keep,label="bounded decoder registers")
    sites=snake(d,K)
    for t,site in enumerate(sites):
        base=list(net.names)
        net.emit({"__mp":aff({"rank":1},-F(t,B),True),"__mm":aff({"rank":-1},F(t,B),True)},keep=base,label="capture distance hinges")
        net.emit({"distance":aff({"__mp":1,"__mm":1})},keep=base,label="capture distance")
        for _ in range(log2_int(B)):
            base=list(net.names)
            net.emit({"__capamp":aff({"distance":2},-1,True)},keep=base,label="bounded capture amplification hinge")
            net.emit({"distance":aff({"distance":2,"__capamp":-1})},keep=base,label="bounded capture amplification combine")
        net.emit({"match":aff({"distance":-1},1)},label="exact safe rank match")
        net.gate([(f"take{j}","match",f"v{j}",2) for j in range(J)])
        net.emit({f"capt{j}":aff({f"capt{j}":1,f"take{j}":1}) for j in range(J)},label="capture accumulation")
        net.clip([f"capt{j}" for j in range(J)],-2,2)
        net.drop("distance","match",*[f"take{j}" for j in range(J)])
        if t==B-1: break
        old,new=site,sites[t+1]
        axis=next(j for j in range(d) if old[j]!=new[j]); sign=1 if new[axis]>old[axis] else -1
        for j in range(p): read_bit(net,f"qbit{j}")
        net.emit({f"choose{q}":aff({f"qbit{j}":(1 if (q>>j)&1 else -1) for j in range(p)},1-sum((q>>j)&1 for j in range(p)),True) for q in range(D+1)},label="finite displacement alphabet")
        net.drop(*[f"qbit{j}" for j in range(p)])
        for j in range(J):
            for b in range(a): read_bit(net,f"cbit{b}")
            net.emit({f"cor{j}":aff({f"cbit{b}":2**b for b in range(a)},-C)},label="bounded signed correction")
            net.drop(*[f"cbit{b}" for b in range(a)])
        base=list(net.names)
        net.emit({f"pred{q}_{j}":aff({f"v{l}":T[j][l] for l in range(J)})
                  for q in range(D+1) for T in [transport(ix,axis,sign*q)] for j in range(J)},keep=base,label="integer transport alternatives")
        net.gate([(f"select{q}_{j}",f"choose{q}",f"pred{q}_{j}",2*AD) for q in range(D+1) for j in range(J)])
        net.emit({f"v{j}":aff({f"select{q}_{j}":1 for q in range(D+1)}|{f"cor{j}":tau}) for j in range(J)},label="transport and correction")
        net.clip([f"v{j}" for j in range(J)],-2,2)
        net.emit({},keep=["rank","code"]+[f"v{j}" for j in range(J)]+[f"xi{j}" for j in range(d)]+[f"capt{j}" for j in range(J)],label="release frame registers")
    net.emit({},keep=[f"capt{j}" for j in range(J)]+[f"xi{j}" for j in range(d)],label="selected jet and local coordinates")
    return net

def square(net,x,out,m):
    net.emit({"__sqg":aff({x:1}),"__sqs":aff({x:1})},label="square initialization")
    for j in range(1,m+1):
        base=list(net.names)
        net.emit({"__sqh":aff({"__sqg":1},-F(1,2),True),"__sqe":aff({"__sqg":1},-1,True)},keep=base,label="triangle hinges")
        net.emit({"__sqg":aff({"__sqg":2,"__sqh":-4,"__sqe":2}),
                  "__sqs":aff({"__sqs":1,"__sqg":-F(2,4**j),"__sqh":F(4,4**j),"__sqe":-F(2,4**j)})},keep=base,label="triangle iterate and square sum")
    net.emit({out:aff({"__sqs":1})},keep=[n for n in net.names if not n.startswith("__sq")],label="square result")

def multiply(net,x,y,out,m):
    base=list(net.names)
    mean_terms={}
    for n in (x,y): mean_terms[n]=mean_terms.get(n,F(0))+F(1,8)
    net.emit({"__mu":aff({x:F(1,4)},F(1,2)),"__mv":aff({y:F(1,4)},F(1,2)),
              "__mz":aff(mean_terms,F(1,2))},label="signed multiplication rescale")
    square(net,"__mu","__su",m); square(net,"__mv","__sv",m); square(net,"__mz","__sz",m)
    net.emit({out:aff({"__sz":32,"__su":-8,"__sv":-8,"__mu":-8,"__mv":-8},4)},keep=base,label="signed multiplication combine")

def add_polynomial(net,params):
    ix,J,k,D,tau=[params[n] for n in ("ix","J","k","D","tau")]
    eta=tau/F(4*J*(k+1)*D**k)
    m=max(1,ceil_fraction(F((12/eta).numerator.bit_length(),2)))
    # Exact sufficient precision check; avoid floating-point logarithms.
    while F(12,4**m)>eta: m+=1
    net.constant("poly",0)
    for j,beta in enumerate(ix):
        net.constant("mon",1)
        for axis,power in enumerate(beta):
            for _ in range(power):
                multiply(net,"mon",f"xi{axis}","mon_next",m)
                net.clip(["mon_next"])
                net.emit({"mon":aff({"mon_next":1})},keep=[n for n in net.names if n!="mon_next"],label="bounded monomial")
        multiply(net,f"capt{j}","mon","term",m)
        net.emit({"poly":aff({"poly":1,"term":D**sum(beta)})},keep=[n for n in net.names if n not in ("term","mon")],label="Taylor accumulation")
    net.clip(["poly"],-1,1)
    net.emit({},keep=["poly"],label="scalar output")
    return net,m

def build_approximant(f_derivative,params):
    """Sequential shifted copies followed by an explicitly compiled median."""
    d,Q=params["d"],params["Q"]
    master=Net([f"x{j}" for j in range(d)])
    originals=list(master.inputs); outputs=[]; iterations=[]
    for phase in range(Q):
        sub=build_interface(f_derivative,params,phase)
        add_decoder_capture(sub,params)
        sub,m=add_polynomial(sub,params); iterations.append(m)
        prefix=f"g{phase}_"
        keep=originals+outputs
        master.emit({prefix+n:aff({n:1}) for n in originals},keep=keep,label="initialize shifted copy")
        for layer,event in zip(sub.layers,sub.events):
            master.emit({prefix+n:aff({prefix+t:c for t,c in a.terms.items()},a.bias,a.relu) for n,a in layer.items()},keep=keep,label="shift "+str(phase)+": "+event[0])
        out=f"out{phase}"
        master.emit({out:aff({prefix+"poly":1})},keep=keep,label="store bounded shifted output")
        outputs.append(out)
    # Insertion sorting in descending order, two ReLU layers per comparator.
    for i in range(1,Q):
        for j in range(i-1,-1,-1):
            a,b=outputs[j],outputs[j+1]; base=list(master.names)
            master.emit({"__cmp":aff({a:1,b:-1},relu=True)},keep=base,label="median comparator hinge")
            master.emit({a:aff({b:1,"__cmp":1}),b:aff({a:1,"__cmp":-1})},keep=base,label="median comparator combine")
    master.emit({"poly":aff({outputs[d]:1})},keep=[],label="median output")
    return master,max(iterations)

def numeric_relu_layers(net):
    """Materialize exact sparse rows of every actual ReLU affine matrix.

    Omitted coefficients are zero but still counted by dense_parameters.
    The final row reconstructs the scalar output named poly.
    """
    previous=list(net.inputs)
    rows=[]
    first=[]
    for j in range(len(previous)):
        first.extend([{"weights":{str(j):"1"},"bias":"0"},
                      {"weights":{str(j):"-1"},"bias":"0"}])
    rows.append({"input_width":len(previous),"rows":first,"relu":True})
    for layer in net.layers:
        positions={n:j for j,n in enumerate(previous)}; encoded=[]
        for n,a in layer.items():
            w={}
            for t,c in a.terms.items():
                j=positions[t]
                w[str(2*j)]=str(c); w[str(2*j+1)]=str(-c)
            encoded.append({"weights":w,"bias":str(a.bias)})
            encoded.append({"weights":{} if a.relu else {j:str(-F(c)) for j,c in w.items()},"bias":"0" if a.relu else str(-a.bias)})
        rows.append({"input_width":2*len(previous),"rows":encoded,"relu":True})
        previous=list(layer)
    j=previous.index("poly")
    rows.append({"input_width":2*len(previous),"rows":[{"weights":{str(2*j):"1",str(2*j+1):"-1"},"bias":"0"}],"relu":False})
    return rows

def safe(x,params,phase):
    theta=F(phase,params["D"])
    for z in x:
        v=F(z)/params["h"]-theta; frac=v-v.__floor__()
        if min(frac,1-frac)<=F(1,4*params["D"]): return False
    return True

def expected_site(x,params,phase):
    K,D,A,n0,h=[params[n] for n in ("K","D","A","n0","h")]
    theta=F(phase,D)
    r=[min(n0*K,max(0,(F(z)/h-theta).__floor__()+1)) for z in x]
    c=tuple(v//K for v in r); i=tuple(v%K for v in r)
    y=tuple(anchor(v,theta,h) for v in r)
    return {"coarse":c,"fine":i,"rank":snake_rank(i,K),
            "address":sum(v*A**j for j,v in enumerate(c)),"anchor":y,
            "xi":tuple((F(z)-a)/h for z,a in zip(x,y))}

def explicit_bound(params,m=0):
    d,J,D,a,p,k,Q,N,K,B=[params[n] for n in ("d","J","D","a","p","k","Q","N","K","B")]
    r=64*(D+1)*(J+1)*(d+a+p+k+1)
    ell=1+log2_int(N*B); frame=J*a+p
    L=100*(d+1)*(ell+params["rD"]+1)+B*(2*log2_int(B)+4*frame+4*J+40)+30*J*(k+1)*(m+1)
    P=32*(r+1)**2*((d+1)*N+3*d*K+d+2)+6*r*r*(L+2)
    R=64+4*params["AD"]+2**(a+1)+p+4*J*D**k
    return {"register_bound":r,"depth_bound":L+2,"dense_parameter_bound":P,
            "width_bound":4*d*(N+K)+2*r,"weight_bound":R}
