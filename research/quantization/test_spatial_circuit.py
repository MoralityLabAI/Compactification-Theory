from fractions import Fraction as F
from itertools import product
from math import factorial
from pathlib import Path
import hashlib
import gzip
import json
import random
import unittest
from spatial_circuit import *

ROOT=Path(__file__).resolve().parent
COUNTS={}
CERTIFICATES=[]
def checked(name,condition):
    assert condition,name
    COUNTS[name]=COUNTS.get(name,0)+1

def quadratic_derivative(beta,x):
    d=len(x)
    if sum(beta)==0: return sum((z*z for z in x),F(0))/(2*d)
    if sum(beta)==1:
        j=beta.index(1); return x[j]/d
    if sum(beta)==2 and 2 in beta: return F(1,d)
    return F(0)

def cubic_derivative(beta,x):
    assert len(x)==1
    n=beta[0]
    return F(factorial(3),factorial(3-n))*x[0]**(3-n)/6 if n<=3 else F(0)

def mixed_derivative(beta,x):
    assert len(x)==2
    if beta==(0,0): return x[0]*x[1]
    if beta==(1,0): return x[1]
    if beta==(0,1): return x[0]
    if beta==(1,1): return F(1)
    return F(0)

class SpatialTests(unittest.TestCase):
    def test_snake_hamiltonian_order(self):
        for d in range(1,5):
            for K in (1,2,4):
                sites=snake(d,K)
                checked("snake_site_count",len(set(sites))==K**d)
                for i,p in enumerate(sites): checked("snake_rank",snake_rank(p,K)==i)
                for a,b in zip(sites,sites[1:]): checked("snake_neighbor",sum(abs(x-y) for x,y in zip(a,b))==1)

    def test_all_dyadic_gate_breakpoints(self):
        for C in (1,2,17):
            n=Net(["bit","z"]); n.gate([("out","bit","z",C)])
            for b in (0,1):
                for z in (-C,-F(C,2),0,F(C,2),C):
                    direct=n.evaluate((b,z))["out"]
                    checked("exact_binary_gate",direct==b*z)
                    checked("signed_pair_compilation",direct==n.evaluate_compiled((b,z))["out"])

    def test_endpoint_microsteps(self):
        for d in range(1,4):
            p=parameters(d,2,2,4)
            for phase in range(p["Q"]):
                theta=F(phase,p["D"])
                for r in range(p["A"]*p["K"]-1):
                    step=(anchor(r+1,theta,p["h"])-anchor(r,theta,p["h"]))/p["mu"]
                    checked("clipped_integer_microsteps",step.denominator==1 and 0<=step<=p["D"])

    def test_spatial_interface(self):
        rng=random.Random(4321515)
        configurations=[(1,1,1),(1,2,2),(1,4,4),(2,1,2),(2,2,2),(3,1,2)]
        for d,n0,K in configurations:
            p=parameters(d,2,n0,K)
            points=[tuple(F(v) for v in x) for x in product((0,1),repeat=d)]
            denominator=16*p["D"]*n0*K
            points += [tuple(F(rng.randrange(denominator+1),denominator) for _ in range(d)) for _ in range(18)]
            # Every coordinate crossing and both margins are covered in 1D.
            if d==1:
                for phase in range(p["Q"]):
                    theta=F(phase,p["D"])
                    for j in range(n0*K):
                        boundary=(j+theta)*p["h"]
                        for shift in (-F(1,4*p["D"]),0,F(1,4*p["D"])):
                            z=boundary+shift*p["h"]
                            if 0<=z<=1: points.append((z,))
            good_count={x:0 for x in points}
            for phase in range(p["Q"]):
                n=build_interface(quadratic_derivative,p,phase)
                c=n.certificate(); bound=explicit_bound(p)
                checked("interface_dense_parameter_bound",c["dense_parameters"]<=bound["dense_parameter_bound"])
                checked("interface_weight_bound",F(c["maximum_weight_or_bias"])<=bound["weight_bound"])
                for x in points:
                    out=n.evaluate(x)
                    checked("global_local_coordinate_clipping",all(0<=out[f"xi{j}"]<=1 for j in range(d)))
                    if not safe(x,p,phase): continue
                    good_count[x]+=1; e=expected_site(x,p,phase)
                    checked("safe_coarse_address",out["address"]==F(e["address"],p["N"]))
                    checked("safe_fine_address",all(out[f"i{j}"]==F(e["fine"][j],K) for j in range(d)))
                    checked("safe_snake_rank",out["rank"]==F(e["rank"],p["B"]))
                    checked("safe_local_coordinates",all(out[f"xi{j}"]==e["xi"][j] for j in range(d)))
                    code,v,_,_,_=encode_block(quadratic_derivative,e["coarse"],F(phase,p["D"]),p)
                    checked("safe_packed_code",out["code"]==code)
                    checked("safe_initial_jets",all(out[f"v{j}"]==v[j] for j in range(p["J"])))
                for x in points[:2]: checked("interface_pair_compilation",n.evaluate(x)==n.evaluate_compiled(x))
            for x,count in good_count.items(): checked("all_point_majority_coverage",count>=d+1)

    def test_decoder_capture_and_polynomial(self):
        for d,s,n0,K,derivative in [(1,F(2),1,1,quadratic_derivative),(1,F(2),2,2,quadratic_derivative),(1,F(5,2),1,2,cubic_derivative),(2,F(2),1,2,quadratic_derivative),(2,F(5,2),1,2,mixed_derivative)]:
            p=parameters(d,s,n0,K)
            points=list(product((F(0),F(1,4),F(3,8),F(1,2),F(3,4),F(1)),repeat=d))
            if d>1: points=points[::4]+[(F(0),F(0)),(F(1),F(1))]
            outputs={x:[] for x in points}
            for phase in range(p["Q"]):
                n=build_interface(derivative,p,phase)
                add_decoder_capture(n,p)
                # This snapshot proves capture using an evaluator separate from
                # the polynomial circuit added below.
                capture_results={x:n.evaluate(x) for x in points}
                for x,out in capture_results.items():
                    checked("all_branch_bounded_capture",all(abs(out[f"capt{j}"])<=2 for j in range(p["J"])))
                    if safe(x,p,phase):
                        e=expected_site(x,p,phase)
                        _,_,states,_,_=encode_block(derivative,e["coarse"],F(phase,p["D"]),p)
                        target=[p["tau"]*v for v in states[e["rank"]]]
                        checked("exact_selected_jet_capture",all(out[f"capt{j}"]==target[j] for j in range(p["J"])))
                n,m=add_polynomial(n,p)
                cert=n.certificate("poly"); bound=explicit_bound(p,m)
                for name,key in (("dense_parameters","dense_parameter_bound"),("maximum_width","width_bound"),("depth_with_scalar_output","depth_bound")):
                    checked("whole_circuit_"+name,cert[name]<=bound[key])
                checked("whole_circuit_weight_bound",F(cert["maximum_weight_or_bias"])<=bound["weight_bound"])
                precision=max(2*(p["payload"]+1),ceil_fraction(p["s"]*(p["q"]+p["rD"])),2*m,p["q"]+p["rD"]+3,log2_int(p["N"]),(d-1)*log2_int(K),4)
                checked("whole_circuit_dyadic_precision",cert["fractional_bits"]<=precision)
                CERTIFICATES.append({"d":d,"s":str(s),"n0":n0,"K":K,"phase":phase,"payload_bits":p["payload"],"polynomial_iterations":m,"certificate":cert,"explicit_bound":bound})
                for x in points:
                    y=n.evaluate(x)["poly"]
                    outputs[x].append(y)
                    if safe(x,p,phase):
                        err=abs(y-derivative((0,)*d,x))
                        C=max(F(1),F(d**p["k"],factorial(p["k"])))+F(p["J"],2)+F(1,2)
                        checked("safe_uniform_error",(err/C)**s.denominator<=p["h"]**s.numerator)
            for x,ys in outputs.items():
                y=sorted(ys)[d]; err=abs(y-derivative((0,)*d,x))
                C=max(F(1),F(d**p["k"],factorial(p["k"])))+F(p["J"],2)+F(1,2)
                checked("median_uniform_error",(err/C)**s.denominator<=p["h"]**s.numerator)

    def test_materialized_full_network(self):
        p=parameters(1,2,2,2)
        n,m=build_approximant(quadratic_derivative,p)
        layers=numeric_relu_layers(n)
        dense=sum((l["input_width"]+1)*len(l["rows"]) for l in layers)
        checked("materialized_dense_slot_count",dense==n.certificate("poly")["dense_parameters"])
        points=[F(0),F(1,16),F(1,4),F(7,16),F(1,2),F(15,16),F(1)]
        for x in points:
            values=[x]
            for l in layers:
                values=[F(r["bias"])+sum((F(c)*values[int(j)] for j,c in r["weights"].items()),F(0)) for r in l["rows"]]
                if l["relu"]: values=[max(F(0),v) for v in values]
            checked("materialized_full_network_agreement",values[0]==n.evaluate((x,))["poly"])
            checked("materialized_full_uniform_error",abs(values[0]-x*x/2)<=F(5,2)*p["h"]**2)
        cert=n.certificate("poly"); bound=explicit_bound(p,m)
        checked("full_median_depth",cert["depth_with_scalar_output"]<=p["Q"]*(bound["depth_bound"]+2)+p["Q"]*(p["Q"]-1)+3)
        checked("full_median_dense_parameters",cert["dense_parameters"]<=p["Q"]*bound["dense_parameter_bound"]+12*bound["register_bound"]**2*(p["Q"]**2+p["Q"]+1))
        example={"parameters":{k:str(v) for k,v in p.items() if k!="ix"},"certificate":cert,"layers":layers,
                 "representation":"Exact rational affine matrices; every hidden layer uses ReLU; omitted entries are zero"}
        with gzip.open(ROOT/"spatial_relu_example.json.gz","wt",encoding="utf-8") as f: json.dump(example,f,separators=(",",":"))
        CERTIFICATES.append({"scope":"fully materialized shifted-median network","certificate":cert,"points":len(points)})

if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SpatialTests))
    report={"status":"passed" if result.wasSuccessful() else "failed","counts":COUNTS,
            "assertions":sum(COUNTS.values()),"certificates":CERTIFICATES,
            "source_sha256":hashlib.sha256((ROOT/"spatial_circuit.py").read_bytes()).hexdigest(),
            "scope":"Finite regression checks of explicit circuits; general claims require the accompanying proof"}
    (ROOT/"spatial_circuit_results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in report.items() if k!="certificates"},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
