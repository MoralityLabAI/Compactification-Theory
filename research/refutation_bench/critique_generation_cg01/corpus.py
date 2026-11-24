"""RB-CG-01: raw generated Lean sources, an independent parser, and critique certificates."""
import argparse
import hashlib
import itertools
import json
import random
import re
import subprocess
from pathlib import Path

SPLITS = {
    "train": (256, 101701, ("reverse", "drop", "strengthen"), (1, 4)),
    "cal": (32, 101702, ("reverse", "drop", "strengthen"), (1, 4)),
    "test_depth": (48, 101703, ("reverse", "drop", "strengthen"), (7, 11)),
    "test_mechanism": (48, 101704, ("vacuity", "unrelated"), (1, 4)),
}
VALUES = tuple(itertools.product((False, True), repeat=3))
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
VARS = ("x", "y", "z")
NAME = r"n[0-9a-f]{8}"
TYPE = r"∀ \(x y z : Bool\), (" + NAME + r") x y z = true"
MAX_LINES = 64
MAX_TOKENS = 64

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def randexpr(rng, depth):
    if depth == 0 or rng.random() < .22:
        return rng.choice(VARS)
    if rng.random() < .22:
        return ("not", randexpr(rng, depth-1))
    return (rng.choice(("and", "or")), randexpr(rng, depth-1), randexpr(rng, depth-1))

def calc(expr, env, defs):
    if isinstance(expr, str):
        return env[expr]
    op, *args = expr
    if op == "call":
        return calc(defs[args[0]], env, defs)
    if op == "not":
        return not calc(args[0], env, defs)
    a, b = (calc(x, env, defs) for x in args)
    return a and b if op == "and" else a or b

def table(expr, defs):
    return tuple(calc(expr, dict(zip(VARS, v)), defs) for v in VALUES)

def render(expr):
    if isinstance(expr, str):
        return expr
    op, *args = expr
    if op == "call":
        return "(" + args[0] + " x y z)"
    if op == "not":
        return "(!" + render(args[0]) + ")"
    return "(" + render(args[0]) + (" && " if op == "and" else " || ") + render(args[1]) + ")"

def ty(name):
    return f"∀ (x y z : Bool), {name} x y z = true"

def generate(split, idx, rng, mechanism, depths):
    names = set()
    def name():
        while True:
            n = f"n{rng.getrandbits(32):08x}"
            if n not in names:
                names.add(n)
                return n
    def call(n):
        return ("call", n)
    while True:
        a, e = randexpr(rng, 2), randexpr(rng, 2)
        aa, ee = table(a, {}), table(e, {})
        if (any(x and not y for x,y in zip(aa,ee)) and
            any(y and not x for x,y in zip(aa,ee)) and
            any(not x and not y for x,y in zip(aa,ee))):
            break
    an, en, bn, pn, bad, bridge_target, independent = [name() for _ in range(7)]
    defs = {an:a, en:e, bn:("or",call(an),call(en))}
    antecedent = call(an)
    if mechanism == "vacuity":
        antecedent = ("and",call(an),("not",call(an)))
    defs[pn] = ("or",("not",antecedent),call(bn))
    if mechanism == "reverse":
        defs[bad] = ("or",("not",call(bn)),call(an))
    elif mechanism in ("drop","vacuity"):
        defs[bad] = call(bn)
    elif mechanism == "strengthen":
        defs[bad] = ("or",("not",call(an)),("and",call(bn),call(en)))
    else:
        defs[bad] = call(en)
    taut = ("or","x",("not","x"))
    defs[bridge_target] = ("and",call(pn),taut)
    r = randexpr(rng,2)
    defs[independent] = ("or",r,("not",r))
    assert all(table(defs[pn],defs)) and all(table(defs[bridge_target],defs))
    assert all(table(defs[independent],defs)) and not all(table(defs[bad],defs))
    # Topologically shuffle independent definition declarations.
    pending = dict(defs)
    lines = []
    emitted = set()
    def references(expr):
        if isinstance(expr,str):
            return set()
        if expr[0]=="call":
            return {expr[1]}
        return set().union(*(references(x) for x in expr[1:]))
    while pending:
        ready = [n for n,x in pending.items() if references(x)<=emitted]
        n = rng.choice(ready)
        lines.append(f"def {n} (x y z : Bool) : Bool := {render(pending.pop(n))}")
        emitted.add(n)
    clean0, taint0, dead, decoy, bridge = [name() for _ in range(5)]
    proofbody = "by intro x y z; cases x <;> cases y <;> cases z <;> decide"
    entries = [(clean0, f"theorem {clean0} : {ty(pn)} := {proofbody}", None),
               (taint0, f"axiom {taint0} : {ty(pn)}", None),
               (dead, f"axiom {dead} : {ty(pn)}", None),
               (decoy, f"theorem {decoy} : {ty(pn)} := {proofbody}", None),
               (bridge, f"theorem {bridge} : ({ty(pn)}) → ({ty(bridge_target)}) := by intro h x y z; change ({pn} x y z && (x || !x)) = true; rw [h x y z] <;> cases x <;> decide", None)]
    roots = []
    chains = {}
    for start in (clean0,taint0,dead,decoy):
        prev = start
        chain = [start]
        depth = rng.randint(*depths) if start in (clean0,taint0) else rng.randint(1,3)
        for _ in range(depth):
            n = name()
            entries.append((n, f"theorem {n} : {ty(pn)} := {prev}", prev))
            prev = n
            chain.append(n)
        chains[prev] = list(reversed(chain))
        roots.append(prev)
    loc = {}
    while entries:
        choices = [i for i,(_,_,dep) in enumerate(entries) if dep is None or dep in loc]
        n,line,_ = entries.pop(rng.choice(choices))
        lines.append(line)
        loc[n] = len(lines)
    assert len(lines)<=MAX_LINES
    raw = "\n".join(lines)+"\n"
    ns = f"CG{SPLITS[split][1]}_{idx}"
    arms = []
    falsifying = [i for i,x in enumerate(table(defs[bad],defs)) if not x]
    for root, ax in zip(roots[:2],("CLEAN","AX")):
        path = [f"L{loc[n]}" for n in chains[root]]
        for target, sem in ((pn,"MATCH"),(bad,"COUNTER"),(bridge_target,"BRIDGE"),(independent,"UNLINKED")):
            tokens = [ax,*path,"SEP",sem]
            if sem=="BRIDGE":
                tokens += [f"L{loc[bridge]}"]
            if sem=="COUNTER":
                tokens += [str(int(x)) for x in VALUES[falsifying[0]]]
            tokens += ["EOS"]
            arms.append({"id":sha(ns+root+target)[:24],"root":root,"target":target,"gold":tokens})
    rng.shuffle(arms)
    native = ["namespace "+ns,raw]
    expected = {}
    def audit(n, axioms=()):
        native.append("#print axioms "+n)
        expected[ns+"."+n] = {ns+"."+a for a in axioms}
    audit(roots[0]); audit(roots[1],(taint0,)); audit(bridge)
    native.append(f"theorem bridgeCert : {ty(bridge_target)} := {bridge} {roots[0]}")
    audit("bridgeCert")
    native.append(f"theorem independentCert : {ty(independent)} := {proofbody}")
    audit("independentCert")
    for i in falsifying:
        args = " ".join(str(v).lower() for v in VALUES[i])
        n = "witness"+str(i)
        native.append(f"theorem {n} : {bad} {args} = false := by decide")
        audit(n)
    native.append("end "+ns)
    return {"id":ns,"split":split,"mechanism":mechanism,"source":raw,"arms":arms,
            "native":"\n".join(native),"expected":{k:sorted(v) for k,v in expected.items()}}

def families():
    for split,(count,seed,mechanisms,depths) in SPLITS.items():
        rng = random.Random(seed)
        for i in range(count):
            yield generate(split,i,rng,mechanisms[i%len(mechanisms)],depths)

# Independent parser. This is a closed raw-source grammar, not a Lean parser.
def parse_expr(text):
    tokens = re.findall(NAME+r"|[xyz]|\|\||&&|!|\(|\)",text)
    assert "".join(tokens)==re.sub(r"\s+","",text),text
    pos=0
    def parse():
        nonlocal pos
        tok=tokens[pos]; pos+=1
        if tok in VARS:
            return tok
        assert tok=="(",(tok,text)
        if tokens[pos]=="!":
            pos+=1
            ans=("not",parse())
        elif re.fullmatch(NAME,tokens[pos]):
            n=tokens[pos]; pos+=1
            assert tokens[pos:pos+3]==list(VARS)
            pos+=3
            ans=("call",n)
        else:
            a=parse()
            op=tokens[pos]; pos+=1
            assert op in ("&&","||")
            ans=("and" if op=="&&" else "or",a,parse())
        assert tokens[pos]==")"
        pos+=1
        return ans
    result=parse()
    assert pos==len(tokens)
    return result

class Parsed:
    def __init__(self,raw):
        self.defs={}; self.proofs={}; self.bridges={}; self.lines=raw.splitlines()
        for i,line in enumerate(self.lines,1):
            m=re.fullmatch(r"def ("+NAME+r") \(x y z : Bool\) : Bool := (.+)",line)
            if m:
                self.defs[m[1]]=parse_expr(m[2]); continue
            m=re.fullmatch(r"theorem ("+NAME+r") : \("+TYPE+r"\) → \("+TYPE+r"\) := (.+)",line)
            if m:
                self.bridges[(m[2],m[3])]=i; continue
            m=re.fullmatch(r"(axiom|theorem) ("+NAME+r") : "+TYPE+r"(?: := (.+))?",line)
            assert m,line
            kind,n,t,body=m.groups()
            assert t in self.defs
            if kind=="axiom":
                assert body is None
                self.proofs[n]=(i,t,"AX",None)
            elif re.fullmatch(NAME,body):
                assert body in self.proofs and self.proofs[body][1]==t
                self.proofs[n]=(i,t,"alias",body)
            else:
                assert body.startswith("by intro")
                self.proofs[n]=(i,t,"CLEAN",None)
    def path(self,root):
        result=[]
        n=root
        while True:
            i,t,k,dep=self.proofs[n]
            result.append("L"+str(i))
            if dep is None:
                return k,result,t
            n=dep
    def valuations(self,target):
        # Direct evaluation of parsed Bool expression, separate from generator's calc.
        def evaluate(expr, bits):
            if isinstance(expr,str):
                return bits[VARS.index(expr)]
            op=expr[0]
            if op=="call":
                return evaluate(self.defs[expr[1]],bits)
            if op=="not":
                return not evaluate(expr[1],bits)
            if op=="and":
                return evaluate(expr[1],bits) and evaluate(expr[2],bits)
            assert op=="or"
            return evaluate(expr[1],bits) or evaluate(expr[2],bits)
        return [evaluate(self.defs[target],bits) for bits in VALUES]
    def symbolic(self,root,target):
        k,path,t=self.path(root)
        vals=self.valuations(target)
        if not all(vals):
            sem=["COUNTER",*[str(int(x)) for x in VALUES[vals.index(False)]]]
        elif target==t:
            sem=["MATCH"]
        elif (t,target) in self.bridges:
            sem=["BRIDGE","L"+str(self.bridges[t,target])]
        else:
            sem=["UNLINKED"]
        return [k,*path,"SEP",*sem,"EOS"]
    def verify(self,root,target,pred):
        res={"syntax":False,"axiom":False,"semantic":False,"complete":False,
             "false_accusation":False,"support":False,"unlinked":False}
        if not isinstance(pred,list) or not pred or pred[-1]!="EOS" or pred.count("SEP")!=1 or pred.count("EOS")!=1:
            return res
        j=pred.index("SEP")
        left,right=pred[:j],pred[j+1:-1]
        if (len(left)<2 or left[0] not in ("CLEAN","AX") or
            not all(re.fullmatch(r"L[1-9]\d*",x) and 1<=int(x[1:])<=len(self.lines) for x in left[1:]) or not right):
            return res
        k,path,t=self.path(root)
        supported=k=="CLEAN" and all(self.valuations(target)) and (target==t or (t,target) in self.bridges)
        res["false_accusation"]=supported and (left[0]=="AX" or right[0]=="COUNTER")
        sem=right[0]
        if sem in ("MATCH","UNLINKED"):
            if len(right)!=1:
                return res
        elif sem=="BRIDGE":
            if len(right)!=2 or not re.fullmatch(r"L[1-9]\d*",right[1]):
                return res
        elif sem=="COUNTER":
            if len(right)!=4 or any(x not in ("0","1") for x in right[1:]):
                return res
        else:
            return res
        res["syntax"]=True
        res["axiom"]=left==[k,*path]
        vals=self.valuations(target)
        if sem=="MATCH":
            res["semantic"]=target==t and all(vals)
        elif sem=="BRIDGE":
            res["semantic"]=(t,target) in self.bridges and right[1]=="L"+str(self.bridges[t,target]) and all(vals)
        elif sem=="COUNTER":
            res["semantic"]=not vals[VALUES.index(tuple(x=="1" for x in right[1:]))]
        else:
            res["semantic"]=all(vals) and target!=t and (t,target) not in self.bridges
        res["complete"]=res["axiom"] and res["semantic"]
        res["support"]=res["complete"] and supported
        res["unlinked"]=res["complete"] and sem=="UNLINKED"
        return res

def render_critique(raw,root,target,pred):
    # Renderer has no gold. Failed certificates are returned as proposals, not claims.
    p=Parsed(raw)
    check=p.verify(root,target,pred)
    if not check["complete"]:
        return "Unverified critique proposal: "+" ".join(pred)
    j=pred.index("SEP"); ax=pred[0]; path=" -> ".join(pred[1:j]); sem=pred[j+1]
    parts=[]
    if ax=="AX":
        parts.append(f"The selected theorem {root} depends on a forbidden axiom along {path}. This is an assumption of the submission; it does not establish inconsistency or falsity.")
    else:
        parts.append(f"The selected theorem {root} has an axiom-free generated proof chain {path} under the declared policy.")
    if sem=="COUNTER":
        bits=pred[j+2:j+5]
        assignment=", ".join(f"{v}={'true' if b=='1' else 'false'}" for v,b in zip(VARS,bits))
        parts.append(f"The advertised universal Boolean proposition {target} is false at {assignment}. This witness concerns the advertised toy statement.")
    elif sem=="MATCH":
        parts.append(f"The supplied theorem type matches the advertised proposition {target}. Its stated scope is preserved.")
    elif sem=="BRIDGE":
        parts.append(f"The supplied bridge at {pred[j+2]} connects the selected theorem type to {target}.")
    else:
        parts.append(f"The target {target} is true in the finite domain, but the artifact supplies no named correspondence under this contract. Its advertised derivation remains unestablished.")
    return " ".join(parts)

def validate(fs):
    assert len(fs)==384 and sum(len(f["arms"]) for f in fs)==3072
    assert len({sha(f["source"]) for f in fs})==len(fs)
    assert len({a["id"] for f in fs for a in f["arms"]})==3072
    for f in fs:
        p=Parsed(f["source"])
        assert not re.search(r"\b(sorry|admit|native_decide|sorryAx)\b",f["native"])
        for a in f["arms"]:
            s=p.symbolic(a["root"],a["target"])
            assert s==a["gold"],(f["id"],s,a["gold"])
            assert p.verify(a["root"],a["target"],s)["complete"]
            assert not p.verify(a["root"],a["target"],s[:-1])["complete"]
            # Independent falsification of a wrong chain, even on supported controls.
            wrong=list(s); wrong[0]="AX" if s[0]=="CLEAN" else "CLEAN"
            assert not p.verify(a["root"],a["target"],wrong)["complete"]
            if "COUNTER" in s:
                vals=p.valuations(a["target"])
                for i,bits in enumerate(VALUES):
                    other=s[:s.index("COUNTER")+1]+[str(int(x)) for x in bits]+["EOS"]
                    assert p.verify(a["root"],a["target"],other)["complete"]== (not vals[i])

def native(fs,out,lean):
    for split in SPLITS:
        sub=[f for f in fs if f["split"]==split]
        total=0
        for off in range(0,len(sub),16):
            group=sub[off:off+16]
            expected={k:set(v) for f in group for k,v in f["expected"].items()}
            path=out/f"{split}-{off}.lean"
            path.write_text("import Lean\n\n"+"\n".join(f["native"] for f in group))
            proc=subprocess.run([lean,str(path)],capture_output=True,text=True,timeout=240)
            log=proc.stdout+proc.stderr
            path.with_suffix(".log").write_text(log)
            assert proc.returncode==0 and not re.search(r"error:|sorryAx|uses 'sorry'",log),log[-6000:]
            found={}
            for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",log):
                assert m[1] not in found
                found[m[1]]={x.strip() for x in (m[3] or "").split(",") if x.strip()}
            assert set(found)==set(expected),(set(expected)-set(found),set(found)-set(expected))
            for n,ax in expected.items():
                assert found[n]-ALLOWED==ax,(n,found[n],ax)
            total+=len(expected)
        print("NATIVE",split,len(sub),total,flush=True)

def export(fs,out):
    for split in SPLITS:
        rows=[]; targets=[]
        for f in fs:
            if f["split"]!=split:
                continue
            for a in f["arms"]:
                rows.append({"id":a["id"],"source":f["source"],"root":a["root"],"target":a["target"]})
                targets.append({"id":a["id"],"certificate":a["gold"],"family":f["id"]})
        rows.sort(key=lambda r:r["id"]);targets.sort(key=lambda r:r["id"])
        (out/f"{split}-inputs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        (out/f"{split}-gold.jsonl").write_text("".join(json.dumps(r)+"\n" for r in targets))

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--lean",default="lean");p.add_argument("--skip-native",action="store_true")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    fs=list(families());validate(fs);export(fs,args.out)
    print("CORPUS",len(fs),sum(len(f["arms"]) for f in fs),flush=True)
    if not args.skip_native:
        native(fs,args.out,args.lean)

if __name__=="__main__":
    main()
