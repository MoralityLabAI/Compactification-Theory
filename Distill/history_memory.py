"""DISTILL-13 public history corpus with referee-only exact quotient labels."""
from collections import defaultdict, deque
from dataclasses import asdict
from fractions import Fraction
import random

from .core import digest, require
from .research_program import Model, BITS
from .program_student import ACTIONS, COST_NAMES, Meter
from .program_quotient import signature
from .program_opportunity import incumbent
from .program_cancellation import Auditor
from .student_campaign import PRICE
from .opportunity_campaign import make_spec

SPLITS=("train","calibration","ordinary","shifted","scarce")
COUNTS=(24,6,12,12,12)
SEEDS=(2026112911,2026112912,2026112913,2026112914,2026112915)
TRIPLETS=48
CTX=12+2+1+5+1+3
EVENT=len(ACTIONS)+3


def specs(split,n):
    seed=SEEDS[SPLITS.index(split)]
    for i in range(n):
        family=(("shared_control","shifted_control","shifted_control")[i%3]
                if split in ("train","calibration") else
                "shared_control" if split=="ordinary" else "shifted_control")
        scarce=(i%3==2 if split in ("train","calibration") else split=="scarce")
        yield make_spec(seed,i,family,scarce)


def context(spec):
    return [w/8 for k in BITS for w in spec["prior_weights"][k]]+list(spec["offsets"])+[
        spec["budget"]/14]+[spec["costs"][a]/6 for a in COST_NAMES]+[
        float(spec["family"]=="shifted_control"),spec["correct_reward"]/20,
        spec["wrong_reward"]/40,spec["certificate_reward"]/30]


def histories(m):
    """One shortest public witness history per literal reachable state."""
    s=m.initial()
    seen={s:()}
    todo=deque([s])
    while todo:
        s=todo.popleft()
        if s.stage==4:
            continue
        for a in m.legal(s):
            for bit,t in m.branches(s,a):
                if t not in seen:
                    seen[t]=seen[s]+((a,bit),)
                    todo.append(t)
    return {s:h for s,h in seen.items() if s.stage<4}


def replay(m,history):
    s=m.initial()
    for a,bit in history:
        matches=[t for b,t in m.branches(s,a) if b==bit]
        require(len(matches)==1,"invalid public history")
        s=matches[0]
    return s


def root_data(spec,limit=TRIPLETS):
    m=Model(spec)
    paths=histories(m)
    groups=defaultdict(list)
    prefixes=defaultdict(list)
    stages=defaultdict(list)
    keys={}
    for s in paths:
        key,g,N=signature(m,s,"marginal",Meter(100000))
        keys[s]=key
        groups[key].append(s)
        prefixes[key[:-1]].append(s)
        stages[s.stage].append(s)
    rng=random.Random(int(digest(spec)[:16],16))
    ref=Auditor(m)
    memo={}
    def outcome(s):
        if s not in memo:
            a=incumbent(m,s,Meter(100000))
            v,r=ref.tail(s)
            M=m.mass(s.mask)
            memo[s]=(a,Fraction(v,PRICE*M),Fraction(r,M))
        return memo[s]
    anchors=[s for s in paths if len(groups[keys[s]])>1]
    rng.shuffle(anchors)
    selected=[]
    for s in anchors:
        key=keys[s]
        # Prefer a different compatible-world mask over a flags-only pair.
        pos=[t for t in groups[key] if t!=s]
        diverse=[t for t in pos if t.mask!=s.mask]
        p=rng.choice(diverse or pos)
        target=outcome(s)
        require(outcome(p)==target,"quotient label mismatch")
        # A different key alone does not prove a different continuation value.
        # Use negatives with different exact incumbent action/value/risk.
        n=None
        hard=False
        for pool,same_prefix in ((prefixes[key[:-1]],True),(stages[s.stage],False)):
            candidates=[t for t in pool if keys[t]!=key]
            rng.shuffle(candidates)
            for t in candidates[:64]:
                if outcome(t)!=target:
                    n=t
                    hard=same_prefix
                    break
            if n is not None:
                break
        if n is None:
            continue
        selected.append((s,p,n,hard))
        if len(selected)>=limit:
            break
    require(selected,"no valid quotient triplets")
    public={}
    labels={}
    triplets=[]
    for s,p,n,hard in selected:
        ids=[]
        for t in (s,p,n):
            h=paths[t]
            require(replay(m,h)==t,"history replay mismatch")
            rid=digest({"spec":spec,"history":h})
            ids.append(rid)
            if rid in public:
                continue
            a,v,r=outcome(t)
            key=keys[t]
            bins=key[-1]
            public[rid]={"history":[[a,b] for a,b in h]}
            labels[rid]={"quotient":key,"stage":t.stage,"belief":[str(Fraction(x,sum(bins))) for x in bins],
                         "action":a,"value":str(v),"risk":str(r)}
        triplets.append({"anchor":ids[0],"positive":ids[1],"negative":ids[2],
                         "same_prefix_negative":hard,
                         "different_mask_positive":s.mask!=p.mask})
    ref.close()
    return {"spec":spec,"records":public},{"records":labels,"triplets":triplets,
             "literal_states":len(paths),"quotient_classes":len(groups)}


def verify(root,ref):
    m=Model(root["spec"])
    audit=Auditor(m)
    require(set(root["records"])==set(ref["records"]),"public/referee mismatch")
    for rid,record in root["records"].items():
        require(set(record)=={"history"},"private fields in public record")
        require(digest({"spec":root["spec"],"history":record["history"]})==rid,"record digest")
        s=replay(m,record["history"])
        k,_,N=signature(m,s,"marginal",Meter(100000))
        y=ref["records"][rid]
        require(digest(k)==digest(y["quotient"]),"quotient mismatch")
        require(y["stage"]==s.stage,"stage mismatch")
        require([Fraction(x) for x in y["belief"]]==[Fraction(x,N) for x in k[-1]],"belief mismatch")
        v,r=audit.tail(s)
        M=m.mass(s.mask)
        require(Fraction(y["value"])==Fraction(v,PRICE*M) and Fraction(y["risk"])==Fraction(r,M),"continuation mismatch")
        require(y["action"]==incumbent(m,s,Meter(100000)),"action mismatch")
    for t in ref["triplets"]:
        a,p,n=(ref["records"][t[k]] for k in ("anchor","positive","negative"))
        require(a["quotient"]==p["quotient"] and a["quotient"]!=n["quotient"],"pair key")
        require(a["stage"]==n["stage"],"negative stage")
        require(tuple(a[k] for k in ("action","value","risk"))!=tuple(n[k] for k in ("action","value","risk")),"negative witness")
        require(tuple(a[k] for k in ("action","value","risk"))==tuple(p[k] for k in ("action","value","risk")),"positive witness")
    audit.close()
