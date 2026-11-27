"""DISTILL-11 policy-relative congruence, integer scaling and replay."""
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from Distill.research_program import Model,new_spec
from Distill.program_student import Meter,Exhausted
from Distill.program_opportunity import incumbent
from Distill.program_quotient import (MODES,Frontier,signature,local_witness,
    audit_states,closure,diagnostic,compare)
from Distill.program_cancellation import Auditor as ExactAuditor
from Distill.quotient_policy import Auditor,certificate,CONFIGS
from Distill.quotient_campaign import campaign

def example(family="shared_control"):
    spec=new_spec(2026112879,0,family,"frontier")
    spec["budget"]=1
    spec["costs"]["control"]=1
    spec["costs"]["law_audit"]=2
    spec["prior_weights"]["nuisance"]=[8,1]
    spec["prior_weights"]["local_c"]=[1,1]
    return Model(spec)

def at(m,stage):
    s=m.initial()
    for _ in range(stage):
        _,s=m.branches(s,"defer")[0]
    return s

class QuotientTests(unittest.TestCase):
    def test_normalized_mass_and_dead_budget(self):
        m=example()
        s=at(m,3)
        a,b=[t for _,t in m.branches(s,"control")]
        ks,gs,ns=signature(m,s,"marginal",Meter(100000))
        ka,ga,na=signature(m,a,"marginal",Meter(100000))
        kb,gb,nb=signature(m,b,"marginal",Meter(100000))
        self.assertEqual(ks,ka)
        self.assertEqual(ka,kb)
        self.assertEqual(ns,na)
        self.assertEqual(na,nb)
        self.assertEqual(gs,ga+gb)
        self.assertNotEqual(ga,gb)
        self.assertEqual(local_witness(m,s,"marginal"),local_witness(m,a,"marginal"))
        self.assertEqual(local_witness(m,a,"marginal"),local_witness(m,b,"marginal"))
        self.assertNotEqual(signature(m,s,"literal",Meter(100000))[0],
                            signature(m,a,"literal",Meter(100000))[0])
        self.assertEqual(a.budget,0)
        self.assertEqual(s.budget,1)
        ref=ExactAuditor(m)
        self.assertEqual(tuple(Fraction(v,m.mass(s.mask)) for v in ref.tail(s)),
                         tuple(Fraction(v,m.mass(a.mask)) for v in ref.tail(a)))
        ref.close()

    def test_affordable_certificate_saturation(self):
        m=Model(new_spec(2026112879,1))
        s=at(m,3)
        d=m.cost("direct_audit")
        x=replace(s,budget=d)
        y=replace(s,budget=d+1)
        z=replace(s,budget=d-1)
        self.assertEqual(local_witness(m,x,"marginal"),local_witness(m,y,"marginal"))
        self.assertNotEqual(signature(m,x,"marginal",Meter(100000))[0],
                            signature(m,z,"marginal",Meter(100000))[0])
        _,c=m.branches(x,"direct_audit")[0]
        self.assertEqual(local_witness(m,c,"marginal"),
                         local_witness(m,replace(c,budget=1),"marginal"))

    def test_nonliteral_reconvergence(self):
        m=example()
        s=at(m,2)
        base=incumbent(m,s,Meter(100000))
        self.assertTrue(m.terminal(base))
        ref=ExactAuditor(m)
        av,ar=ref.explicit(s,"control")
        bv,br=ref.explicit(s,base)
        exact=[av-bv,av-bv,ar-br,ar-br]
        stats={}
        for mode in MODES:
            f=Frontier(m,Meter(200000),mode)
            f.init(s,"control",base)
            while f.entries:
                f.expand()
            self.assertEqual(f.total,exact)
            stats[mode]=(f.expanded,f.quotient_merges,f.cancelled_pairs)
        self.assertGreater(stats["marginal"][1],0)
        self.assertGreater(stats["marginal"][2],0)
        self.assertLess(stats["marginal"][0],stats["literal"][0])
        ref.close()

    def test_bounds_interrupt_and_congruence(self):
        for fam in ("shared_control","shifted_control"):
            spec=new_spec(2026112879,2,fam,"frontier")
            spec["budget"]=5
            m=Model(spec)
            states=closure(m,[m.initial()],all_actions=True)
            self.assertGreater(len(states),100)
            audited=audit_states(m,{s for s in states if s.stage<4})
            self.assertLessEqual(audited["marginal"]["classes"],audited["flags"]["classes"])
            self.assertLess(audited["flags"]["classes"],audited["literal"]["classes"])
            s=at(m,2)
            base=incumbent(m,s,Meter(100000))
            ref=ExactAuditor(m)
            bv,br=ref.action(s,base)
            for a in m.legal(s):
                if a==base:
                    continue
                av,ar=ref.action(s,a)
                for mode in MODES:
                    f=Frontier(m,Meter(200000),mode)
                    f.init(s,a,base)
                    prev=list(f.total)
                    entries=dict(f.entries)
                    if f.entries:
                        f.meter.cap=f.meter.used+8
                        with self.assertRaises(Exhausted):
                            f.expand()
                        self.assertEqual(f.total,prev)
                        self.assertEqual(f.entries,entries)
                    f.meter.cap=200000
                    while f.entries:
                        f.expand()
                    self.assertEqual(f.total,[av-bv,av-bv,ar-br,ar-br])
                    try:
                        t=compare(m,s,a,base,Meter(2048),mode)
                    except Exhausted:
                        continue
                    lo,hi,rl,rh=t["bounds"]
                    self.assertLessEqual(lo,av-bv)
                    self.assertLessEqual(av-bv,hi)
                    self.assertLessEqual(rl,ar-br)
                    self.assertLessEqual(ar-br,rh)
            ref.close()

    def test_risk_certificate(self):
        spec=example().spec.copy()
        spec["budget"]=2
        m=Model(spec)
        _,s=m.branches(m.initial(),"control")[0]
        for _ in range(3):
            _,s=m.branches(s,"defer")[0]
        for mode in MODES:
            met=Meter(8192)
            b=incumbent(m,s,met)
            rec,st=certificate(m,s,b,met,CONFIGS["late"],16,mode=mode)
            self.assertEqual(rec["chosen"],"screen")
            self.assertEqual(rec["risk_upper_numerator"],0)
            ref=Auditor(m)
            self.assertTrue(ref.verify(s,rec))
            self.assertGreater(st["search_work"],0)
            ref.close()

    def test_diagnostic_and_campaign(self):
        d=diagnostic(example())
        self.assertGreaterEqual(d["candidate_pairs"],d["overlap"]["literal"]["pairs_with_overlap"])
        self.assertGreaterEqual(d["overlap"]["marginal"]["pairs_with_overlap"],
                                d["overlap"]["literal"]["pairs_with_overlap"])
        with tempfile.TemporaryDirectory() as tmp:
            r=campaign(Path(tmp)/"run",counts=(1,1,1,1),caps=(8192,))
            self.assertEqual(len(r["row_index"]),3)
            self.assertEqual(len(r["diagnostic_index"]),3)
            self.assertFalse(r["training_admission"])
            for cohort in r["summary"].values():
                for p in cohort["policies"].values():
                    self.assertLessEqual(p["max_work"],8192)
                    self.assertEqual(p["risk_increased_roots"],0)
