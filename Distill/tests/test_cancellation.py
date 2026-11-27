"""DISTILL-10 interval, cost, and risk certificates."""
import copy
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from Distill.core import digest
from Distill.research_program import Model,new_spec
from Distill.program_student import Meter,Exhausted
from Distill.program_cancellation import (Frontier,Auditor,Policy,CONFIGS,incumbent,
    compare,certificate,late_compare,candidates)
from Distill.cancellation_campaign import campaign,evaluate

def late(known):
    spec=new_spec(2026112869,0,"shared_control","frontier")
    spec["budget"]=2 if known else 1
    spec["prior_weights"]["local_c"]=[1,1]
    spec["prior_weights"]["nuisance"]=[8,1]
    spec["costs"]["control"]=1 if known else 2
    spec["costs"]["law_audit"]=2
    m=Model(spec)
    s=m.initial()
    if known:
        _,s=m.branches(s,"control")[0]
    for _ in range(3):
        _,s=m.branches(s,"defer")[0]
    return m,s

class CancellationTests(unittest.TestCase):
    def test_exact_cancellation_and_split(self):
        m=Model(new_spec(2026112862,0))
        s=m.initial()
        audit=Auditor(m)
        av,ar=audit.explicit(s,"recommend:0")
        bv,br=audit.explicit(s,"defer")
        exact=[av-bv,av-bv,ar-br,ar-br]
        a=Frontier(m,Meter(200000),cancel=True)
        a.init(s,"recommend:0","defer")
        self.assertEqual(a.total,exact)
        self.assertEqual(a.cancelled_pairs,1)
        self.assertFalse(a.entries)
        b=Frontier(m,Meter(200000),cancel=False)
        b.init(s,"recommend:0","defer")
        self.assertTrue(b.entries)
        while b.entries:
            b.expand()
        self.assertEqual(b.total,exact)
        self.assertLess(a.meter.used,b.meter.used)
        audit.close()

    def test_general_bounds_and_partial_expansion(self):
        m=Model(new_spec(2026112862,1))
        s=m.initial()
        for _ in range(2):
            _,s=m.branches(s,"defer")[0]
        base=incumbent(m,s,Meter(100000))
        audit=Auditor(m)
        bv,br=audit.action(s,base)
        for a in m.legal(s):
            if a==base: continue
            av,ar=audit.action(s,a)
            for cancel in (True,False):
                f=Frontier(m,Meter(100000),cancel=cancel)
                f.init(s,a,base)
                before=list(f.total)
                entries=dict(f.entries)
                f.meter.cap=f.meter.used+24
                if f.entries:
                    with self.assertRaises(Exhausted):
                        f.expand()
                    self.assertEqual(f.total,before)
                    self.assertEqual(f.entries,entries)
                for cap in (2048,4096):
                    try:
                        t=compare(m,s,a,base,Meter(cap),cancel=cancel)
                    except Exhausted:
                        continue
                    lo,hi,rl,rh=t["bounds"]
                    self.assertLessEqual(lo,av-bv); self.assertLessEqual(av-bv,hi)
                    self.assertLessEqual(rl,ar-br); self.assertLessEqual(ar-br,rh)
                f.meter.cap=100000
                while f.entries:
                    f.expand()
                self.assertEqual(f.total,[av-bv,av-bv,ar-br,ar-br])
        audit.close()

    def test_risk_veto_and_safe_opportunity(self):
        for known in (False,True):
            m,s=late(known)
            meter=Meter(8192)
            b=incumbent(m,s,meter)
            self.assertEqual(b,"defer")
            self.assertIn("screen",candidates(m,s,b,Meter(100000)))
            ref=Auditor(m)
            trial=late_compare(m,s,"screen",b,Meter(100000))
            av,ar=ref.explicit(s,"screen")
            bv,br=ref.explicit(s,b)
            self.assertEqual(trial["bounds"],[av-bv,av-bv,ar-br,ar-br])
            receipt,stats=certificate(m,s,b,meter,CONFIGS["late"],16)
            self.assertIsNotNone(receipt)
            self.assertTrue(ref.verify(s,receipt))
            self.assertGreater(stats["search_work"],0)
            if known:
                self.assertEqual(receipt["chosen"],"screen")
                self.assertEqual(receipt["risk_upper_numerator"],0)
                bad=copy.deepcopy(receipt)
                bad["risk_upper_numerator"]=-1
                bad["receipt_id"]=digest({k:v for k,v in bad.items() if k!="receipt_id"})
                with self.assertRaises(ValueError):
                    ref.verify(s,bad)
            else:
                self.assertGreater(ar-br,0)
                self.assertIsNone(receipt["chosen"])
            ref.close()

    def test_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=campaign(Path(tmp)/"run",counts=(1,1,1,1),caps=(8192,))
            self.assertEqual(len(r["row_index"]),3)
            self.assertFalse(r["training_admission"])
            for cohort in r["summary"].values():
                for name,p in cohort["policies"].items():
                    self.assertLessEqual(p["max_work"],8192)
                    if name.startswith(("paired_","split_")):
                        self.assertEqual(p["risk_increased_roots"],0)
