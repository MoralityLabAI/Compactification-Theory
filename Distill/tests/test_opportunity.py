"""Independent checks of priced opportunity certificates and metered continuation."""
import copy
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from Distill.core import digest
from Distill.research_program import Model,new_spec,validate_spec
from Distill.program_student import Meter
from Distill.program_opportunity import (CONFIGS,Bounds,Auditor,Policy,incumbent,reserve,
    certificate,NONTERMINAL_WORK,TERMINAL_WORK)
from Distill.opportunity_campaign import make_spec,evaluate,campaign

def late_spec():
    spec=new_spec(2026112859,0,"shared_control","frontier")
    spec["budget"]=1
    spec["prior_weights"]["local_c"]=[1,1]
    spec["prior_weights"]["nuisance"]=[8,1]
    spec["costs"]["control"]=2
    spec["costs"]["law_audit"]=2
    validate_spec(spec)
    return spec

class OpportunityTests(unittest.TestCase):
    def test_work_reserve_all_small_states(self):
        spec=make_spec(2026112854,2,"shifted_control",True)
        spec["budget"]=5
        m=Model(spec)
        stack=[m.initial()]
        seen=set()
        while stack:
            s=stack.pop()
            if s.stage==4 or s in seen:
                continue
            seen.add(s)
            meter=Meter(100000)
            a=incumbent(m,s,meter)
            self.assertLessEqual(meter.used,TERMINAL_WORK if m.terminal(a) else NONTERMINAL_WORK)
            for _,t in m.branches(s,a):
                self.assertLessEqual(meter.used+reserve(t),reserve(s))
            for choice in m.legal(s):
                stack.extend(t for _,t in m.branches(s,choice))
        self.assertGreater(len(seen),100)

    def test_bounded_intervals_and_exact_world_replay(self):
        m=Model(make_spec(2026112852,0,"shared_control"))
        s=m.initial()
        _,s=m.branches(s,"defer")[0]
        audit=Auditor(m)
        for a in m.legal(s):
            truth=audit.action(s,a)
            self.assertEqual(truth,audit.replay_action(s,a))
            for depth in (0,1,4):
                engine=Bounds(m,Meter(1000000))
                lo,hi=engine.action(s,a,depth)
                self.assertLessEqual(lo,truth)
                self.assertLessEqual(truth,hi)
        audit.close()

    def test_positive_late_certificate(self):
        spec=late_spec()
        m=Model(spec)
        s=m.initial()
        for _ in range(3):
            _,s=m.branches(s,"defer")[0]
        meter=Meter(8192)
        base=incumbent(m,s,meter)
        self.assertEqual(base,"defer")
        receipt,stats=certificate(m,s,base,meter,CONFIGS["late"],16)
        self.assertEqual(receipt["chosen"],"screen")
        self.assertGreater(receipt["net_lower_numerator"],0)
        self.assertGreater(stats["search_work"],0)
        audit=Auditor(m)
        self.assertTrue(audit.verify(s,receipt))
        audit.replay_action(s,"screen")
        bad=copy.deepcopy(receipt)
        bad["net_lower_numerator"]+=receipt["mass"]
        bad["receipt_id"]=digest({k:v for k,v in bad.items() if k!="receipt_id"})
        with self.assertRaises(ValueError):
            audit.verify(s,bad)
        audit.close()

    def test_meter_and_forced_base(self):
        spec=late_spec()
        base,_=evaluate(spec,None,8192)
        p,rs=evaluate(spec,CONFIGS["late"],8192)
        forced,_=evaluate(spec,CONFIGS["late"],8192,forced=True)
        self.assertGreater(Fraction(p["priced_utility"]),Fraction(base["priced_utility"]))
        self.assertEqual(forced["utility"],base["utility"])
        self.assertEqual(Fraction(p["accepted"]),1)
        self.assertEqual(Fraction(p["deviations"]),1)
        self.assertEqual(Fraction(forced["deviations"]),0)
        self.assertTrue(rs)
        self.assertLessEqual(p["max_work"],8192)
        self.assertEqual(Fraction(p["fallbacks"]),0)
        m=Model(spec)
        s=m.initial()
        for _ in range(3):
            _,s=m.branches(s,"defer")[0]
        meter=Meter(400)
        a,info,receipt=Policy(CONFIGS["late"]).choose(m,s,meter)
        self.assertEqual(a,"defer")
        self.assertEqual(info["attempts"],0)
        self.assertIsNone(receipt)

    def test_campaign_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=campaign(Path(tmp)/"run",counts=(1,1,1,1),caps=(8192,))
            self.assertEqual(len(r["rows"]),3)
            self.assertFalse(r["training_admission"])
            for row in r["rows"]:
                for p in row["policies"].values():
                    self.assertLessEqual(p["max_work"],8192)
                    self.assertLessEqual(Fraction(p["utility"]),Fraction(row["oracle_value"]))
