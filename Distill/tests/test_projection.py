"""Exact projection, interruption, cache and paired evaluator checks."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from Distill.research_program import Model,new_spec
from Distill.program_student import Meter,Exhausted
from Distill.program_quotient import signature,closure,Frontier,budget_cap
from Distill.program_projection import Projector,OPTIMIZATIONS,prunable
from Distill.program_opportunity import incumbent
from Distill.program_cancellation import Auditor
from Distill.amortized_quotient_campaign import campaign,micro

class ProjectionTests(unittest.TestCase):
    def test_exact_on_complete_closure(self):
        for family in ("shared_control","shifted_control"):
            m=Model(new_spec(2026112899,1,family,"frontier"))
            ps=[Projector(m,opt) for opt in OPTIMIZATIONS]
            for s in closure(m,[m.initial()],all_actions=True):
                ref=signature(m,s,"marginal",Meter(100000))
                for p in ps:
                    self.assertEqual(ref,signature(m,s,"marginal",Meter(100000),p))
            for p in ps:
                self.assertLessEqual(len(p.cache),32)

    def test_cache_and_interruption(self):
        m=Model(new_spec(2026112899,2))
        s=m.initial()
        p=Projector(m,"bitset_cache",capacity=1)
        with self.assertRaises(Exhausted):
            signature(m,s,"marginal",Meter(24),p)
        self.assertIsNone(p.planes)
        self.assertEqual(len(p.cache),0)
        ref=signature(m,s,"marginal",Meter(100000))
        cold=Meter(100000)
        self.assertEqual(signature(m,s,"marginal",cold,p),ref)
        warm=Meter(100000)
        self.assertEqual(signature(m,s,"marginal",warm,p),ref)
        self.assertEqual(p.stats["cache_hits"],1)
        self.assertLess(warm.used,cold.used)
        t=m.branches(s,"defer")[0][1]
        self.assertEqual(signature(m,t,"marginal",Meter(100000),p),
                         signature(m,t,"marginal",Meter(100000)))
        self.assertEqual(p.stats["cache_evictions"],1)
        self.assertEqual(len(p.cache),1)
        with self.assertRaises((ValueError,AssertionError)):
            p.project(Model(m.spec),s,Meter(100000))

    def test_exact_frontier_and_prune(self):
        m=Model(new_spec(2026112899,3))
        s=m.initial()
        for _ in range(2):
            s=m.branches(s,"defer")[0][1]
        b=incumbent(m,s,Meter(100000))
        ref=Auditor(m)
        bv,br=ref.action(s,b)
        for a in m.legal(s):
            if a==b:
                continue
            av,ar=ref.action(s,a)
            for opt in OPTIMIZATIONS:
                frontier=Frontier(m,Meter(200000),"marginal",Projector(m,opt))
                frontier.init(s,a,b)
                if frontier.entries:
                    before=(dict(frontier.entries),list(frontier.total))
                    frontier.meter.cap=frontier.meter.used+8
                    with self.assertRaises(Exhausted):
                        frontier.expand()
                    self.assertEqual(before,(frontier.entries,frontier.total))
                frontier.meter.cap=200000
                while frontier.entries:
                    frontier.expand()
                self.assertEqual(frontier.total,[av-bv,av-bv,ar-br,ar-br])
        # A deterministic, discarded-flag action with spare experimental budget.
        s=replace(s,mask=1,budget=budget_cap(m,s)+m.cost("screen"))
        b=incumbent(m,s,Meter(100000))
        self.assertTrue(prunable(m,s,"screen",Meter(100000)))
        self.assertLessEqual(ref.action(s,"screen")[0],ref.action(s,b)[0])
        ref.close()

    def test_micro_and_campaign(self):
        m=Model(new_spec(2026112899,4))
        self.assertTrue(micro(m.spec))
        with tempfile.TemporaryDirectory() as t:
            report=campaign(Path(t)/"run",counts=(1,1,1,1),caps=(8192,))
            self.assertEqual(len(report["row_index"]),3)
            self.assertFalse(report["training_admission"])
            self.assertTrue(report["microbench"])
            for group in report["summary"].values():
                for arm in group.values():
                    self.assertLessEqual(arm["max_work"],8192)

if __name__=="__main__":
    unittest.main()
