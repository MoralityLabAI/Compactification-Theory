import tempfile
import unittest
from pathlib import Path

from Distill.research_program import Model, new_spec
from Distill.program_cancellation import Auditor
from Distill.lazy_certificate_memory import Memo, ARMS, streams, campaign
from Distill.program_quotient import signature
from Distill.program_student import Meter
from Distill.history_memory import histories


class LazyMemoryTests(unittest.TestCase):
    def test_unequal_mass_reuse(self):
        m=Model(new_spec(2026113099,5))
        paths=histories(m)
        by={}
        pair=None
        for s in paths:
            if s.stage!=3:
                continue
            k,_,_=signature(m,s,"marginal",Meter(100000))
            if k in by and m.mass(by[k].mask)!=m.mass(s.mask):
                pair=(by[k],s)
                break
            by[k]=s
        self.assertIsNotNone(pair)
        mem=Memo(m,"root_quotient",8)
        ref=Auditor(m)
        for s in pair:
            self.assertEqual(mem.query(s),(*ref.tail(s),m.mass(s.mask)))
        self.assertEqual(mem.stats["root_hits"],1)
        self.assertEqual(mem.stats["different_mask_hits"],1)
        ref.close()

    def test_empty_start_and_capacity(self):
        m=Model(new_spec(2026113099,6))
        qs=streams(m,24,7)
        self.assertEqual(len(set(qs["distinct"])),24)
        ref=Auditor(m)
        for name in ARMS:
            mem=Memo(m,name,2)
            self.assertEqual(len(mem.cache),0)
            for s in qs["distinct"]+qs["rollout"]:
                self.assertEqual(mem.query(s),(*ref.tail(s),m.mass(s.mask)))
                self.assertLessEqual(len(mem.cache),2)
            if name=="cold":
                self.assertEqual(mem.stats["writes"],0)
                self.assertEqual(mem.stats["root_hits"],0)
        ref.close()

    def test_smoke(self):
        with tempfile.TemporaryDirectory() as t:
            r=campaign(Path(t)/"run",n=1,count=8,caps=(2,4))
            self.assertFalse(r["training_admission"])
            self.assertEqual(len(r["row_index"]),12)
            for group in r["summary"].values():
                for a in group["arms"].values():
                    self.assertEqual(a["requests"],8)
                    self.assertEqual(a["incorrect_reuses"],0)
