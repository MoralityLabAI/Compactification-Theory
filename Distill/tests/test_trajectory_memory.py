import tempfile
import unittest
from pathlib import Path

from Distill.research_program import Model,new_spec
from Distill.program_student import Meter,Exhausted
from Distill.program_cancellation import Auditor
from Distill.lazy_certificate_memory import Memo
from Distill.trajectory_memory import Actor,action_value,audit_receipt,campaign,CONFIGS
from Distill.program_opportunity import incumbent
from Distill.program_oracle import actual_transition


class TrajectoryTests(unittest.TestCase):
    def test_interrupted_cache(self):
        m=Model(new_spec(2026120199,1))
        s=m.initial()
        mem=Memo(m,"recursive_hybrid",8)
        meter=Meter(30)
        with self.assertRaises(Exhausted):
            mem.query(s,meter)
        self.assertEqual(mem.stats["work"],meter.used)
        self.assertEqual(len(mem.cache),0)
        ref=Auditor(m)
        meter=Meter(100000)
        self.assertEqual(mem.query(s,meter),(*ref.tail(s),m.mass(s.mask)))
        self.assertLessEqual(mem.peak,8)
        ref.close()

    def test_exact_comparisons_and_fresh_episode(self):
        m=Model(new_spec(2026120199,2))
        ref=Auditor(m)
        s=m.initial()
        for _ in range(2):
            s=m.branches(s,"defer")[0][1]
        for mode in ("cold","recursive_literal","recursive_quotient","recursive_hybrid"):
            mem=Memo(m,mode,8)
            for a in m.legal(s):
                self.assertEqual(action_value(m,s,a,mem,Meter(100000)),ref.action(s,a))
        for config in CONFIGS.values():
            actor=Actor(m,config)
            self.assertEqual(len(actor.memo.cache),0)
            met=Meter(8192)
            state=m.initial()
            while state.stage<4:
                a,r=actor.choose(state,met)
                if r:
                    audit_receipt(ref,state,r)
                _,state,_=actual_transition(m,state,a,0)
            self.assertLessEqual(actor.stats["deviations"],1)
            self.assertLessEqual(met.used,8192)
        ref.close()

    def test_campaign(self):
        with tempfile.TemporaryDirectory() as t:
            r=campaign(Path(t)/"run",counts=(1,1,1,1),caps=(8192,))
            self.assertFalse(r["training_admission"])
            self.assertEqual(len(r["row_index"]),3)
            for group in r["summary"].values():
                for arm in group.values():
                    self.assertLessEqual(arm["cache_peak"],32)
