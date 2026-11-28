import tempfile
import unittest
from pathlib import Path

from Distill.research_program import Model,new_spec
from Distill.program_student import Meter
from Distill.program_opportunity import incumbent
from Distill.compiled_incumbent import (compiled_incumbent,check_compiler,campaign,
    base_table,fused_trial,COMPILED_TERMINAL,TERMINAL_ORIGINAL)
from Distill.program_cancellation import gate, candidates, Auditor


class CompilerTests(unittest.TestCase):
    def test_full_state_parity(self):
        for i,family in enumerate(("shared_control","shifted_control")):
            spec=new_spec(2026120199,20+i,family)
            d=check_compiler(spec)
            self.assertGreater(d["states"],0)
            self.assertGreater(d["strictly_cheaper_states"],0)
            self.assertLess(d["compiled_selection_work"],d["original_selection_work"])

    def test_terminal_and_fused(self):
        m=Model(new_spec(2026120199,23))
        ref=Auditor(m)
        s=m.initial()
        for _ in range(3):
            s=m.branches(s,"defer")[0][1]
        # Exhaustive legal history checks elsewhere cover natural budgets.
        from dataclasses import replace
        s=replace(s,budget=1)
        old,new=Meter(10000),Meter(10000)
        b=incumbent(m,s,old)
        self.assertEqual(compiled_incumbent(m,s,new),b)
        if gate(m,s,b,"late"):
            self.assertEqual(old.used,TERMINAL_ORIGINAL)
            self.assertEqual(new.used,COMPILED_TERMINAL)
            t=base_table(m,s,b,Meter(10000))
            bv,br=ref.action(s,b)
            for a in candidates(m,s,b,Meter(10000)):
                av,ar=ref.action(s,a)
                self.assertEqual(fused_trial(m,s,a,t,Meter(10000)),(av-bv,ar-br))
        ref.close()

    def test_campaign_smoke(self):
        with tempfile.TemporaryDirectory() as t:
            r=campaign(Path(t)/"run",counts=(1,1,1,1),caps=(8192,))
            self.assertFalse(r["training_admission"])
            for group in r["summary"].values():
                self.assertGreaterEqual(float(__import__("fractions").Fraction(group["compiled"]["priced_delta"])),0)
                self.assertEqual(group["compiled"]["risk_delta"],"0")
