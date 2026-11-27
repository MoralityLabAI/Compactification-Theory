import unittest
import tempfile
from fractions import Fraction
from pathlib import Path
from Distill.research_program import Model,new_spec
from Distill.program_oracle import replay,Baseline
from Distill.program_student import Meter
from Distill.program_residual import Counterfactual,Residual,learn_gate,labeled_rows,select,reachable,DIM
from Distill.residual_campaign import campaign,evaluate,FIXED

class ResidualTests(unittest.TestCase):
    def test_one_deviation_value_replay(self):
        spec=new_spec(2026112841,0)
        model=Model(spec)
        cf=Counterfactual(model)
        state=model.initial()
        base,rs,star=cf.label(state)
        base_value=Fraction(replay(model,cf.base.choose)["utility"])
        self.assertEqual(rs[base],0)
        self.assertEqual(star[base],0)
        for a in model.legal(state):
            value=Fraction(replay(model,lambda s:a if s==state else cf.base.choose(s))["utility"])
            self.assertEqual(value-base_value,rs[a])
        cf.close()

    def test_fit_and_label_coverage(self):
        spec=new_spec(2026112841,1)
        rows,stats=labeled_rows(spec)
        self.assertTrue(rows)
        self.assertGreater(stats["incumbent_states"],0)
        self.assertTrue(all(len(r["features"])==DIM for r in rows))
        self.assertTrue(all(r["residual"][r["base"]]==0 for r in rows))
        a=Residual.fit(rows)
        b=Residual.fit(rows)
        self.assertEqual(a.weights,b.weights)
        final,st=labeled_rows(spec,provisional=a)
        self.assertGreaterEqual(st["reachable"],stats["reachable"])
        self.assertGreater(st["provisional_states"],0)
        self.assertEqual(st["action_labels"],sum(len(r["residual"]) for r in final))

    def test_gate_charged_before_scoring(self):
        spec=new_spec(2026112841,2)
        m=Model(spec)
        s=m.initial()
        rows,_=labeled_rows(spec)
        a=Residual.fit(rows)
        base_meter=Meter(8192)
        base,info=select(m,s,base_meter,"reuse_first")
        skip_meter=Meter(8192)
        out,info=select(m,s,skip_meter,"gated",a,{},FIXED)
        self.assertEqual(out,base)
        self.assertEqual(info["score_calls"],0)
        self.assertEqual(info["gate_skips"],1)
        self.assertEqual(skip_meter.used-base_meter.used,32)
        low_meter=Meter(base_meter.used)
        out,info=select(m,s,low_meter,"always",a)
        self.assertEqual(out,base)
        self.assertEqual(info["fallbacks"],1)
        self.assertEqual(low_meter.used,base_meter.used)

    def test_full_replay_and_smoke(self):
        spec=new_spec(2026112841,0)
        got=evaluate(spec,"reuse_first",32768)
        want=replay(Model(spec),Baseline(Model(spec),"reuse_first").choose)
        self.assertEqual(got["utility"],want["utility"])
        with tempfile.TemporaryDirectory() as tmp:
            r=campaign(Path(tmp)/"run",counts=(1,1,1,1),caps=(8192,),limit=32)
            self.assertEqual(len(r["rows"]),2)
            self.assertFalse(r["training_admission"])
            for row in r["rows"]:
                for p in row["policies"].values():
                    self.assertLessEqual(p["max_work"],8192)
                    self.assertLessEqual(Fraction(p["utility"]),Fraction(row["oracle_value"]))
