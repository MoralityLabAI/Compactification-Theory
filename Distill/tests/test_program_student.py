import copy
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from Distill.core import digest
from Distill.program_student import ACTIONS,FEATURES,Ranker,Meter,Exhausted,public_features,state_from_observation
from Distill.student_campaign import cheap,evaluate,labeled,campaign
from Distill.research_program import Model,Program,new_spec
from Distill.program_oracle import Baseline
from Distill.coe_certificate import Native,graph,transport,verify_certificate,root_graph,benchmark

class StudentTests(unittest.TestCase):
    def test_public_state_and_features(self):
        spec=new_spec(2026112831,0)
        env=Program(spec,63)
        for action in ("screen","direct_audit","law_audit","certify:1","transport"):
            obs=env.observe()
            m,s=state_from_observation(obs)
            self.assertEqual(s,env.state)
            self.assertEqual(public_features(m,s),public_features(env.model,env.state))
            self.assertEqual(len(public_features(m,s)),FEATURES)
            env.step(action)
        obs=env.observe()
        obs["remaining_credits"]+=1
        obs["state_id"]=digest({k:v for k,v in obs.items() if k!="state_id"})
        with self.assertRaises(ValueError):
            state_from_observation(obs)

    def test_gradient(self):
        model=Ranker(17,3)
        x=[.15*((j%7)-3) for j in range(FEATURES)]
        targets={0:-.2,2:-.1,7:-.8}
        _,g=model.loss_gradient(x,targets)
        for j in (0,FEATURES,model.offset_h,model.offset_h+17,model.offset_o,model.offset_o+10):
            orig=model.p[j]
            model.p[j]=orig+1e-5
            hi=model.loss_gradient(x,targets)[0]
            model.p[j]=orig-1e-5
            lo=model.loss_gradient(x,targets)[0]
            model.p[j]=orig
            self.assertAlmostEqual(g[j],(hi-lo)/2e-5,places=6)

    def test_meter_and_cheap_parity(self):
        m=Meter(10)
        m.charge(9)
        with self.assertRaises(Exhausted):
            m.charge(2)
        self.assertEqual(m.used,9)
        spec=new_spec(2026112831,1)
        env=Program(spec,63)
        for _ in range(8):
            s=env.state
            if s.stage==4: break
            for name in ("stop","direct_first","myopic","reuse_first"):
                self.assertEqual(cheap(env.model,s,Meter(100000),name),Baseline(env.model,name).choose(s))
            env.step(Baseline(env.model,"reuse_first").choose(s))
        row=evaluate(spec,"reuse_first",8192)
        self.assertLessEqual(row["max_work"],8192)
        self.assertLessEqual(Fraction(row["priced_utility"]),Fraction(row["utility"]))

    def test_fit_determinism_and_campaign_smoke(self):
        rows=labeled(new_spec(2026112831,0))
        xy=[(x,y) for _,_,x,y in rows]
        a,b=Ranker(17,3),Ranker(17,3)
        a.fit(xy,epochs=2)
        b.fit(xy,epochs=2)
        self.assertEqual(a.p,b.p)
        with tempfile.TemporaryDirectory() as tmp:
            r=campaign(Path(tmp)/"run",counts=(1,1,1,1),epochs=1,seeds=(17,),caps=(8192,))
            self.assertEqual(len(r["fit"]),3)
            self.assertEqual(len(r["rows"]),2)
            self.assertFalse(r["training_admission"])
            for row in r["rows"]:
                for policies in row["policies"].values():
                    for p in policies:
                        self.assertLessEqual(p["max_work"],8192)
                        self.assertLessEqual(Fraction(p["utility"]),Fraction(row["oracle_value"]))

class CertificateTests(unittest.TestCase):
    def test_native_transport_and_reject(self):
        native=Native()
        anchor,targets=root_graph(2026112835,0)
        c=native.cold(anchor)
        for g,p in targets[:2]:
            target,receipt=transport(c,g,p)
            self.assertEqual(verify_certificate(target),verify_certificate(native.cold(g)))
            self.assertEqual(receipt["source_certificate_id"],c["certificate_id"])
        with self.assertRaises(ValueError):
            transport(c,*targets[2])
        bad=copy.deepcopy(c)
        bad["words"][0]=bad["words"][1]
        bad["certificate_id"]=digest({k:v for k,v in bad.items() if k!="certificate_id"})
        with self.assertRaises(ValueError):
            verify_certificate(bad)

    def test_lower_bound_family_incomplete(self):
        # Odd cycle has width two, but its largest clique gives only one bit.
        g=graph(5,[(i,(i+1)%5) for i in range(5)])
        with self.assertRaises(ValueError):
            Native().cold(g)

    def test_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=benchmark(Path(tmp)/"reuse",roots=4)
            self.assertEqual(r["totals"]["cold_solves"],16)
            self.assertEqual(r["totals"]["reuse_solves"],8)
            self.assertEqual(r["totals"]["transport_success"],8)
            self.assertEqual(r["totals"]["transport_reject"],4)
            self.assertEqual(r["checked_target_widths"],16)
