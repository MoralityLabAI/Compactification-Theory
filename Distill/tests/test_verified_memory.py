import importlib.util
import tempfile
import unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec("numpy"),"NumPy only needed for retrieval")
class VerifiedMemoryTests(unittest.TestCase):
    def test_exact_reference_and_encoder(self):
        import numpy as np
        from Distill.history_memory import context,CTX,EVENT
        from Distill.history_memory_train import RNN
        from Distill.program_student import ACTIONS
        from Distill.research_program import new_spec,Model
        from Distill.verified_memory import encode,cold_tail,records,evaluate_bank
        spec=new_spec(2026113099,1)
        model=RNN(311)
        h=(("screen",0),("defer",None),("control",1))
        for bag in (False,True):
            events=[(ACTIONS.index(a),0 if b is None else b+1) for a,b in h]
            if bag:
                events.sort()
            e=np.zeros((1,len(events),EVENT),np.float32)
            for i,(a,b) in enumerate(events):
                e[0,i,a]=1
                e[0,i,len(ACTIONS)+b]=1
            x=np.array([context(spec)],np.float32)
            d={"x":x,"e":e,"m":np.ones((1,len(events),1),np.float32)}
            z,work=encode(model,spec,h,bag)
            self.assertTrue(np.allclose(z,model.forward(d,np.array([0]))[0][0],atol=1e-6))
            self.assertGreater(work,0)
        value,work,visits=cold_tail(Model(spec),Model(spec).initial())
        self.assertGreater(work,0)
        self.assertGreater(visits,0)
        models={"random_init":(model,False)}
        b,q=records(spec,models,8,12)
        r=evaluate_bank(b,q,models)
        self.assertEqual(r["quotient_index"]["hits"],r["quotient_index"]["coverage"])
        self.assertEqual(r["literal_index"]["hits"],0)
        self.assertTrue(all(x["incorrect_reuses"]==0 for x in r.values()))

    def test_campaign_smoke(self):
        from Distill.verified_memory import campaign
        with tempfile.TemporaryDirectory() as t:
            r=campaign(Path(t)/"run",n=1,train_n=1,epochs=1,seeds=(311,),limit=8,caps=(8,),query_count=8)
            self.assertFalse(r["training_admission"])
            self.assertEqual(len(r["summary"]),3)
            for cohort in r["summary"].values():
                for arm in cohort.values():
                    self.assertEqual(arm["hits"]+arm["fallbacks"],arm["queries"])
                    self.assertEqual(arm["incorrect_reuses"],0)
