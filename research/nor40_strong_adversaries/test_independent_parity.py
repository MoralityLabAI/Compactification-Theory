"""NOR-40 separate CAL/EVAL replayer parity, exclusively historical NOR-39 source."""
from __future__ import annotations
from hashlib import sha256
from zipfile import ZipFile
import json
import unittest

from research.nor40_strong_adversaries import experiment as graded
from research.nor40_strong_adversaries import receipt_audit as independent


class NOR40IndependentPreDataParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with ZipFile(graded.ROOT/
                     "research/nor39_clean_global_uniform/results/nor39_original_first_hosted.zip"
                     ) as z:
            row=json.loads(z.read("nor39_fixture.json"))["cases"][0]
        assert row["replicate"]==68 and row["degree"]==512
        cls.row=row
        cls.source=row["full_public_input_sha256"]
        cls.proposal=graded.nor37.public_proposal(row)
        cls.pred=cls.proposal["pred"]
        cls.geometry=graded.nor37.source_geometry(
            cls.pred,cls.proposal["header"],cls.source)

    def test_full_2048_CAL_and_stronger_attack_receipt_parity(self):
        old,cats,u,cu,cv,bu,bv,scores=self.geometry
        audit_old,atoms,union,acu,acv,abu,abv,curvature=independent.geometry(
            self.pred,self.proposal["header"])
        self.assertEqual([len(a) for a in atoms],[len(a) for a in cats])
        self.assertEqual((u,cu,cv,bu,bv),(union,acu,acv,abu,abv))
        self.assertEqual(list(scores),curvature)
        self.assertEqual(old,audit_old)
        cal,diagnostics=graded.schedules(old,cats,scores,self.source,"cal")
        other,counts=independent.policy_schedules(audit_old,atoms,curvature,
                                                   self.source,"cal")
        self.assertEqual(graded.digest(graded.canonical(cal)),
                         sha256(independent.canonical(other)).hexdigest())
        self.assertEqual(diagnostics,counts)
        exposure=graded.trial_exposure(cal,len(self.pred))
        prior=graded.trial_exposure(cal,len(self.pred),graded.BASE_CAL)
        baseline,strong,certificates,other_exposure=independent.independent_strong_games(
            other,self.source,len(self.pred))
        self.assertEqual(exposure,other_exposure)
        for p in graded.POLICIES:
            base,attack,certificate=graded.stronger_calibrated_attacks(
                exposure,prior,p,self.source)
            self.assertEqual(base,baseline[p])
            self.assertEqual(attack,strong[p])
            self.assertEqual(certificate,certificates[p])
        blind=graded.blind_attacks(self.source,cats,u,cu,cv,bu,bv,
                                  len(self.pred),exposure)
        other_blind=independent.independent_blind(self.source,atoms,union,
            acu,acv,abu,abv,len(self.pred),other_exposure)
        self.assertEqual(blind,other_blind)

    def test_512_independent_EVAL_and_exact_top(self):
        old,cats,u,cu,cv,bu,bv,scores=self.geometry
        audit_old,atoms,union,acu,acv,abu,abv,curvature=independent.geometry(
            self.pred,self.proposal["header"])
        eva,b=graded.schedules(old,cats,scores,self.source,"eval")
        other,b2=independent.policy_schedules(audit_old,atoms,curvature,
                                              self.source,"eval")
        self.assertEqual(graded.canonical(eva),independent.canonical(other))
        self.assertEqual(b,b2)
        self.assertEqual(graded.exact_top(eva,cu,cv),independent.exact_top(other,acu,acv))
        for policy in ("original_mix","global7_mix","global8_mix"):
            self.assertEqual(
                [graded.realized_component(self.source,t,policy)
                 for t in range(graded.EVAL)],
                [independent.realized_choice(self.source,t,policy)
                 for t in range(independent.EVAL)])


if __name__=="__main__":
    unittest.main()
