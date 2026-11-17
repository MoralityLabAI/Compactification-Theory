"""NOR-41 evaluator/replayer parity against archived NOR-40 source only."""
from __future__ import annotations
from hashlib import sha256
from zipfile import ZipFile
import json,random,unittest

from research.nor41_scale_ood import experiment as grader
from research.nor41_scale_ood import receipt_audit as replay


class NOR41ArchivedSourceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive=grader.ROOT/"research/nor40_strong_adversaries/results/nor40_original_first_hosted.zip"
        with ZipFile(archive) as z:
            row=json.loads(z.read("nor40_fixture.json"))["cases"][0]
        cls.row=row
        cls.src=row["full_public_input_sha256"]
        cls.public=grader.nor37.public_proposal(row)
        cls.pred=cls.public["pred"]
        cls.old,cls.cats,cls.u,cls.cu,cls.cv,cls.bu,cls.bv,cls.scores=(
            grader.nor37.source_geometry(cls.pred,cls.public["header"],cls.src))

    def test_synthetic_regime_interpreter_exact_rng(self):
        for seed in (0,44,77157):
            for q in (512,768,1536):
                left=grader.public_input(random.Random(seed),"wide_regime_blocks",q)
                right=replay.independent_input(random.Random(seed),"wide_regime_blocks",q)
                self.assertEqual(left,right)
        self.assertNotEqual(
            grader.public_input(random.Random(44),"wide_dense_small",768),
            grader.public_input(random.Random(44),"wide_regime_blocks",768))

    def test_full_CAL4096_greedy_swap_exact_pool16_and_EVAL_parity(self):
        old,atoms,u,cu,cv,bu,bv,curvature=replay.geometry(
            self.pred,self.public["header"])
        self.assertEqual(old,self.old)
        self.assertEqual(tuple(map(len,atoms)),tuple(map(len,self.cats)))
        self.assertEqual((u,cu,cv,bu,bv),(self.u,self.cu,self.cv,self.bu,self.bv))
        self.assertEqual(curvature,list(self.scores))
        self.assertEqual(replay.historical_source_ids(),grader.history())
        for policy in (*grader.POLICIES,grader.DIAGNOSTIC):
            self.assertEqual(grader.components(policy),replay.component_parts(policy))
        plans=grader.schedules(self.old,self.cats,self.scores,self.src,"cal")
        independent=replay.policy_schedules(old,atoms,curvature,self.src,"cal")
        self.assertEqual(grader.digest(grader.canonical(plans)),
                         sha256(replay.canonical(independent)).hexdigest())
        exp=grader.trial_exposure(plans,len(self.pred))
        trained,proofs,otherexp=replay.independent_nested_games(
            independent,self.src,len(self.pred))
        self.assertEqual(exp,otherexp)
        for n in grader.BUDGETS:
            for p in grader.POLICIES:
                attacks,certificate=grader.nested_attacks(exp,p,self.src,n)
                self.assertEqual(attacks,trained[str(n)][p])
                self.assertEqual(certificate,proofs[str(n)][p])
        self.assertEqual(
            grader.blind_attacks(self.src,self.cats,self.u,self.cu,self.cv,
                                 self.bu,self.bv,len(self.pred),exp),
            replay.independent_blind(self.src,atoms,u,cu,cv,bu,bv,len(self.pred),
                                     otherexp))
        ev=grader.schedules(self.old,self.cats,self.scores,self.src,"eval")
        other=replay.policy_schedules(old,atoms,curvature,self.src,"eval")
        self.assertEqual(grader.canonical(ev),replay.canonical(other))
        self.assertEqual(grader.exact_top(ev,self.cu,self.cv),
                         replay.exact_top(other,cu,cv))
        for policy in ("original_mix","deterministic_mix","global7_mix"):
            self.assertEqual(
                [grader.realized_component(self.src,t,policy)
                 for t in range(grader.EVAL)],
                [replay.realized_choice(self.src,t,policy)
                 for t in range(replay.EVAL)])


if __name__=="__main__":
    unittest.main()
