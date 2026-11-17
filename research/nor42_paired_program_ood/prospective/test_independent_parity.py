"""NOR-42 distinct scorer/replayer parity using only an old NOR41 original source.

No invocation of NOR42 new_source, make_fixture, restore_worlds or evaluate,
nor use of any NOR42 registered TEST identity.
"""
from __future__ import annotations
import json,unittest
from fractions import Fraction
from hashlib import sha256
from zipfile import ZipFile

from research.nor42_paired_program_ood.prospective import experiment as x
from research.nor42_paired_program_ood.prospective import receipt_audit as independent


class NOR42HistoricalParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=x.ROOT/"research/nor41_scale_ood/results/nor41_original_first_hosted.zip"
        with ZipFile(path) as z:
            cls.old_row=json.loads(z.read("nor41_fixture.json"))["cases"][0]
        assert cls.old_row["replicate"]==84
        cls.proposal=x.nor37.public_proposal(cls.old_row)
        cls.src=cls.old_row["full_public_input_sha256"]

    def test_original_388_source_exclusion_parity(self):
        self.assertEqual(x.history(),independent.history())
        self.assertIn(self.src,x.history())
        self.assertEqual(len(x.history()),388)
        self.assertEqual(independent.REGISTERED,x.REGISTERED)

    def test_policy_weights_and_public_source_plan_parity(self):
        pred=self.proposal["pred"]
        header=self.proposal["header"]
        left=x.nor37.source_geometry(pred,header,self.src)
        right=independent.geometry(pred,header)
        old,cats,u,cu,cv,bu,bv,scores=left
        other,atoms,otheru,othercu,othercv,otherbu,otherbv,otherv=right
        self.assertEqual(old,other)
        self.assertEqual([len(c) for c in cats],[len(c) for c in atoms])
        self.assertEqual((u,cu,cv,bu,bv),(otheru,othercu,othercv,otherbu,otherbv))
        self.assertEqual(list(scores),otherv)
        for p in (*x.POLICIES,x.DIAGNOSTIC):
            self.assertEqual(x.components(p),independent.component_parts(p))
        a=x.schedules(old,cats,scores,self.src)
        b=independent.policy_schedules(other,atoms,otherv,self.src)
        self.assertEqual(x.canonical(a),independent.canonical(b))
        self.assertEqual(len(a["original"]["uniform"]),x.EVAL)
        q=tuple(self.old_row["public_query_input"])
        # Deliberately synthetic late suffix on a previously graded source.
        hidden=x.hidden_future(q,pred,self.old_row["degree"],1,"I")
        independently=independent.hidden_after_public(
            q,pred,self.old_row["degree"],1,"I")
        self.assertEqual(hidden,independently)
        S=frozenset(i for i,(v,w) in enumerate(zip(pred,hidden)) if v!=w)
        self.assertEqual(min(S),self.old_row["degree"]+32)
        self.assertEqual(x.weighted_detection(a,S),
                         independent.independent_weighted_detection(b,S))
        self.assertEqual(x.uniform_exact(len(pred),len(S)),
                         independent.exact_uniform(len(pred),len(S)))

    def test_hidden_suffix_normalized_transform_for_completed_source(self):
        pred=self.proposal["pred"]
        q=tuple(self.old_row["public_query_input"])
        for op in x.corpus.OPS:
            for beta in x.BETAS:
                a=x.hidden_future(q,pred,self.old_row["degree"],beta,op)
                b=independent.hidden_after_public(
                    q,pred,self.old_row["degree"],beta,op)
                self.assertEqual(a,b)


if __name__=="__main__":
    unittest.main()
