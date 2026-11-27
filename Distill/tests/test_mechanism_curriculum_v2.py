"""Regression tests for continuation equivalence and multi-context compression."""
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from Distill.core import digest
from Distill.mechanism_curriculum_v2 import (project,state,assess_quotient,assess_quantization,
    normalize,content_id,export)

class MechanismV2Tests(unittest.TestCase):
    def test_budget_alias_and_equal_value_not_congruence(self):
        p=project()
        a=assess_quotient({"projects":[p,p],"states":[state(2,True,True),state(4,True,True)]})
        self.assertEqual((a["value_a"],a["value_b"]),("9/1","18/1"))
        self.assertFalse(a["equivalent"])
        self.assertIsNotNone(a["witness"])
        q=project(direct=2,transport=2)
        b=assess_quotient({"projects":[q,q],"states":[state(4,True,True),state(4,False,True)]})
        self.assertTrue(b["equal_value_unsafe_merge"])
        self.assertFalse(b["equivalent"])
        c=assess_quotient({"projects":[p,p],"states":[state(6,True,True),state(10,True,True)]})
        self.assertTrue(c["equivalent"])
        self.assertEqual(c["baseline_regrets"]["capped_eligibility"],"0/1")
        dead=project(scope=False)
        self.assertTrue(assess_quotient({"projects":[dead,dead],"states":[state(6,True,True),state(6,False,True)]})["equivalent"])

    def test_quantization_tradeoff_and_normalization(self):
        d={"bins":2,"atoms":[{"p":"1/10","w":1},{"p":"2/5","w":1},{"p":"9/10","w":1}],
           "contexts":[{"fp":3,"fn":7,"w":1,"defer":None},{"fp":7,"fn":3,"w":1,"defer":None}]}
        a=assess_quantization(d)
        self.assertGreater(F(a["optimal_regret"]),0)
        self.assertEqual(a["independent_partitions"],2)
        x={"bins":2,"atoms":[{"p":"18/20","w":2},{"p":"4/10","w":2},{"p":"1/10","w":1},{"p":"2/20","w":1}],
           "contexts":[{"fp":7,"fn":3,"w":2,"defer":None},{"fp":3,"fn":7,"w":2,"defer":None}]}
        self.assertEqual(content_id("multi_context_quantization",d),content_id("multi_context_quantization",x))
        p=project()
        q={"projects":[p,p],"states":[state(2,True,True),state(4,False,True)]}
        self.assertEqual(content_id("receipt_continuation",q),content_id("receipt_continuation",{**q,"states":q["states"][::-1]}))

    def test_full_export(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m=export(root/"a")
            self.assertEqual(m,export(root/"b"))
            self.assertEqual(m["total_rows"],216)
            self.assertFalse(m["training_admission"])
            seen=set()
            counts={}
            for sp in m["splits"]:
                name=sp["split"]
                p=[json.loads(x) for x in (root/"a"/(name+".public.jsonl")).read_text().splitlines()]
                r=[json.loads(x) for x in (root/"a"/(name+".referee.jsonl")).read_text().splitlines()]
                self.assertEqual(digest(p),sp["public_sha256"])
                self.assertEqual(digest(r),sp["referee_sha256"])
                self.assertEqual(len(p),sp["rows"])
                for x,y in zip(p,r):
                    self.assertEqual(x["row_id"],content_id(x["mechanism"],x["instance"]))
                    self.assertNotIn(x["row_id"],seen)
                    seen.add(x["row_id"])
                    self.assertEqual(y["public_sha256"],digest(x))
                    self.assertEqual(set(x["legal_actions"]),set(y["action_regrets"]))
                    self.assertEqual(F(y["action_regrets"][y["optimal_action"]]),0)
                    self.assertTrue(all(F(v)>=0 for v in y["action_regrets"].values()))
                    self.assertNotIn("optimal_action",x)
                    if sp["structural_shift"] and x["mechanism"]=="multi_context_quantization":
                        self.assertTrue(all(c["defer"] is not None for c in x["instance"]["contexts"]))
                    if sp["structural_shift"] and x["mechanism"]=="receipt_continuation":
                        self.assertGreaterEqual(len(x["instance"]["projects"]),3)
                        self.assertFalse(x["instance"]["projects"][-1]["scope"])
                counts[name]=sp["counts"]
            self.assertEqual(len(seen),216)
            for v in counts.values():
                self.assertGreater(v["equal_value_unsafe_merge"],0)
                self.assertGreater(v["positive_value_gap"],0)
                self.assertGreater(v["equivalent"],0)
                self.assertGreater(v["positive_optimal_regret"],0)
                self.assertGreater(v["brier_decision_disagreement"],0)
            print("DISTILL_12_MECHANISM_V2",json.dumps(m,sort_keys=True))
