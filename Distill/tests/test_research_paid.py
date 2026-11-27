"""Paid planning tests: invented rates and oracle fixtures, no provider calls."""
import tempfile
import unittest
from pathlib import Path
from Distill.batch import dollars,rates
from Distill.core import split_quiz,write_json
from Distill.research_cases import compile_research
from Distill.research_oracle import reference
from Distill.ledger import Ledger
from Distill.research_paid import create_plan,load_paid_plan,freeze,grade

def fixture_groups():
    return [split_quiz(compile_research("research_portfolio",seed=123,ordinal=i),max_questions=2) for i in range(3)]

def reply(p):
    answers=[]
    for q in p["questions"]:
        f={k:"fixture only" for k in q["fields"]}
        if q["id"]=="case.solve":f["answer"]=reference(p["public_case"])
        answers.append({"question_id":q["id"],"status":"answered","fields":f})
    return {"schema_version":"distill.response.v2","packet_id":p["packet_id"],
            "model":{"provider":"fixture","revision":"offline","run_id":"test"},"answers":answers}

def plan(root,groups,active):
    return create_plan(root,groups=groups,source={"fixture":True},
        prices=rates(dollars("10"),dollars("50")),model="fixture",
        rate_evidence="invented rates for test",terms_evidence="no provider",
        active=dollars(active),input_cap=1000,output_cap=1000)

class ResearchPaidTests(unittest.TestCase):
    def test_whole_case_cap_and_frozen_grading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"paid"; groups=fixture_groups()
            p=plan(root,groups,"0.25")
            self.assertEqual(len(p["jobs"]),4)
            self.assertEqual(p["authorized_max_microusd"],240000)
            self.assertEqual(p["reserve_microusd"],199750000)
            self.assertEqual(load_paid_plan(root)[0],p)
            ledger=Ledger(root)
            self.assertEqual(ledger.status()["authorized_budget_microusd"],250000)
            r=Path(d)/"responses";r.mkdir()
            with self.assertRaises(ValueError):freeze(root,r)
            for i,packet in enumerate(groups[0]+groups[1]):
                rid=packet["packet_id"]
                ledger.reserve(rid,counted_input_tokens=100)
                with self.assertRaises(ValueError):ledger.reserve(rid,counted_input_tokens=100)
                ledger.settle(rid,input_tokens=100,output_tokens=100,provider_receipt="fixture-"+str(i))
                write_json(r/(rid+".json"),reply(packet))
            self.assertEqual(Ledger(root).status()["billed_at_supplied_rates_microusd"],24000)
            freeze(root,r)
            result=grade(root)
            self.assertEqual((result["planned_packets"],result["exact_pass"]),(4,2))
            self.assertFalse(result["training_admission"])
            (root/"frozen"/(groups[0][0]["packet_id"]+".json")).write_text("{}")
            with self.assertRaises(ValueError):grade(root)

    def test_overrun_and_partial_case_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"paid";g=fixture_groups()
            with self.assertRaises(ValueError):plan(root,g,"0.11")
            with self.assertRaises(ValueError):plan(root,[g[0][:1]],"1")
            p=plan(root,g,"1")
            ledger=Ledger(root)
            a,b=[j["request_id"] for j in p["jobs"][:2]]
            ledger.reserve(a,counted_input_tokens=100)
            ledger.settle(a,input_tokens=100,output_tokens=1001,provider_receipt="fixture-overrun")
            self.assertTrue(ledger.status()["frozen"])
            with self.assertRaises(ValueError):ledger.reserve(b,counted_input_tokens=100)
            (root/"public"/(a+".txt")).write_text("changed")
            with self.assertRaises(ValueError):Ledger(root)
