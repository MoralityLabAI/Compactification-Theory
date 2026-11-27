"""Structured transport round-trips all research families, fixture answers only."""
import copy
import json
import unittest
from Distill.core import split_quiz
from Distill.research_cases import FAMILIES,DOCKET,compile_research
from Distill.research_oracle import reference,grade_research
from Distill.structured_research import SCHEMA,wire_schema,render,convert,convert_bytes

MODEL={"provider":"fixture","revision":"offline","run_id":"test"}

def flat(tree,path=""):
    if "decision" in tree:
        return [{"path":path,"decision":tree["decision"]}]
    return [{"path":path,"test":tree["test"]}]+flat(tree["zero"],path+"0")+flat(tree["one"],path+"1")

def fixture(packet):
    answers={}
    for q in packet["questions"]:
        f={k:"fixture" for k in q["fields"]}
        if q["id"]=="case.solve":
            v=reference(packet["public_case"])
            if packet["research"]["family"]=="adaptive_evidence":
                v={"nodes":flat(v["policy"]),"value":v["value"]}
            f["answer"]=v
        answers[q["id"]]={"status":"answered","fields":f}
    return {"schema_version":SCHEMA,"packet_id":packet["packet_id"],"questions":answers}

class StructuredResearchTests(unittest.TestCase):
    def test_all_families_and_abstain(self):
        for family in (*FAMILIES,DOCKET):
            seed=0 if family==DOCKET else 5
            packet=compile_research(family,seed=seed,ordinal=0)
            for p in split_quiz(packet,max_questions=2):
                self.assertNotIn("$ref",json.dumps(wire_schema(p)))
                self.assertIn(p["packet_id"],render(p))
                w=fixture(p)
                response,receipt=convert_bytes(p,json.dumps(w).encode(),MODEL)
                self.assertFalse(receipt["training_admission"])
                self.assertEqual(response,convert(p,w,MODEL))
                grade=grade_research(p,response)
                for q in grade["checks"]:
                    if q["question_id"]=="case.solve":
                        self.assertEqual(q["exact_check"],"pass")
                q=p["questions"][0]["id"]
                w["questions"][q]={"status":"abstain","fields":{"reason":"fixture"}}
                self.assertEqual(convert(p,w,MODEL)["answers"][0]["status"],"abstain")

    def test_wire_rejects_extra_keys_missing_nodes_and_duplicates(self):
        p=split_quiz(compile_research("adaptive_evidence",seed=3,ordinal=0),max_questions=2)[0]
        w=fixture(p)
        q=copy.deepcopy(w)
        q["questions"]["case.solve"]["fields"]["extra"]="bad"
        with self.assertRaises(ValueError):convert(p,q,MODEL)
        q=copy.deepcopy(w)
        nodes=q["questions"]["case.solve"]["fields"]["answer"]["nodes"]
        nodes.append(copy.deepcopy(nodes[0]))
        with self.assertRaises(ValueError):convert(p,q,MODEL)
        q=copy.deepcopy(w)
        q["questions"]["case.solve"]["fields"]["answer"]["nodes"]=[{"path":"","test":"t0"}]
        with self.assertRaises(ValueError):convert(p,q,MODEL)
        q=copy.deepcopy(w)
        q["questions"]["case.solve"]["fields"]["answer"]["nodes"]=[{"path":"","decision":"A"},{"path":"0","decision":"B"}]
        with self.assertRaises(ValueError):convert(p,q,MODEL)
        raw=json.dumps(w).replace('"schema_version":','"schema_version":"wrong","schema_version":',1).encode()
        with self.assertRaises(ValueError):convert_bytes(p,raw,MODEL)
        with self.assertRaises(ValueError):convert_bytes(p,b"{",MODEL)
