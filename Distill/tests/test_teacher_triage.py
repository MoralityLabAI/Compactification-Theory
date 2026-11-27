"""Teacher triage tests use synthetic fixture answers."""
import copy
import unittest
from Distill.inky_free import config, roster
from Distill.research_oracle import reference
from Distill.teacher_triage import compare_rows, shortlist, holdout_binding

def reply(p):
    answers=[]
    for q in p["questions"]:
        fields={k:"fixture only" for k in q["fields"]}
        if q["id"]=="case.solve":
            fields["answer"]=reference(p["public_case"])
        answers.append({"question_id":q["id"],"status":"answered","fields":fields})
    return {"schema_version":"distill.response.v2","packet_id":p["packet_id"],
            "model":{"provider":"fixture","revision":"offline","run_id":"test"},"answers":answers}

class TriageTests(unittest.TestCase):
    def test_complete_matched_roster_and_selection(self):
        packets=roster(config("fixture",2026112805,0,False,research_per_family=6))
        rows=[(p,reply(p)) for p in packets]
        report=compare_rows(rows,rows)
        self.assertEqual(report["aggregate"]["neon"]["exact_pass"],31)
        self.assertEqual(report["aggregate"]["neon"]["planned_slots"],124)
        self.assertEqual(shortlist(report,12),[])
        changed=copy.deepcopy(rows)
        changed[0]=(changed[0][0],None)
        report=compare_rows(rows,changed)
        self.assertEqual(report["aggregate"]["inky"]["exact_pass"],30)
        picks=shortlist(report,12)
        self.assertEqual(len(picks),1)
        self.assertEqual(picks[0]["priority"],2)
        self.assertEqual(len(picks[0]["sections"]),2)
        with self.assertRaises(ValueError):
            compare_rows(rows,changed[:-1])
        with self.assertRaises(ValueError):
            holdout_binding(packets,packets)
        fresh=roster(config("fixture",2026112991,0,False,research_per_family=6))
        hold=holdout_binding(packets,fresh)
        self.assertEqual(len(hold["eligible_packet_ids"]),60)
        self.assertEqual(len(hold["excluded_fixed_docket"]),2)
