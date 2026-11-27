"""Offline fixtures, not model evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from Distill.core import write_json
from Distill.teacher_qualification import (prepare_exam,load_exam,freeze_exam,grade_exam,
    frozen_exam,review_rows,prepare_pairs,compare_pairs)
from Distill.research_oracle import reference
from Distill.program_runner import Journal, fixture_raw, fixture_response, run

def reply(p):
    answers=[]
    for q in p["questions"]:
        fields={k:"fixture only" for k in q["fields"]}
        if q["id"]=="case.solve":
            fields["answer"]=reference(p["public_case"])
        answers.append({"question_id":q["id"],"status":"answered","fields":fields})
    return {"schema_version":"distill.response.v2","packet_id":p["packet_id"],
            "model":{"provider":"fixture","revision":"offline","run_id":"test"},"answers":answers}

class TeacherQualificationTests(unittest.TestCase):
    def test_exam_freeze_grade_and_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"exam"
            m=prepare_exam(root,"fixture")
            self.assertEqual((m["requests"],m["question_slots"]),(62,124))
            _,packets=load_exam(root)
            r=Path(d)/"responses"; r.mkdir()
            for p in packets:
                write_json(r/(p["packet_id"]+".json"),reply(p))
            last=r/(packets[-1]["packet_id"]+".json")
            last.unlink()
            with self.assertRaises(ValueError):
                freeze_exam(root,r)
            last.write_text("not json")
            freeze_exam(root,r)
            g=grade_exam(root)
            self.assertEqual(g["counts"]["format_valid"],61)
            self.assertEqual(g["counts"]["decisions"],31)
            self.assertEqual(g["counts"]["exact_pass"],31)
            self.assertFalse(g["training_admission"])
            with self.assertRaises(FileExistsError):
                freeze_exam(root,r)
            _,_,rows=frozen_exam(root)
            directory=Path(d)/"review"
            review_rows(rows,directory,create=True)
            for path in directory.iterdir():
                record=json.loads(path.read_text())
                record["reviewer"]="fixture reviewer"
                for row in record["rows"]:
                    row["scores"]={k:2 for k in row["scores"]}
                    row["evidence"]="fixture only"
                write_json(path,record)
            audit=review_rows(rows,directory)
            self.assertEqual(audit["counts"]["reviewed_slots"],122)
            self.assertFalse(audit["all_planned_slots_reviewed"])
            self.assertFalse(audit["training_admission"])
            (root/"frozen"/last.name).write_text("{}")
            with self.assertRaises(ValueError):
                grade_exam(root)

    def test_paired_worlds_and_regret(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"pairs"
            m=prepare_pairs(root,pairs_per_family=1,fixture=True)
            self.assertEqual(compare_pairs(root)["complete_pairs"],0)
            for pair in m["pairs"]:
                a,b=(Journal(root/pair["pair_id"]/arm) for arm in ("neon","inky"))
                self.assertEqual(a.index,b.index)
                self.assertEqual(a.manifest["spec"],b.manifest["spec"])
                for j in (a,b):
                    self.assertTrue(run(j,lambda obs:fixture_raw(fixture_response(obs,j.fixture_choice("reuse_first"))),32)["complete"])
            g=compare_pairs(root)
            self.assertTrue(g["fixture"])
            self.assertEqual(g["complete_pairs"],2)
            self.assertTrue(all(r["paired_utility_delta"]==0 and r["neon"]["utility_minus_reuse_first"]==0 for r in g["rows"]))
            self.assertFalse(g["training_admission"])
            self.assertEqual(g,compare_pairs(root))
