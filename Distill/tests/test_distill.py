import copy
import json
import tempfile
import unittest
from pathlib import Path
from Distill.core import compile_quiz, evaluate, load_adapter, read_json, render_prompt, validate_response, write_json
from Distill.checkers import example_checkers

ROOT = Path(__file__).resolve().parents[2]
CATALOG = json.loads((ROOT / "Distill/catalog.json").read_text())

def response(packet, case_answer="3"):
    rows = []
    for q in packet["questions"]:
        fields = {field: "fixture explanation" for field in q["fields"]}
        if q["id"] == "case.solve":
            fields["answer"] = case_answer
        rows.append({"question_id": q["id"], "status": "answered", "fields": fields})
    return {"schema_version": "distill.response.v1", "packet_id": packet["packet_id"],
            "model": {"provider": "fixture", "revision": "fixture", "run_id": "test"}, "answers": rows}

class DistillTests(unittest.TestCase):
    def setUp(self):
        self.adapters = [load_adapter(ROOT / row["path"]) for row in CATALOG["adapters"]]
        self.pluto = self.adapters[0]
        self.coe = self.adapters[1]

    def test_all_adapters_deterministic_and_steps_present(self):
        for row, adapter in zip(CATALOG["adapters"], self.adapters):
            self.assertEqual(row["source"], adapter["source"])
            p = compile_quiz(adapter)
            self.assertEqual(p, compile_quiz(adapter))
            self.assertEqual(len([q for q in p["questions"] if q["category"] == "step"]), len(adapter["steps"]))
            self.assertGreater(len(p["questions"]), len(compile_quiz(adapter, profile="quick")["questions"]))
            self.assertIn(p["packet_id"], render_prompt(p))
            self.assertEqual(len(validate_response(p, response(p))), len(p["questions"]))

    def test_exact_checks_and_reference_only(self):
        for adapter, correct in ((self.pluto, "2"), (self.coe, "2")):
            p = compile_quiz(adapter)
            for value, expected in ((correct, "pass"), ("99", "fail")):
                r = evaluate(p, response(p, value), checkers=example_checkers(p))
                self.assertFalse(r["training_admission"])
                self.assertEqual(r["custody"], "reference_only")
                self.assertEqual(r["checks"][-1]["exact_check"], expected)
                self.assertEqual(r["checks"][-1]["explanation_review"], "pending")
                self.assertEqual(r["checks"][0]["exact_check"], "not_checked")

    def test_fresh_case(self):
        case = copy.deepcopy(self.pluto["example_case"])
        case["case_id"] = "new"
        case["data"]["deadline"] = 13
        p = compile_quiz(self.pluto, public_case=case)
        self.assertNotEqual(p["packet_id"], compile_quiz(self.pluto)["packet_id"])
        self.assertEqual(evaluate(p, response(p, "3"), checkers=example_checkers(p))["checks"][-1]["exact_check"], "pass")

    def test_sealed_and_label_keys_rejected(self):
        for key in ("expected_answer", "referee", "sealed_labels", "api_key"):
            c = copy.deepcopy(self.pluto["example_case"])
            c["data"]["nested"] = {key: "x"}
            with self.assertRaises(ValueError):
                compile_quiz(self.pluto, public_case=c)
        c = copy.deepcopy(self.pluto["example_case"])
        c["classification"] = "sealed_test"
        with self.assertRaises(ValueError):
            compile_quiz(self.pluto, public_case=c)

    def test_response_binding_fields_and_duplicates(self):
        p = compile_quiz(self.pluto)
        for change in ("id", "duplicate", "missing", "extra"):
            r = response(p)
            if change == "id":
                r["packet_id"] = "bad"
            elif change == "duplicate":
                r["answers"][1] = r["answers"][0]
            elif change == "missing":
                r["answers"].pop()
            else:
                r["answers"][0]["fields"]["extra"] = "bad"
            with self.assertRaises(ValueError):
                validate_response(p, r)

    def test_abstention_and_tampering(self):
        p = compile_quiz(self.pluto)
        r = response(p)
        r["answers"][0].update(status="abstain", fields={"reason": "insufficient premises"})
        self.assertEqual(evaluate(p, r)["checks"][0]["exact_check"], "abstained")
        p["context"] = "changed"
        with self.assertRaises(ValueError):
            validate_response(p, r)

    def test_checker_failure_is_scoped(self):
        p = compile_quiz(self.pluto)
        def broken(*args):
            raise RuntimeError("private diagnostic")
        r = evaluate(p, response(p), checkers={"case.solve": broken})
        self.assertEqual(r["checks"][-1]["exact_check"], "checker_error")
        self.assertNotIn("private diagnostic", json.dumps(r))

    def test_exclusive_files_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.json"
            write_json(path, {"x": 1})
            self.assertEqual(read_json(path), {"x": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {})
            path.write_text('{"x": 1, "x": 2}')
            with self.assertRaises(ValueError):
                read_json(path)

if __name__ == "__main__":
    unittest.main()
