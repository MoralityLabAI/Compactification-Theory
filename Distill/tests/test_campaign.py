import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from Distill.core import compile_quiz, digest, render_prompt, split_quiz, validate_response
from Distill.registry import load_registry, inventory, REGISTRY, ROOT
from Distill.batch import dollars, rates, cost, plan_campaign, load_plan
from Distill.ledger import Ledger
from Distill.native import grade_native

def answer(p, value="2"):
    answers = []
    for q in p["questions"]:
        fields = {k: "fixture only" for k in q["fields"]}
        if q["id"] == "case.solve":
            fields["answer"] = value
        answers.append({"question_id":q["id"], "status":"answered", "fields":fields})
    return {"schema_version":"distill.response.v1", "packet_id":p["packet_id"],
            "model":{"provider":"fixture","revision":"test","run_id":"offline"}, "answers":answers}

class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = load_registry()

    def test_cross_repo_registry_all_difficulties_and_sections(self):
        self.assertEqual(len(self.adapters), 14)
        self.assertEqual(len({a["source"]["repository"] for a in self.adapters.values()}), 4)
        inv = inventory(self.adapters)
        counts = {d:sum(row["question_counts"][d] for row in inv["rows"]) for d in ("foundation","standard","frontier")}
        print("DISTILL_02_COUNTS", counts)
        ids = set()
        for a in self.adapters.values():
            for d in counts:
                p = compile_quiz(a, difficulty=d)
                self.assertNotIn(p["packet_id"], ids)
                ids.add(p["packet_id"])
                self.assertEqual(p, compile_quiz(a, difficulty=d))
                sections = split_quiz(p, max_questions=5)
                self.assertEqual([q for s in sections for q in s["questions"]], p["questions"])
                for i,s in enumerate(sections,1):
                    self.assertEqual(s["section"]["index"], i)
                    self.assertEqual(s["section"]["parent_packet_id"], p["packet_id"])
                    self.assertLessEqual(len(s["questions"]), 5)
                    self.assertIn(s["packet_id"], render_prompt(s))
                    validate_response(s, answer(s))

    def test_v01_packet_compatibility(self):
        p = compile_quiz(self.adapters["IRC-PG-01"])
        p.pop("difficulty")
        p.pop("difficulty_instruction")
        p["library_version"] = "0.1.0"
        p.pop("packet_id")
        p["packet_id"] = digest(p)
        validate_response(p, answer(p))

    def test_bad_difficulty_and_section_binding(self):
        a = self.adapters["COE-01"]
        with self.assertRaises(ValueError):
            compile_quiz(a, difficulty="wrong")
        s = split_quiz(compile_quiz(a))[0]
        s["section"]["index"] = 0
        s.pop("packet_id")
        s["packet_id"] = digest(s)
        with self.assertRaises(ValueError):
            render_prompt(s)

    def test_native_exact_check_and_parity(self):
        for key in ("IRC-PG-01","COE-01"):
            p = compile_quiz(self.adapters[key])
            for value,status in (("2","pass"),("99","fail")):
                r = grade_native(p, answer(p,value))
                row = next(x for x in r["checks"] if x["question_id"]=="case.solve")
                self.assertEqual(row["exact_check"],status)
                self.assertEqual(row["explanation_review"],"pending")
                self.assertFalse(r["training_admission"])
                self.assertEqual(len(r["native_verifier"]["blob"]),40)
        with self.assertRaises(ValueError):
            p = compile_quiz(self.adapters["ARG-01"])
            grade_native(p,answer(p))

    def test_native_fresh_cases(self):
        for name,key,value,expected in (("IRC-PG-01","deadline",13,"3"),("COE-01","vertices",6,"1")):
            a=self.adapters[name]
            case=copy.deepcopy(a["example_case"])
            case["data"][key]=value
            p=compile_quiz(a,public_case=case)
            section=next(s for s in split_quiz(p) if any(q["id"]=="case.solve" for q in s["questions"]))
            r=grade_native(section,answer(section,expected))
            self.assertEqual(r["checks"][-1]["exact_check"],"pass")

    def test_integer_costs(self):
        self.assertEqual(cost(rates(dollars("4"),dollars("20")),6000,5000),124000)
        self.assertEqual(cost(rates(1,1),1,1),1)
        for n in ("NaN","Infinity","-1","0.0000001"):
            with self.assertRaises(ValueError):
                dollars(n)

    def _plan(self, root, budget="1", reserve="0", **kwargs):
        return plan_campaign(root, prices=rates(dollars("4"),dollars("20")),
                             budget_microusd=dollars(budget),reserve_microusd=dollars(reserve),
                             **kwargs)

    def test_plan_cap_and_round_robin(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"run"
            p=self._plan(root,budget="1",reserve="0.10")
            self.assertEqual(len(p["jobs"]),7)
            self.assertEqual(p["authorized_max_microusd"],868000)
            self.assertEqual(len({j["gym_id"] for j in p["jobs"]}),7)
            self.assertGreater(p["omitted_count"],0)
            self.assertEqual(load_plan(root)[0],p)
            with self.assertRaises(FileExistsError):
                self._plan(root)

    def test_plan_full_roster(self):
        with tempfile.TemporaryDirectory() as d:
            p=self._plan(Path(d)/"run",budget="200",reserve="10")
            self.assertEqual(p["omitted_count"],0)
            self.assertEqual(len({(j["gym_id"],j["difficulty"]) for j in p["jobs"]}),42)
            self.assertLessEqual(p["authorized_max_microusd"],190000000)
            print("DISTILL_02_PLAN",len(p["jobs"]),sum(j["question_count"] for j in p["jobs"]),p["authorized_max_microusd"])

    def test_ledger_duplicate_pending_settlement_and_overrun(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"run"
            p=self._plan(root)
            first,second,third=[j["request_id"] for j in p["jobs"][:3]]
            ledger=Ledger(root)
            self.assertEqual(ledger.status()["reserved_microusd"],0)
            with self.assertRaises(ValueError):
                ledger.reserve(first,counted_input_tokens=6001)
            ledger.reserve(first,counted_input_tokens=2000)
            self.assertEqual(Ledger(root).status()["reserved_microusd"],124000)
            with self.assertRaises(ValueError):
                ledger.reserve(first,counted_input_tokens=2000)
            s=ledger.settle(first,input_tokens=2000,output_tokens=3000,provider_receipt="actual-id-1")
            self.assertEqual(s["billed_at_supplied_rates_microusd"],68000)
            self.assertEqual(s["reserved_microusd"],0)
            ledger.reserve(second,counted_input_tokens=2000)
            s=ledger.settle(second,input_tokens=2000,output_tokens=5001,provider_receipt="actual-id-2")
            self.assertTrue(s["frozen"])
            with self.assertRaises(ValueError):
                ledger.reserve(third,counted_input_tokens=2000)

    def test_ledger_concurrent_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"run"
            p=self._plan(root)
            rid=p["jobs"][0]["request_id"]
            def attempt(_):
                try:
                    Ledger(root).reserve(rid,counted_input_tokens=2000)
                    return True
                except ValueError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                result=list(pool.map(attempt,range(2)))
            self.assertEqual(sum(result),1)
            self.assertEqual(Ledger(root).status()["reserved_microusd"],124000)

    def test_tampered_packet_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"run"
            p=self._plan(root)
            path=root/"packets"/(p["jobs"][0]["request_id"]+".json")
            path.write_text("{}")
            with self.assertRaises(ValueError):
                Ledger(root)
