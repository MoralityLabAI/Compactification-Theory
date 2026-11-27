"""Research decisions must be feasible, scoped, and independently replayable."""
import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from Distill.core import digest, read_json, render_prompt, split_quiz, validate_response
from Distill.inky_free import config, make_plan, Journal, run, fixture_sample
from Distill.registry import load_registry
from Distill.research_cases import FAMILIES, DOCKET, compile_research, generate_research
from Distill.research_oracle import adaptive_oracle, assess, reference, grade_research
from Distill.research_curriculum import build
from Distill.review import template, validate_review

def reply(packet, value=None):
    answers=[]
    for q in packet["questions"]:
        fields={k:"offline fixture; prose is not evidence" for k in q["fields"]}
        if q["id"]=="case.solve":
            fields["answer"]=reference(packet["public_case"]) if value is None else value
        answers.append({"question_id":q["id"],"status":"answered","fields":fields})
    return {"schema_version":"distill.response.v2","packet_id":packet["packet_id"],
            "model":{"provider":"fixture","revision":"offline","run_id":"test"},"answers":answers}

def independent_value(d):
    """Set-indexed conditional-probability recursion, separate from bitmask DP."""
    hs=d["hypotheses"]
    def solve(indices,budget,available):
        mass=sum(hs[i]["weight"] for i in indices)
        terminal=[Fraction(0)]
        for k in ("A","B"):
            terminal.append(Fraction(sum(hs[i]["weight"]*(d["correct_reward"] if hs[i]["class"]==k else d["wrong_reward"]) for i in indices),mass))
        values=terminal
        for j in available:
            t=d["tests"][j]
            if t["cost"]>budget:
                continue
            groups=[tuple(i for i in indices if t["bits"][i]==b) for b in (0,1)]
            if not all(groups):
                continue
            v=-Fraction(t["cost"])
            for g in groups:
                v+=Fraction(sum(hs[i]["weight"] for i in g),mass)*solve(g,budget-t["cost"],tuple(k for k in available if k!=j))
            values.append(v)
        return max(values)
    return solve(tuple(range(len(hs))),d["budget"],tuple(range(len(d["tests"]))))

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters=load_registry()

    def test_seeded_exact_roster_and_v2(self):
        seen=set()
        count=0
        for name in FAMILIES:
            packets=generate_research(name,seed=2026112805,count=12,adapters=self.adapters)
            self.assertEqual(packets,generate_research(name,seed=2026112805,count=12,adapters=self.adapters))
            for p in packets:
                h=p["research"]["content_sha256"]
                self.assertNotIn(h,seen)
                seen.add(h)
                self.assertEqual(p["library_version"],"0.4.0")
                self.assertIn('{"question_id":"case.solve"',render_prompt(p))
                good=grade_research(p,reply(p),adapters=self.adapters)
                self.assertEqual(good["checks"][0]["exact_check"],"pass")
                self.assertEqual(good["checks"][0]["explanation_review"],"pending")
                self.assertFalse(good["training_admission"])
                self.assertEqual(grade_research(p,reply(p,{"wrong":1}),adapters=self.adapters)["checks"][0]["exact_check"],"fail")
                if name=="adaptive_evidence":
                    self.assertEqual(adaptive_oracle(p["public_case"]["data"])[0],independent_value(p["public_case"]["data"]))
                count+=1
        print("DISTILL_05_EXACT_CASES",count,len(seen))

    def test_known_decisions(self):
        d={"hypotheses":[{"weight":1,"class":"A"},{"weight":1,"class":"B"}],
           "tests":[{"id":"t0","cost":1,"bits":[0,1]}],"budget":1,"correct_reward":12,"wrong_reward":-16}
        case={"kind":"adaptive_evidence","data":d}
        self.assertEqual(reference(case)["value"],"11/1")
        sub=assess(case,{"policy":{"decision":"defer"},"value":"0/1"})
        self.assertTrue(sub["feasible"])
        self.assertFalse(sub["optimal"])
        self.assertEqual(sub["gap"],"11/1")
        bad=assess(case,{"policy":{"test":"t0","zero":{"decision":"A"},"one":{"decision":"B"}},"value":"12/1"})
        self.assertTrue(bad["feasible"])
        self.assertFalse(bad["certificate_consistent"])
        self.assertFalse(bad["optimal"])
        self.assertFalse(assess(case,{"policy":{"test":"t9","zero":{"decision":"A"},"one":{"decision":"B"}},"value":"0/1"})["feasible"])
        robust={"kind":"robust_discrimination","data":{"mechanisms":["h0","h1"],"error_bits":0,"budget":3,
                "experiments":[{"id":"c0","cost":1,"requires":[],"predictions":[0,0]},
                               {"id":"e0","cost":2,"requires":["c0"],"predictions":[0,1]}]}}
        self.assertFalse(assess(robust,{"status":"plan","experiments":["e0"],"cost":2})["feasible"])
        self.assertTrue(assess(robust,{"status":"plan","experiments":["c0","e0"],"cost":3})["optimal"])
        robust["data"]["error_bits"]=1
        self.assertTrue(assess(robust,{"status":"infeasible"})["optimal"])
        audit={"kind":"proof_obligations","data":{"budget":2,"claims":[
            {"id":"c0","state":"open","requires":[],"value":2},
            {"id":"c1","state":"checked","requires":["c0"],"value":4},
            {"id":"c2","state":"refuted","requires":[],"value":99},
            {"id":"c3","state":"checked","requires":["c2"],"value":99}],
            "jobs":[{"id":"j0","cost":2,"resolves":["c0"]}]}}
        self.assertEqual(reference(audit),{"current":[],"jobs":["j0"],"conditional":["c0","c1"],"conditional_value":6})

    def test_docket_and_tamper(self):
        p=compile_research(DOCKET,seed=0,ordinal=0,difficulty="frontier",adapters=self.adapters)
        value=reference(p["public_case"])
        self.assertEqual(value,{"ordinary_loss_millionths":3258465,"ood_loss_millionths":3104903,"learned_promotion_supported":False})
        self.assertEqual(grade_research(p,reply(p))["checks"][0]["exact_check"],"pass")
        a=copy.deepcopy(p)
        a["questions"][0]["prompt"]="tampered prompt"
        a.pop("packet_id")
        a["packet_id"]=digest(a)
        with self.assertRaises(ValueError):
            grade_research(a,reply(a))
        r=reply(p)
        r["schema_version"]="distill.response.v1"
        with self.assertRaises(ValueError):
            validate_response(p,r)

    def test_research_review(self):
        p=compile_research("research_portfolio",seed=9,ordinal=0,adapters=self.adapters)
        r=reply(p)
        record=template(p,r)
        self.assertEqual(record["schema_version"],"distill.semantic_review.v2")
        self.assertNotIn("experimental_identification",record["rows"][0]["scores"])
        self.assertIn("experimental_identification",record["rows"][2]["scores"])
        record["reviewer"]="fixture reviewer"
        for row in record["rows"]:
            row["scores"]={k:1 for k in row["scores"]}
            row["evidence"]="fixture only"
        result=validate_review(p,r,record)
        self.assertEqual(result["dimension_maxima"]["mathematical_validity"],8)
        self.assertEqual(result["dimension_maxima"]["experimental_identification"],2)
        self.assertFalse(result["training_admission"])

    def test_legacy_roster_and_research_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"legacy"
            old=make_plan(root,config("fixture",20261127,3,True))
            self.assertEqual(old["plan_id"],"7196340a619f6cd1dbe11e08b548f5ce40463937263b682d2e1ed30dc258f627")
            root=Path(d)/"research"
            plan=make_plan(root,config("fixture",2026112805,0,False,research_per_family=6,max_generation=8192,effort=.6))
            j=Journal(root)
            self.assertEqual(len(j.packets),62)
            self.assertEqual(sum(len(p["questions"]) for p in j.packets),124)
            run(j,fixture_sample,31)
            s=run(Journal(root),fixture_sample,31)
            self.assertEqual(s["counts"]["research_finished"],62)
            self.assertEqual(s["counts"]["research_exact_pass"],31)
            self.assertEqual(s["counts"]["research_certificate_consistent"],31)
            self.assertEqual(Journal(root).summary(replay=True)["counts"],s["counts"])
            print("DISTILL_05_FIXTURE",json.dumps({"requests":62,"questions":124,"decisions":31,"plan_id":plan["plan_id"]}))

    def test_curriculum(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"curriculum"
            m=build(root,counts=(48,16,32))
            all_ids=set()
            for s in m["splits"]:
                p=read_json(root/"public"/(s["split"]+".json"))
                r=read_json(root/"referee"/(s["split"]+".json"))
                self.assertEqual(digest(p),s["public_sha256"])
                self.assertEqual(digest(r),s["referee_sha256"])
                self.assertFalse(r["training_admission"])
                for x,y in zip(p["rows"],r["rows"]):
                    self.assertNotIn(x["root_id"],all_ids)
                    all_ids.add(x["root_id"])
                    self.assertEqual(digest(x),y["public_sha256"])
                    self.assertTrue(all("value" not in a for a in x["actions"]))
                    self.assertTrue(all(Fraction(a["regret"])>=0 for a in y["actions"]))
                    self.assertEqual(min(Fraction(a["regret"]) for a in y["actions"]),0)
            self.assertEqual(len(all_ids),96)
            print("DISTILL_05_CURRICULUM",json.dumps(m,sort_keys=True))
