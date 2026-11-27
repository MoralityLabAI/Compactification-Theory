import copy
import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter
from Distill.core import digest,split_quiz,render_prompt,read_json
from Distill.registry import load_registry
from Distill.batch import dollars,rates,plan_campaign,load_plan
from Distill.instances import FAMILIES,compile_instance,generate_instances
from Distill.instance_oracle import exact_value,grade_instance

def answer(packet,value):
    rows=[]
    for q in packet["questions"]:
        fields={k:"offline fixture placeholder" for k in q["fields"]}
        if q["id"]=="case.solve":
            fields["answer"]=json.dumps(value)
        rows.append({"question_id":q["id"],"status":"answered","fields":fields})
    return {"schema_version":"distill.response.v1","packet_id":packet["packet_id"],
            "model":{"provider":"fixture","revision":"fixture","run_id":"offline"},"answers":rows}

class ProceduralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters=load_registry()

    def test_seeded_families(self):
        self.assertEqual(len(FAMILIES),10)
        totals=Counter()
        seen=set()
        for name in FAMILIES:
            a=self.adapters[name]
            parents=generate_instances(a,seed=912027,count=32)
            self.assertEqual(parents,generate_instances(a,seed=912027,count=32))
            for p in parents:
                h=p["instance"]["content_sha256"]
                self.assertNotIn(h,seen)
                seen.add(h)
                v=exact_value(p["public_case"])
                self.assertEqual(p["adapter_sha256"],digest(a))
                self.assertEqual(p["purpose"],"reference_evaluation")
                self.assertNotIn("expected_answer",render_prompt(p))
                r=grade_instance(p,answer(p,v),adapter=a)
                self.assertEqual(r["checks"][0]["exact_check"],"pass")
                self.assertEqual(r["checks"][0]["explanation_review"],"pending")
                self.assertFalse(r["training_admission"])
                self.assertEqual(grade_instance(p,answer(p,{"wrong":1}),adapter=a)["checks"][0]["exact_check"],"fail")
                totals[p["public_case"]["kind"]]+=1
        print("DISTILL_03_VERIFIED_CASES",len(seen),dict(totals))

    def test_known_mathematics(self):
        fixtures=[
            ("equal_weight_fronts",{"fronts":5,"deadline":10,"command_ticks":2,"switch_ticks":2},3),
            ("graph_observer",{"vertices":3,"edges":[[0,1],[1,2],[0,2]]},2),
            ("arithmetic_trace",{"initial":-2,"instructions":[{"op":"if_even_add","operand":3},{"op":"multiply","operand":-2},{"op":"subtract","operand":4}]},[-2,1,-2,-6]),
            ("modular_trace",{"modulus":7,"initial":2,"operations":[[3,1],[2,4]]},[2,0,4]),
            ("formal_reciprocal",{"coefficients":[1,-1,0,0]},[1,1,1,1]),
            ("lagrange_identity",{"u":[1,0,0],"v":[0,2,0]},4),
            ("prefix_recurrence",{"input":[1,2,3],"parents":[[],[0],[0,1]],"modulus":7,"initial":[5,5,5],"changed":[1]},{"depth":3,"values":[1,3,0],"affected":[1,2]}),
            ("regime_counts",{"counts":[2,3]},{"action":1,"probability":"3/5"}),
            ("anchored_parity",{"vertices":3,"edges":[[0,1,1]],"anchor":[0,1],"target":1},0),
            ("anchored_parity",{"vertices":3,"edges":[[0,1,1]],"anchor":[0,1],"target":2},"abstain"),
            ("gf2_query",{"n":3,"equations":[[1,1],[2,0]],"query_mask":3},1),
            ("gf2_query",{"n":3,"equations":[[1,1],[2,0]],"query_mask":4},"abstain"),
        ]
        for kind,data,expected in fixtures:
            self.assertEqual(exact_value({"kind":kind,"data":data}),expected)

    def test_tamper_and_strict_json(self):
        a=self.adapters["HG-07"]
        p=compile_instance(a,seed=77,ordinal=0)
        v=exact_value(p["public_case"])
        bad=copy.deepcopy(p)
        bad["questions"][0]["prompt"]="different question"
        bad.pop("packet_id")
        bad["packet_id"]=digest(bad)
        with self.assertRaises(ValueError):
            grade_instance(bad,answer(bad,v),adapter=a)
        a=self.adapters["IRC-PG-01"]
        p=compile_instance(a,seed=77,ordinal=0)
        s=split_quiz(p,max_questions=1)[0]
        v=exact_value(s["public_case"])
        self.assertEqual(grade_instance(s,answer(s,v),adapter=a)["checks"][0]["exact_check"],"pass")
        wrong=answer(s,v)
        wrong["answers"][0]["fields"]["answer"]="true"
        self.assertEqual(grade_instance(s,wrong,adapter=a)["checks"][0]["exact_check"],"fail")

    def test_budgeted_seeded_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"run"
            p=plan_campaign(root,prices=rates(dollars("4"),dollars("20")),
                budget_microusd=dollars("200"),reserve_microusd=dollars("10"),
                instances_per_gym=96,instance_seed=20261127)
            self.assertEqual(p["instance_candidate_sections"],960)
            self.assertLessEqual(p["authorized_max_microusd"],190000000)
            self.assertEqual(p,load_plan(root)[0])
            self.assertEqual(len(set(j["request_id"] for j in p["jobs"])),len(p["jobs"]))
            cases=set()
            for job in p["jobs"]:
                q=read_json(root/"packets"/(job["request_id"]+".json"))
                if "instance" in q:
                    h=q["instance"]["content_sha256"]
                    self.assertNotIn(h,cases)
                    cases.add(h)
                    v=exact_value(q["public_case"])
                    r=grade_instance(q,answer(q,v),adapter=self.adapters[q["gym_id"]])
                    self.assertEqual(r["checks"][0]["exact_check"],"pass")
            self.assertEqual(len(cases),p["instance_selected_sections"])
            print("DISTILL_03_CAMPAIGN",len(p["jobs"]),sum(j["question_count"] for j in p["jobs"]),len(cases),p["omitted_count"],p["authorized_max_microusd"])
