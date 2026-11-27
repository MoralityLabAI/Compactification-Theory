"""Longitudinal evidence, scoped transport, optimal decisions and replay."""
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from Distill.core import digest, read_json
from Distill.research_program import Model, Program, new_spec, validate_action, prompt
from Distill.program_oracle import evaluate_root
from Distill.program_curriculum import build
from Distill.program_runner import prepare, Journal, fixture_response, fixture_raw, external_raw, run

class ProgramTests(unittest.TestCase):
    def spec(self,family="shared_control"):
        s=new_spec(2026112820,0,family,"frontier")
        s["budget"]=14
        s["costs"]={"screen":1,"control":1,"law_audit":1,"direct_audit":3,"transport":1}
        s["offsets"]=[1,0]
        return s

    def index(self,m,bits):
        return next(i for i,w in enumerate(m.worlds) if w.bits==bits)

    def test_conflict_and_scoped_transport(self):
        s=self.spec()
        p=Program(s,self.index(Model(s),(1,1,0,1,0,1)))
        self.assertEqual(p.observe()["compatible_worlds"],64)
        p.step("screen")
        self.assertEqual(p.events[-1]["observation"],0)
        p.step("direct_audit")
        self.assertEqual(p.events[-1]["observation"],1)
        self.assertEqual(p.observe()["current_truth_masses"][0],0)
        p.step("law_audit")
        p.step("certify:1")
        self.assertIn("transport",p.model.legal(p.state))
        p.step("transport")
        e=p.events[-1]["receipt"]
        self.assertEqual(e["observation"],0)
        self.assertEqual(e["scope"],["transfer_a","transfer_b"])
        self.assertEqual(set(e["premises"]),{p.events[1]["receipt"]["receipt_id"],p.events[2]["receipt"]["receipt_id"]})
        p.step("certify:0")
        p.step("transport")
        p.step("certify:1")
        self.assertNotIn("transport",p.model.legal(p.state))
        before=p.observe()
        with self.assertRaises(ValueError):
            p.step("transport")
        self.assertEqual(p.observe(),before)
        p.step("direct_audit")
        p.step("certify:0")
        self.assertEqual(p.result()["certified"],4)
        self.assertEqual(p.result()["transports"],2)
        self.assertEqual(p.result()["net_utility"],30)

    def test_negative_law_and_no_privileged_feedback(self):
        s=self.spec()
        m=Model(s)
        a=Program(s,self.index(m,(0,0,0,0,0,0)))
        b=Program(s,self.index(m,(1,0,0,0,0,1)))
        self.assertEqual(a.observe(),b.observe())
        self.assertEqual(a.step("screen"),b.step("screen"))
        self.assertEqual(a.observe(),b.observe())
        self.assertEqual(a.step("recommend:0"),b.step("recommend:0"))
        self.assertEqual(a.observe(),b.observe())
        a.step("law_audit")
        self.assertNotIn("transport",a.model.legal(a.state))
        self.assertNotIn("world_index",json.dumps(a.observe()))
        self.assertNotIn("net_utility",json.dumps(a.observe()))
        p=Program(s,self.index(m,(1,0,0,0,0,0)))
        p.step("direct_audit"); p.step("law_audit"); p.step("certify:1")
        self.assertNotIn("transport",p.model.legal(p.state))

    def test_shifted_screen(self):
        a=self.spec()
        b=self.spec("shifted_control")
        bits=(1,1,0,0,1,0)
        p,q=Program(a,self.index(Model(a),bits)),Program(b,self.index(Model(b),bits))
        for env in (p,q):
            env.step("defer"); env.step("defer"); env.step("defer"); env.step("screen")
        self.assertEqual(p.events[-1]["observation"],1)
        self.assertEqual(q.events[-1]["observation"],0)

    def test_exact_planner_replay(self):
        for family in ("shared_control","shifted_control"):
            spec=new_spec(2026112821,1,family,"frontier")
            r=evaluate_root(spec)
            oracle=Fraction(r["oracle_value"])
            self.assertGreaterEqual(Fraction(r["transport_value"]),0)
            self.assertEqual(Fraction(r["policies"]["oracle"]["regret"]),0)
            for row in r["policies"].values():
                self.assertLessEqual(Fraction(row["utility"]),oracle)
                self.assertLessEqual(Fraction(row["credits"]),spec["budget"])
            self.assertEqual(Fraction(r["policies"]["oracle_no_transport"]["transports"]),0)

    def test_stale_response(self):
        p=Program(self.spec(),0)
        obs=p.observe()
        self.assertIn("alternative",prompt(obs))
        response=fixture_response(obs,"screen")
        self.assertEqual(validate_action(obs,response),"screen")
        p.step("screen")
        with self.assertRaises(ValueError):
            validate_action(p.observe(),response)
        with self.assertRaises(ValueError):
            validate_action(p.observe(),fixture_response(p.observe(),"certify:0"))

    def test_curriculum_split_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"cur"
            m=build(root,counts=(2,1,1,1),per_stage=2,benchmark=True)
            roots=set()
            states=set()
            for split in m["splits"]:
                for e in split["entries"]:
                    rid=e["root_id"]
                    self.assertNotIn(rid,roots)
                    roots.add(rid)
                    x=read_json(root/"public"/split["split"]/(rid+".json"))
                    y=read_json(root/"referee"/split["split"]/(rid+".json"))
                    self.assertEqual(digest(x),e["public_sha256"])
                    self.assertEqual(digest(y),e["referee_sha256"])
                    self.assertFalse(y["training_admission"])
                    for p,r in zip(x["rows"],y["rows"]):
                        self.assertNotIn(p["state_id"],states)
                        states.add(p["state_id"])
                        self.assertEqual(digest(p),r["public_sha256"])
                        self.assertTrue(all(Fraction(a["regret"])>=0 for a in r["actions"]))
                        self.assertEqual(min(Fraction(a["regret"]) for a in r["actions"]),0)
                        self.assertEqual({a["action"] for a in p["observation"]["legal_actions"]},{a["action"] for a in r["actions"]})
                        self.assertNotIn("optimal_value",json.dumps(p))
            self.assertEqual(len(roots),5)
            self.assertEqual(m["splits"][-1]["family"],"shifted_control")
            self.assertIn("shifted_eval",m["benchmark_summary"])

    def test_journal_restart_and_invalid_action(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"program"
            prepare(root,seed=1,ordinal=0,family="shared_control",difficulty="frontier",mode="fixture",world_seed=2)
            j=Journal(root)
            def sampler(obs):
                return fixture_raw(fixture_response(obs,j.fixture_choice("oracle")))
            partial=run(j,sampler,1)
            self.assertEqual(partial["accepted"],1)
            self.assertNotIn("result",partial)
            result=run(Journal(root),sampler,32)
            self.assertTrue(result["complete"])
            self.assertEqual(result,Journal(root).status())
            self.assertEqual(run(Journal(root),sampler,1),result)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"pending"
            prepare(root,seed=1,ordinal=0,family="shared_control",difficulty="standard",mode="external",world_seed=2)
            j=Journal(root)
            obs=j.claim()
            self.assertEqual(j.status()["pending"],1)
            with self.assertRaises(ValueError):
                Journal(root).claim()
            response=fixture_response(obs,"screen")
            self.assertEqual(j.finish(external_raw(response))["status"],"accepted")
            nxt=j.claim()
            self.assertNotEqual(nxt["state_id"],obs["state_id"])
            self.assertEqual(j.finish(external_raw(response))["status"],"invalid")
            self.assertEqual(Journal(root).status()["invalid"],1)
            with self.assertRaises(ValueError):
                Journal(root).claim()

    def test_concurrent_claim_and_private_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"external"
            prepare(root,seed=1,ordinal=0,family="shared_control",difficulty="standard",mode="external",world_seed=2)
            def claim(_):
                try:
                    return Journal(root).claim()["state_id"]
                except ValueError:
                    return None
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sum(x is not None for x in pool.map(claim,range(2))),1)
            ref=root/"referee.json"
            private=json.loads(ref.read_text())
            private["world_index"]=(private["world_index"]+1)%64
            ref.write_text(json.dumps(private))
            with self.assertRaises(ValueError):
                Journal(root)
