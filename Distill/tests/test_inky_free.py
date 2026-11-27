import copy
import json
import os
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from Distill.core import read_json,write_json,digest,validate_response
from Distill.inky_free import (FREE_MODEL,TOKENIZER_MODEL,MAX_GENERATION,CONTEXT,
    TinkerFree,Journal,config,make_plan,load_plan,fixture_sample,score,run,visible_text)
from Distill.review import RUBRIC,template,validate_review

class InkyTests(unittest.TestCase):
    def plan(self,d,mode="fixture",count=1,design=False):
        path=Path(d)/"run"
        make_plan(path,config(mode,20261127,count,design))
        return path

    def test_full_offline_campaign_and_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.plan(d,count=3,design=True)
            j=Journal(root)
            self.assertEqual(len(j.packets),55)
            self.assertEqual(sum(len(p["questions"]) for p in j.packets),215)
            self.assertEqual(len({p["gym_id"] for p in j.packets}),25)
            self.assertEqual(run(j,fixture_sample,10)["counts"]["finished"],10)
            self.assertEqual(run(Journal(root),fixture_sample,100)["counts"]["finished"],55)
            r=Journal(root).summary(replay=True)
            self.assertEqual(r["counts"]["format_valid"],55)
            self.assertEqual(r["counts"]["exact_pass"],30)
            self.assertEqual(r["counts"]["procedural_finished"],30)
            self.assertIsNone(r["sample_input_tokens"])
            self.assertFalse(r["training_admission"])
            self.assertEqual(run(Journal(root),fixture_sample,1)["counts"]["finished"],55)
            print("DISTILL_04_FIXTURE",json.dumps(r,sort_keys=True))

    def test_invalid_truncated_wrong_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(self.plan(d))
            for variant,status,exact in (("invalid","invalid","not_checked"),("truncated","unclean_termination","not_checked"),("wrong","valid","fail")):
                p=j.next_packet()
                r=j.finish(p,fixture_sample(p,variant))
                self.assertEqual(r["format"],status)
                self.assertEqual(r["exact_check"],exact)
            p=j.next_packet()
            raw=fixture_sample(p)
            raw["text"]=raw["text"].replace('"schema_version":','"schema_version":"duplicate","schema_version":',1)
            self.assertEqual(j.finish(p,raw)["format"],"invalid")
            self.assertEqual(j.summary(replay=True)["counts"]["finished"],4)

    def test_pending_and_stop_are_durable(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.plan(d)
            j=Journal(root)
            def broken(packet):
                raise TimeoutError("never persist private exception detail")
            with self.assertRaises(TimeoutError):
                run(j,broken,1)
            self.assertEqual(Journal(root).summary()["counts"]["pending"],1)
            with self.assertRaises(ValueError):
                run(Journal(root),fixture_sample,1)
            with j.tx() as db:
                r=db.execute("SELECT state,error_type,raw FROM attempts").fetchone()
            self.assertEqual(r,("pending","TimeoutError",None))
        with tempfile.TemporaryDirectory() as d:
            root=self.plan(d)
            (root/"STOP").touch()
            with self.assertRaises(ValueError):
                run(Journal(root),fixture_sample,1)
            self.assertEqual(Journal(root).summary()["counts"],{})

    def test_duplicate_concurrent_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.plan(d)
            def attempt(_):
                try:
                    return Journal(root).next_packet()["packet_id"]
                except ValueError:
                    return None
            with ThreadPoolExecutor(max_workers=2) as pool:
                out=list(pool.map(attempt,range(2)))
            self.assertEqual(sum(x is not None for x in out),1)
            self.assertEqual(Journal(root).summary()["counts"]["pending"],1)

    def test_plan_and_mode_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.plan(d)
            j=Journal(root)
            p=j.next_packet()
            raw=fixture_sample(p)
            raw["requested_model"]=FREE_MODEL
            with self.assertRaises(ValueError):
                j.finish(p,raw)
            with j.tx() as db:
                self.assertIsNotNone(db.execute("SELECT raw FROM attempts").fetchone()[0])
            self.assertEqual(j.summary()["counts"]["pending"],1)
        with tempfile.TemporaryDirectory() as d:
            root=self.plan(d)
            plan=read_json(root/"plan.json")
            plan["config"]["model"]=TOKENIZER_MODEL
            plan.pop("plan_id")
            plan["plan_id"]=digest(plan)
            (root/"plan.json").unlink()
            write_json(root/"plan.json",plan)
            with self.assertRaises(ValueError):
                load_plan(root)

    def test_cap_overrun_stops(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(self.plan(d,mode="live"))
            p=j.next_packet()
            raw=fixture_sample(p)
            raw.update(provider="tinker",requested_model=FREE_MODEL,model_revision=None,
                       input_tokens=100,output_tokens=MAX_GENERATION+1,
                       billing="free_tier_requested_not_invoice_verified")
            self.assertTrue(j.finish(p,raw)["cap_overrun"])
            with self.assertRaises(ValueError):
                j.next_packet()

    def test_tinker_mock_uses_only_free_id_and_bounds(self):
        calls=[]
        class Future:
            def __init__(self,value): self.value=value
            def result(self,timeout=None): return self.value
        class Renderer:
            def build_generation_prompt(self,messages,effort):
                calls.append(("render",messages,effort))
                return types.SimpleNamespace(length=100)
            def get_stop_sequences(self): return [7]
            def parse_response(self,tokens):
                return {"role":"assistant","content":[{"type":"thinking","thinking":"private"},
                                                       {"type":"text","text":"visible answer"}]},types.SimpleNamespace(is_clean=True)
        class Sampler:
            def sample(self,**kwargs):
                calls.append(("sample",kwargs))
                return Future(types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=[1,2,3],stop_reason="stop")]))
        class Service:
            def create_sampling_client(self,**kwargs):
                calls.append(("create",kwargs))
                return Future(Sampler())
        tinker=types.ModuleType("tinker")
        tinker.ServiceClient=Service
        tinker.SamplingParams=lambda **kw:kw
        cookbook=types.ModuleType("tinker_cookbook")
        cookbook.__path__=[]
        cookbook.model_info=types.SimpleNamespace(get_recommended_renderer_name=lambda m:"inkling-test")
        renderers=types.ModuleType("tinker_cookbook.renderers")
        renderers.get_renderer=lambda name,tok:Renderer()
        tokenizer=types.ModuleType("tinker_cookbook.tokenizer_utils")
        tokenizer.get_tokenizer=lambda m:object()
        with tempfile.TemporaryDirectory() as d:
            j=Journal(self.plan(d,mode="live"))
            with patch.dict(os.environ,{"TINKER_API_KEY":"mock-key"}),patch.dict(sys.modules,{
                "tinker":tinker,"tinker_cookbook":cookbook,
                "tinker_cookbook.renderers":renderers,"tinker_cookbook.tokenizer_utils":tokenizer}),patch("importlib.metadata.version",return_value="mock"):
                runner=TinkerFree()
                raw=runner.sample(j.packets[0])
                self.assertEqual(raw["text"],"visible answer")
                self.assertEqual(raw["requested_model"],FREE_MODEL)
                self.assertEqual(calls[0],("create",{"base_model":FREE_MODEL}))
                self.assertEqual(calls[-1][1]["sampling_params"]["max_tokens"],MAX_GENERATION)
                self.assertEqual(calls[-1][1]["num_samples"],1)
                old=len(calls)
                runner.renderer.build_generation_prompt=lambda *args,**kwargs:types.SimpleNamespace(length=CONTEXT)
                with self.assertRaises(ValueError):
                    runner.sample(j.packets[0])
                self.assertEqual(len(calls),old)
        self.assertEqual(visible_text({"role":"assistant","content":"hello"}),"hello")
        self.assertEqual(visible_text({"role":"assistant","content":"<|message_model|><|content_thinking|>private reasoning"}),"")
        self.assertEqual(visible_text({"role":"assistant","content":"<|message_model|><|content_thinking|>private<|content_text|>{\"ok\":1}<|end|>"}),"{\"ok\":1}")

    def test_review_binding(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(self.plan(d))
            p=j.packets[0]
            response=json.loads(fixture_sample(p)["text"])
            record=template(p,response)
            record["reviewer"]="test reviewer (fixture)"
            for row in record["rows"]:
                row["scores"]={k:1 for k in RUBRIC}
                row["evidence"]="fixture exercise, no mathematical claim"
            result=validate_review(p,response,record)
            self.assertEqual(result["reviewed_questions"],3)
            self.assertEqual(result["max_per_dimension"],6)
            self.assertFalse(result["training_admission"])
            bad=copy.deepcopy(record)
            bad["rows"][0]["scores"]["checkability"]=True
            with self.assertRaises(ValueError):
                validate_review(p,response,bad)
            bad=copy.deepcopy(record)
            bad["response_sha256"]="0"*64
            with self.assertRaises(ValueError):
                validate_review(p,response,bad)

    def test_full_design_roster_and_generation_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"full-design"
            cfg=config("fixture",20261127,0,False,design_mode="full",design_profile="full",
                       design_difficulties=("foundation","standard","frontier"),section_size=5,
                       max_generation=8192,effort=0.5)
            plan=make_plan(root,cfg)
            j=Journal(root)
            self.assertEqual(len(j.packets),547)
            self.assertEqual(sum(len(p["questions"]) for p in j.packets),2585)
            self.assertEqual(len({p["gym_id"] for p in j.packets}),25)
            self.assertEqual({p["difficulty"] for p in j.packets},{"foundation","standard","frontier"})
            self.assertEqual(len({p["gym_id"] for p in j.packets[:25]}),25)
            self.assertEqual(run(j,fixture_sample,10)["counts"]["format_valid"],10)
            self.assertEqual(Journal(root).summary(replay=True)["counts"]["finished"],10)
            self.assertEqual(plan["config"]["max_generation"],8192)
            self.assertEqual(plan["config"]["effort"],0.5)
        with self.assertRaises(ValueError):
            config("live",20261127,0,False)
        with self.assertRaises(ValueError):
            config("live",20261127,3,False,max_generation=CONTEXT)
        with self.assertRaises(ValueError):
            config("live",20261127,3,False,effort=1.0)
        with self.assertRaises(ValueError):
            config("live",20261127,3,False,design_mode="full",section_size=6)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"live-tuned"
            make_plan(root,config("live",20261127,1,False,max_generation=8192,effort=0.5))
            j=Journal(root)
            p=j.next_packet()
            raw=fixture_sample(p)
            raw.update(provider="tinker",requested_model=FREE_MODEL,model_revision=None,
                       input_tokens=1100,output_tokens=8192,
                       billing="free_tier_requested_not_invoice_verified")
            self.assertFalse(j.finish(p,raw)["cap_overrun"])
            p2=j.next_packet()
            raw2=fixture_sample(p2)
            raw2.update(provider="tinker",requested_model=FREE_MODEL,model_revision=None,
                        input_tokens=1100,output_tokens=8193,
                        billing="free_tier_requested_not_invoice_verified")
            self.assertTrue(j.finish(p2,raw2)["cap_overrun"])
