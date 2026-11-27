"""DISTILL-04: bounded, free-tier Inkling evaluation transport.

No training calls, paid model selection, automatic fallback, or answer admission.
The offline fixture never imports a provider SDK. Live sampling is opt-in.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from .core import (MAX_BYTES, VERSION, _unique, _reject_constant, canonical,
                   compile_quiz, digest, evaluate, read_json, render_prompt,
                   require, split_quiz, validate_response, write_json)
from .instance_oracle import grade_instance, exact_value
from .instances import FAMILIES, generate_instances
from .registry import load_registry

FREE_MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
TOKENIZER_MODEL = "thinkingmachines/Inkling-2-Mini"
CONTEXT = 12_288
HARD_CONTEXT = 16_384
MAX_GENERATION = 4096
EFFORT = 0.9
TEMPERATURE = 0.3
SCHEMA = "distill.inky_plan.v1"
LEVELS = ("foundation", "standard", "frontier")

def utc():
    return datetime.now(timezone.utc).isoformat()

def config(mode, seed, per_family, include_design):
    require(mode in ("fixture", "live"), "unsupported mode")
    require(type(seed) is int and 0 <= seed < 2**63, "invalid seed")
    require(type(per_family) is int and 1 <= per_family <= 96, "invalid cases per family")
    require(type(include_design) is bool, "invalid design switch")
    return {"mode": mode, "seed": seed, "per_family": per_family,
            "include_design": include_design, "model": FREE_MODEL if mode == "live" else "offline-fixture",
            "context": CONTEXT, "hard_context": HARD_CONTEXT, "max_generation": MAX_GENERATION,
            "effort": EFFORT, "temperature": TEMPERATURE,
            "purpose": "reference_evaluation", "training_admission": False}

def roster(cfg):
    expected = config(cfg["mode"], cfg["seed"], cfg["per_family"], cfg["include_design"])
    require(cfg == expected, "run configuration differs from registered free-only policy")
    adapters = load_registry()
    queues = [generate_instances(adapters[name], seed=cfg["seed"], count=cfg["per_family"],
                                difficulties=LEVELS) for name in FAMILIES]
    # Interleave families before repeating any family.
    packets = [queue[i] for i in range(cfg["per_family"]) for queue in queues]
    if cfg["include_design"]:
        packets += [split_quiz(compile_quiz(a, profile="quick", difficulty="standard"),
                               max_questions=5)[0] for a in adapters.values()]
    require(len({p["packet_id"] for p in packets}) == len(packets), "duplicate packet")
    return packets

def make_plan(root, cfg):
    packets = roster(cfg)
    plan = {"schema_version": SCHEMA, "library_version": VERSION, "config": cfg,
            "packets": [{"packet_id": p["packet_id"], "gym_id": p["gym_id"],
                         "difficulty": p["difficulty"], "question_count": len(p["questions"]),
                         "kind": "procedural" if "instance" in p else "design"}
                        for p in packets]}
    plan["plan_id"] = digest(plan)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "packets").mkdir()
    for p in packets:
        write_json(root / "packets" / (p["packet_id"] + ".json"), p)
    write_json(root / "plan.json", plan)
    return plan

def load_plan(root):
    root = Path(root)
    plan = read_json(root / "plan.json")
    require(isinstance(plan, dict) and set(plan) == {"schema_version","library_version","config","packets","plan_id"}, "invalid plan")
    require(plan["schema_version"] == SCHEMA and plan["library_version"] == VERSION, "unsupported plan")
    require(plan["plan_id"] == digest({k:v for k,v in plan.items() if k != "plan_id"}), "plan digest mismatch")
    packets = roster(plan["config"])
    expected = [{"packet_id":p["packet_id"],"gym_id":p["gym_id"],"difficulty":p["difficulty"],
                 "question_count":len(p["questions"]),"kind":"procedural" if "instance" in p else "design"} for p in packets]
    require(plan["packets"] == expected, "plan roster mismatch")
    for p in packets:
        require(read_json(root / "packets" / (p["packet_id"] + ".json")) == p, "packet does not match regenerated roster")
    return plan, packets

def visible_text(message):
    """Extract assistant text blocks only; ignore reasoning/tool blocks."""
    require(isinstance(message, dict) and message.get("role") == "assistant", "expected assistant message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    require(isinstance(content, list), "unexpected assistant content")
    texts = []
    for part in content:
        require(isinstance(part, dict), "unexpected content block")
        if part.get("type") == "text":
            require(isinstance(part.get("text"), str), "invalid text block")
            texts.append(part["text"])
    return "".join(texts)

class TinkerFree:
    """Use the free model ID for every sampling client. The base ID is metadata only."""
    def __init__(self):
        require(bool(os.environ.get("TINKER_API_KEY")), "TINKER_API_KEY is not configured")
        import tinker
        from tinker_cookbook import model_info
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        self.tinker = tinker
        self.renderer_name = model_info.get_recommended_renderer_name(TOKENIZER_MODEL)
        self.renderer = get_renderer(self.renderer_name, get_tokenizer(TOKENIZER_MODEL))
        self.sdk_version = importlib.metadata.version("tinker")
        self.cookbook_version = importlib.metadata.version("tinker-cookbook")
        self.service = tinker.ServiceClient()
        self.client = self.service.create_sampling_client(base_model=FREE_MODEL).result(timeout=180)

    def sample(self, packet):
        prompt = self.renderer.build_generation_prompt(
            [{"role":"user","content":render_prompt(packet)}], effort=EFFORT)
        n = prompt.length
        require(type(n) is int and 0 < n and n + MAX_GENERATION <= CONTEXT <= HARD_CONTEXT,
                "rendered context exceeds cap")
        start = time.perf_counter()
        result = self.client.sample(
            prompt=prompt, num_samples=1,
            sampling_params=self.tinker.SamplingParams(max_tokens=MAX_GENERATION,
                temperature=TEMPERATURE, stop=self.renderer.get_stop_sequences()),
        ).result(timeout=600)
        require(len(result.sequences) == 1, "unexpected number of sampled sequences")
        seq = result.sequences[0]
        m = len(seq.tokens)
        message, termination = self.renderer.parse_response(seq.tokens)
        return {"provider":"tinker","requested_model":FREE_MODEL,"model_revision":None,
                "renderer":self.renderer_name,"sdk_version":self.sdk_version,
                "cookbook_version":self.cookbook_version,"input_tokens":n,"output_tokens":m,
                "clean":bool(termination.is_clean),"stop_reason":str(seq.stop_reason)[:128],
                "text":visible_text(message),"elapsed_ms":round((time.perf_counter()-start)*1000),
                "utc":utc(),"billing":"free_tier_requested_not_invoice_verified"}

def fixture_sample(packet, variant="correct"):
    """Oracle fixture, never a model answer or evidence of model competence."""
    fields = []
    for q in packet["questions"]:
        a = {key:"offline fixture placeholder; independent review required" for key in q["fields"]}
        if q["id"] == "case.solve":
            a["answer"] = json.dumps(exact_value(packet["public_case"])) if variant != "wrong" else '{"wrong":1}'
        fields.append({"question_id":q["id"],"status":"answered","fields":a})
    obj = {"schema_version":"distill.response.v1","packet_id":packet["packet_id"],
           "model":{"provider":"fixture","revision":"fixture","run_id":"offline"},"answers":fields}
    return {"provider":"fixture","requested_model":"offline-fixture","model_revision":"fixture",
            "renderer":None,"sdk_version":None,"cookbook_version":None,"input_tokens":None,
            "output_tokens":None,"clean":variant!="truncated","stop_reason":"fixture",
            "text":canonical(obj).decode("ascii") if variant!="invalid" else "invalid JSON",
            "elapsed_ms":0,"utc":utc(),"billing":"no_provider_call"}

def score(packet, raw, mode):
    require(isinstance(raw, dict) and set(raw)=={"provider","requested_model","model_revision",
        "renderer","sdk_version","cookbook_version","input_tokens","output_tokens","clean",
        "stop_reason","text","elapsed_ms","utc","billing"}, "invalid transport receipt")
    require(raw["provider"] == ("tinker" if mode=="live" else "fixture"), "provider mode mismatch")
    require(raw["requested_model"] == (FREE_MODEL if mode=="live" else "offline-fixture"), "model mismatch")
    require(type(raw["clean"]) is bool and isinstance(raw["text"],str), "invalid response")
    require(type(raw["elapsed_ms"]) is int and raw["elapsed_ms"]>=0, "invalid latency")
    require(len(raw["text"].encode("utf-8")) < MAX_BYTES//2, "oversized visible response")
    cap_overrun=False
    if mode=="live":
        n,m=raw["input_tokens"],raw["output_tokens"]
        require(type(n) is int and type(m) is int and n>0 and m>=0, "invalid token accounting")
        cap_overrun = n+MAX_GENERATION>CONTEXT or m>MAX_GENERATION or n+m>HARD_CONTEXT
    result={"packet_id":packet["packet_id"],"raw_sha256":digest(raw),"mode":mode,
            "custody":"reference_only","training_admission":False,
            "format":"not_checked","exact_check":"not_checked","cap_overrun":cap_overrun,
            "semantic_review":"pending","grade":None}
    if cap_overrun:
        result["format"]="cap_overrun"
    elif not raw["clean"]:
        result["format"]="unclean_termination"
    else:
        try:
            response=json.loads(raw["text"],object_pairs_hook=_unique,parse_constant=_reject_constant)
            validate_response(packet,response)
        except (ValueError,TypeError,KeyError,RecursionError):
            result["format"]="invalid"
            return result
        result["format"]="valid"
        grade=grade_instance(packet,response) if "instance" in packet else evaluate(packet,response)
        result["grade"]=grade
        for row in grade["checks"]:
            if row["question_id"]=="case.solve":
                result["exact_check"]=row["exact_check"]
    return result

class Journal:
    """One request may be pending at a time; no automatic resubmission."""
    def __init__(self, root):
        self.root=Path(root)
        self.plan,self.packets=load_plan(root)
        self.mode=self.plan["config"]["mode"]
        self.path=self.root/"inky.sqlite3"
        self.by_id={p["packet_id"]:p for p in self.packets}

    @contextmanager
    def tx(self):
        db=sqlite3.connect(self.path,timeout=30,isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS meta (plan_id TEXT PRIMARY KEY)")
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                packet_id TEXT PRIMARY KEY, state TEXT NOT NULL,
                started TEXT NOT NULL, finished TEXT, raw TEXT, receipt TEXT, error_type TEXT)""")
            rows=db.execute("SELECT plan_id FROM meta").fetchall()
            if not rows:
                db.execute("INSERT INTO meta VALUES (?)",(self.plan["plan_id"],))
            else:
                require(rows==[(self.plan["plan_id"],)],"journal plan mismatch")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def next_packet(self):
        with self.tx() as db:
            require(not (self.root/"STOP").exists(),"STOP file present")
            rows=db.execute("SELECT packet_id,state,receipt FROM attempts").fetchall()
            require(all(r[0] in self.by_id for r in rows),"unplanned journal entry")
            require(all(r[1]=="finished" for r in rows),"pending request; reconcile before further sampling")
            require(all(not json.loads(r[2])["cap_overrun"] and json.loads(r[2])["exact_check"]!="checker_error" for r in rows),"prior cap/checker error; stop")
            seen={r[0] for r in rows}
            for p in self.packets:
                if p["packet_id"] not in seen:
                    db.execute("INSERT INTO attempts(packet_id,state,started) VALUES (?,?,?)",(p["packet_id"],"pending",utc()))
                    return p
        return None

    def finish(self, packet, raw):
        rid=packet["packet_id"]
        require(self.by_id.get(rid)==packet,"unplanned packet")
        receipt=score(packet,raw,self.mode)
        # Store raw visible response and grade atomically with terminal state.
        with self.tx() as db:
            require(db.execute("SELECT state FROM attempts WHERE packet_id=?",(rid,)).fetchone()==("pending",),"attempt not pending")
            db.execute("UPDATE attempts SET state='finished',finished=?,raw=?,receipt=? WHERE packet_id=?",
                       (utc(),canonical(raw).decode("ascii"),canonical(receipt).decode("ascii"),rid))
        return receipt

    def record_error(self, packet_id, error):
        # Keep ambiguous calls pending. Do not save exception text or credentials.
        with self.tx() as db:
            db.execute("UPDATE attempts SET error_type=? WHERE packet_id=? AND state='pending'",
                       (type(error).__name__[:96],packet_id))

    def summary(self, *, replay=False):
        with self.tx() as db:
            rows=db.execute("SELECT packet_id,state,raw,receipt FROM attempts").fetchall()
        counts=Counter()
        per_gym={}
        total_in=total_out=total_ms=0
        for rid,state,raw,receipt in rows:
            p=self.by_id[rid]
            if state!="finished":
                counts["pending"]+=1
                continue
            raw,r=json.loads(raw),json.loads(receipt)
            require(r["raw_sha256"]==digest(raw) and r["packet_id"]==rid,"stored response hash mismatch")
            if replay:
                require(r==score(p,raw,self.mode),"replay mismatch")
            counts["finished"]+=1
            counts["format_"+r["format"]]+=1
            g=per_gym.setdefault(p["gym_id"],Counter())
            g["finished"]+=1
            g["format_"+r["format"]]+=1
            if "instance" in p:
                counts["procedural_finished"]+=1
                counts["exact_"+r["exact_check"]]+=1
                g["exact_"+r["exact_check"]]+=1
            total_in+=raw["input_tokens"] or 0
            total_out+=raw["output_tokens"] or 0
            total_ms+=raw["elapsed_ms"]
        return {"schema_version":"distill.inky_summary.v1","plan_id":self.plan["plan_id"],"mode":self.mode,
                "requested_model":self.plan["config"]["model"],"planned_requests":len(self.packets),
                "counts":dict(counts),"per_gym":{k:dict(v) for k,v in sorted(per_gym.items())},
                "sample_input_tokens":total_in if self.mode=="live" else None,
                "sample_output_tokens":total_out if self.mode=="live" else None,
                "sampling_elapsed_ms":total_ms if self.mode=="live" else None,
                "replayed":replay,"training_admission":False,
                "scope":"fixture validation only" if self.mode=="fixture" else "public development evaluation; provider invoice unverified"}

def run(journal, sampler, limit):
    require(type(limit) is int and 1<=limit<=100,"invalid per-launch limit")
    for _ in range(limit):
        packet=journal.next_packet()
        if packet is None:
            break
        try:
            raw=sampler(packet)
            receipt=journal.finish(packet,raw)
        except BaseException as error:
            journal.record_error(packet["packet_id"],error)
            raise
        if receipt["cap_overrun"] or receipt["exact_check"]=="checker_error":
            break
    return journal.summary()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--out",required=True)
    a.add_argument("--mode",choices=("fixture","live"),default="fixture")
    a.add_argument("--seed",type=int,default=20261127)
    a.add_argument("--per-family",type=int,default=3)
    a.add_argument("--include-design",action="store_true")
    for name in ("run","status","replay"):
        a=sub.add_parser(name)
        a.add_argument("--campaign",required=True)
        if name=="run":
            a.add_argument("--limit",type=int,default=10)
            a.add_argument("--confirm-free-tier",action="store_true")
    args=p.parse_args()
    if args.command=="prepare":
        plan=make_plan(args.out,config(args.mode,args.seed,args.per_family,args.include_design))
        result={"plan_id":plan["plan_id"],"planned_requests":len(plan["packets"]),"mode":args.mode,"provider_calls":0}
    else:
        j=Journal(args.campaign)
        if args.command=="run":
            if j.mode=="live":
                require(args.confirm_free_tier,"confirm account free-tier eligibility before live sampling")
                require(not (j.root/"STOP").exists(),"STOP file present")
                require(not j.summary()["counts"].get("pending"),"pending request; reconcile first")
                sampler=TinkerFree().sample
            else:
                require(not args.confirm_free_tier,"fixture lane cannot become live")
                sampler=fixture_sample
            result=run(j,sampler,args.limit)
        else:
            result=j.summary(replay=args.command=="replay")
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=="__main__":
    try:
        main()
    except (ValueError,RuntimeError,ImportError) as error:
        print("Distill runner stopped: "+type(error).__name__,file=sys.stderr)
        sys.exit(2)
