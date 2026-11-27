"""Restart-safe longitudinal research runner; fixture, external, or free Inky.

The public observation is the only model input. Keep local run directories
private: the host manifest/referee files and journal are not model inputs.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import secrets
import sqlite3
import sys
from .core import canonical, digest, read_json, write_json, require, _unique, _reject_constant
from .instances import Rng
from .research_program import Model, Program, ACTION_SCHEMA, new_spec, prompt, validate_action
from .program_oracle import Oracle, Baseline
from .inky_free import TinkerFree, FREE_MODEL, CONTEXT, HARD_CONTEXT

SCHEMA="distill.program_run.v1"
CALIBRATION_SCHEMA="distill.program_run.v2"
MAX_GEN=4096
EFFORT=.6
MAX_TEXT=100000
CALIBRATION_PROFILES=((4096,.6),)+tuple((cap,e) for cap in (4096,8192) for e in (.15,.25,.4))

def policy_schema(mode,max_generation,effort):
    require(type(max_generation) is int and type(effort) in (int,float) and (max_generation,effort) in CALIBRATION_PROFILES,"unregistered sampling profile")
    if (max_generation,effort)==(MAX_GEN,EFFORT):
        return SCHEMA
    require(mode in ("fixture","live"),"calibration profile is free-only or fixture")
    return CALIBRATION_SCHEMA

def prepare(out,*,seed,ordinal,family,difficulty,mode,world_seed=None,max_generation=MAX_GEN,effort=EFFORT):
    require(mode in ("fixture","external","live"),"invalid program mode")
    require(world_seed is None or type(world_seed) is int and 0<=world_seed<2**63,"invalid world seed")
    schema=policy_schema(mode,max_generation,effort)
    spec=new_spec(seed,ordinal,family,difficulty)
    model=Model(spec)
    if world_seed is None:
        world_seed=secrets.randbits(63)
    r=Rng(["distill.program_referee.v1",world_seed,digest(spec)])
    draw=r.below(model.mass(model.full_mask))
    index=0
    while draw>=model.worlds[index].mass:
        draw-=model.worlds[index].mass
        index+=1
    private={"schema_version":"distill.program_referee.v1","world_seed":world_seed,"world_index":index,
             "nonce":secrets.token_hex(32)}
    manifest={"schema_version":schema,"mode":mode,"spec":spec,
              "requested_model":FREE_MODEL if mode=="live" else mode,
              "max_generation":max_generation,"effort":effort,"referee_sha256":digest(private),
              "training_admission":False}
    manifest["run_id"]=digest(manifest)
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/"manifest.json",manifest)
    write_json(root/"referee.json",private)
    return {"run_id":manifest["run_id"],"program_id":digest(spec),"mode":mode,
            "provider_calls":0,"training_admission":False}

def load(root):
    manifest=read_json(root/"manifest.json")
    private=read_json(root/"referee.json")
    require(manifest["schema_version"]==policy_schema(manifest["mode"],manifest["max_generation"],manifest["effort"]) and manifest["run_id"]==digest({k:v for k,v in manifest.items() if k!="run_id"}),"manifest mismatch")
    require(manifest["training_admission"] is False and manifest["referee_sha256"]==digest(private),"referee binding mismatch")
    require(manifest["mode"] in ("fixture","external","live") and
            manifest["requested_model"]==(FREE_MODEL if manifest["mode"]=="live" else manifest["mode"])
            ,"runner policy mismatch")
    require(private["schema_version"]=="distill.program_referee.v1","invalid referee")
    model=Model(manifest["spec"])
    r=Rng(["distill.program_referee.v1",private["world_seed"],digest(manifest["spec"])])
    draw=r.below(model.mass(model.full_mask))
    index=0
    while draw>=model.worlds[index].mass:
        draw-=model.worlds[index].mass
        index+=1
    require(index==private["world_index"],"referee draw mismatch")
    return manifest,index

def fixture_response(obs,action):
    return {"schema_version":ACTION_SCHEMA,"state_id":obs["state_id"],"action":action,
            "agenda":"Offline policy fixture; plan across the remaining projects.",
            "evidence":"Use only the public finite-model observation.",
            "decision_basis":"Fixture policy selection, not a model explanation.",
            "revision_rule":"Recompute after the next observation."}

def fixture_raw(response):
    return {"provider":"fixture","requested_model":"fixture","clean":True,
            "text":canonical(response).decode("ascii"),"billing":"no_provider_call"}

def external_raw(response):
    return {"provider":"external","requested_model":"external","clean":True,
            "text":canonical(response).decode("ascii"),"billing":"external_unverified"}

def decode(raw,mode,obs,*,max_generation=MAX_GEN):
    require(isinstance(raw,dict) and isinstance(raw.get("text"),str),"invalid raw response")
    require(len(raw["text"].encode("utf-8"))<=MAX_TEXT,"response exceeds text cap")
    require(raw.get("requested_model")==(FREE_MODEL if mode=="live" else mode),"wrong provider model")
    if mode=="live":
        require(raw.get("provider")=="tinker","wrong live provider")
        n,m=raw.get("input_tokens"),raw.get("output_tokens")
        require(type(n) is int and type(m) is int and n>0 and m>=0 and
                n+max_generation<=CONTEXT<=HARD_CONTEXT and m<=max_generation and n+m<=HARD_CONTEXT,
                "token cap violation")
    else:
        require(raw.get("provider")==mode,"wrong offline provider")
    require(raw.get("clean") is True,"unclean termination")
    response=json.loads(raw["text"],object_pairs_hook=_unique,parse_constant=_reject_constant)
    return response,validate_action(obs,response)

class Journal:
    def __init__(self,root):
        self.root=Path(root)
        self.manifest,self.index=load(self.root)
        self.path=self.root/"program.sqlite3"

    @contextmanager
    def tx(self):
        db=sqlite3.connect(self.path,timeout=30,isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS meta (run_id TEXT PRIMARY KEY)")
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                seq INTEGER PRIMARY KEY, state TEXT NOT NULL, observation TEXT NOT NULL,
                raw TEXT, raw_sha256 TEXT, response TEXT, event TEXT, error_type TEXT)""")
            rows=db.execute("SELECT run_id FROM meta").fetchall()
            if not rows:
                db.execute("INSERT INTO meta VALUES (?)",(self.manifest["run_id"],))
            else:
                require(rows==[(self.manifest["run_id"],)],"journal identity mismatch")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def reconstruct(self,db):
        env=Program(self.manifest["spec"],self.index)
        rows=db.execute("SELECT seq,state,observation,raw,raw_sha256,response,event,error_type FROM attempts ORDER BY seq").fetchall()
        for i,row in enumerate(rows):
            seq,state,observation,raw,rhash,response,event,error=row
            require(seq==i and env.state.stage<4,"journal sequence mismatch")
            obs=json.loads(observation)
            require(obs==env.observe(),"observation replay mismatch")
            if raw is not None:
                require(rhash==digest(json.loads(raw)),"raw response digest mismatch")
            if state=="accepted":
                require(raw is not None and response is not None and event is not None,"incomplete accepted row")
                r,action=decode(json.loads(raw),self.manifest["mode"],obs,max_generation=self.manifest["max_generation"])
                require(r==json.loads(response) and env.step(action)==json.loads(event),"decision replay mismatch")
            else:
                require(state in ("pending","invalid") and i==len(rows)-1,"invalid attempt ordering")
                require(response is None and event is None,"failed attempt applied")
                if state=="invalid":
                    require(raw is not None,"invalid response missing")
                    try:
                        decode(json.loads(raw),self.manifest["mode"],obs,max_generation=self.manifest["max_generation"])
                    except (ValueError,TypeError,KeyError,RecursionError,OverflowError):
                        pass
                    else:
                        raise ValueError("invalid-response replay disagrees")
        return env,rows

    def claim(self):
        with self.tx() as db:
            env,rows=self.reconstruct(db)
            require(not (self.root/"STOP").exists(),"STOP file present")
            require(not rows or rows[-1][1]=="accepted","pending or invalid request; no resubmission")
            if env.state.stage==4:
                return None
            obs=env.observe()
            db.execute("INSERT INTO attempts(seq,state,observation) VALUES (?,?,?)",
                       (len(rows),"pending",canonical(obs).decode("ascii")))
            return obs

    def finish(self,raw):
        with self.tx() as db:
            env,rows=self.reconstruct(db)
            require(rows and rows[-1][1]=="pending" and rows[-1][3] is None,"no unanswered claim")
            seq=rows[-1][0]
            db.execute("UPDATE attempts SET raw=?,raw_sha256=? WHERE seq=?",
                       (canonical(raw).decode("ascii"),digest(raw),seq))
        with self.tx() as db:
            env,rows=self.reconstruct(db)
            require(rows[-1][0]==seq and rows[-1][1]=="pending","claim changed")
            obs=env.observe()
            require(digest(json.loads(rows[-1][3]))==digest(raw),"stored raw mismatch")
            try:
                response,action=decode(raw,self.manifest["mode"],obs,max_generation=self.manifest["max_generation"])
            except (ValueError,TypeError,KeyError,RecursionError,OverflowError) as error:
                db.execute("UPDATE attempts SET state='invalid',error_type=? WHERE seq=?",
                           (type(error).__name__,seq))
                return {"status":"invalid","state_id":obs["state_id"]}
            event=env.step(action)
            db.execute("UPDATE attempts SET state='accepted',response=?,event=? WHERE seq=?",
                       (canonical(response).decode("ascii"),canonical(event).decode("ascii"),seq))
            return {"status":"accepted","event":event,"complete":env.state.stage==4}

    def record_error(self,error):
        with self.tx() as db:
            db.execute("UPDATE attempts SET error_type=? WHERE state='pending'",(type(error).__name__[:96],))

    def status(self):
        with self.tx() as db:
            env,rows=self.reconstruct(db)
            result={"run_id":self.manifest["run_id"],"mode":self.manifest["mode"],
                    "accepted":sum(row[1]=="accepted" for row in rows),
                    "pending":sum(row[1]=="pending" for row in rows),
                    "invalid":sum(row[1]=="invalid" for row in rows),
                    "complete":env.state.stage==4,"remaining_credits":env.state.budget,
                    "training_admission":False}
            if env.state.stage==4:
                result["result"]=env.result()
            return result

    def fixture_choice(self,name):
        with self.tx() as db:
            env,rows=self.reconstruct(db)
            require(rows and rows[-1][1]=="pending","no pending fixture claim")
            if name=="oracle":
                policy=Oracle(env.model)
                a=policy.choose(env.state)
                policy.close()
                return a
            return Baseline(env.model,name).choose(env.state)

def run(journal,sampler,limit):
    require(type(limit) is int and 1<=limit<=32,"invalid launch limit")
    for _ in range(limit):
        obs=journal.claim()
        if obs is None:
            break
        try:
            raw=sampler(obs)
            receipt=journal.finish(raw)
        except BaseException as error:
            journal.record_error(error)
            raise
        if receipt["status"]!="accepted" or receipt.get("complete"):
            break
    return journal.status()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--out",required=True)
    a.add_argument("--seed",type=int,default=2026112820)
    a.add_argument("--ordinal",type=int,default=0)
    a.add_argument("--family",choices=("shared_control","shifted_control"),default="shared_control")
    a.add_argument("--difficulty",choices=("standard","frontier"),default="frontier")
    a.add_argument("--mode",choices=("fixture","external","live"),default="fixture")
    a.add_argument("--world-seed",type=int)
    a.add_argument("--max-generation",type=int,default=MAX_GEN)
    a.add_argument("--effort",type=float,default=EFFORT)
    for name in ("run","status","replay","next","submit"):
        a=sub.add_parser(name)
        a.add_argument("--campaign",required=True)
        if name=="run":
            a.add_argument("--limit",type=int,default=1)
            a.add_argument("--confirm-free-tier",action="store_true")
            a.add_argument("--fixture-policy",choices=("oracle","stop","myopic","direct_first","reuse_first"),default="oracle")
        if name=="submit":
            a.add_argument("--response",required=True)
    a=p.parse_args()
    if a.command=="prepare":
        result=prepare(a.out,seed=a.seed,ordinal=a.ordinal,family=a.family,difficulty=a.difficulty,mode=a.mode,world_seed=a.world_seed,max_generation=a.max_generation,effort=a.effort)
    else:
        j=Journal(a.campaign)
        mode=j.manifest["mode"]
        if a.command in ("status","replay"):
            result=j.status()
        elif a.command=="next":
            require(mode=="external","next only for external policy")
            obs=j.claim()
            result={"complete":True} if obs is None else {"observation":obs,"prompt":prompt(obs)}
        elif a.command=="submit":
            require(mode=="external","submit only for external policy")
            result=j.finish(external_raw(read_json(a.response)))
        else:
            require(mode in ("fixture","live"),"external policy uses next/submit")
            require(type(a.limit) is int and 1<=a.limit<=32,"invalid launch limit")
            status=j.status()
            require(not status["pending"] and not status["invalid"],"resolve stopped run; no resubmission")
            require(not (j.root/"STOP").exists(),"STOP file present")
            if status["complete"]:
                result=status
            elif mode=="live":
                require(a.confirm_free_tier,"confirm free-tier eligibility")
                t=TinkerFree(max_generation=j.manifest["max_generation"],effort=j.manifest["effort"])
                result=run(j,lambda obs:t.sample_prompt(prompt(obs)),a.limit)
            else:
                require(not a.confirm_free_tier,"fixture cannot become live")
                result=run(j,lambda obs:fixture_raw(fixture_response(obs,j.fixture_choice(a.fixture_policy))),a.limit)
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=="__main__":
    try:
        main()
    except (ValueError,RuntimeError,ImportError,TimeoutError) as e:
        print("Research program stopped: "+type(e).__name__,file=sys.stderr)
        sys.exit(2)
