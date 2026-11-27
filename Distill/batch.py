"""Offline, provider-neutral request planning and durable spend reservations.

All money is integer micro-USD. No API client, training export, or live requests.
A reservation is an authorization ceiling under the supplied rates and token
caps; it is not a guarantee about a provider invoice or an untrusted worker.
"""
from __future__ import annotations
import hashlib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from .core import (DIFFICULTIES, VERSION, _packet, compile_quiz, digest, object_keys,
                   read_json, render_prompt, require, split_quiz, write_json)
from .registry import load_registry
from .instances import FAMILIES, generate_instances

MILLION = 1_000_000

def integer(x, name, lo=0, hi=10**15):
    require(type(x) is int and lo <= x <= hi, f"invalid {name}")
    return x

def dollars(value):
    try:
        n = Decimal(str(value)) * MILLION
        require(n.is_finite() and n == n.to_integral_value(), "money must have at most six decimal places")
        return integer(int(n), "micro-USD")
    except (InvalidOperation, OverflowError):
        raise ValueError("invalid money") from None

def rates(input_micros, output_micros, fee_micros=0):
    return {"input_microusd_per_million": integer(input_micros, "input rate"),
            "output_microusd_per_million": integer(output_micros, "output rate"),
            "fee_microusd_per_request": integer(fee_micros, "request fee")}

def cost(prices, input_tokens, output_tokens):
    object_keys(prices, {"input_microusd_per_million", "output_microusd_per_million", "fee_microusd_per_request"}, "rates")
    p = rates(prices["input_microusd_per_million"], prices["output_microusd_per_million"], prices["fee_microusd_per_request"])
    i = integer(input_tokens, "input tokens", hi=10**9)
    o = integer(output_tokens, "output tokens", hi=10**9)
    return (i*p["input_microusd_per_million"] + o*p["output_microusd_per_million"] + MILLION-1)//MILLION + p["fee_microusd_per_request"]

def prompt_hash(packet):
    return hashlib.sha256(render_prompt(packet).encode("utf-8")).hexdigest()

def _job(packet, input_cap, output_cap, prices):
    return {"request_id": packet["packet_id"], "gym_id": packet["gym_id"],
            "difficulty": packet["difficulty"], "section_index": packet["section"]["index"],
            "section_count": packet["section"]["count"], "question_count": len(packet["questions"]),
            "prompt_sha256": prompt_hash(packet), "input_token_cap": input_cap,
            "output_token_cap": output_cap, "max_cost_microusd": cost(prices, input_cap, output_cap)}

def plan_campaign(out, *, prices, budget_microusd, reserve_microusd=0,
                  profile="full", difficulties=("foundation", "standard", "frontier"),
                  section_size=5, input_cap=6000, output_cap=5000, adapters=None,
                  instances_per_gym=0, instance_seed=20261127):
    integer(budget_microusd, "budget", 1)
    integer(reserve_microusd, "reserve", 0, budget_microusd)
    integer(input_cap, "input cap", 1, 10**7)
    integer(output_cap, "output cap", 1, 10**7)
    require(isinstance(difficulties, (list, tuple)) and 1 <= len(difficulties) <= 3 and len(set(difficulties)) == len(difficulties) and set(difficulties) <= DIFFICULTIES.keys(), "invalid difficulties")
    adapters = load_registry() if adapters is None else adapters
    require(bool(adapters), "empty gym roster")
    # Complete one round across gyms and difficulty groups before the next.
    queues = [split_quiz(compile_quiz(a, profile=profile, difficulty=d), max_questions=section_size)
              for d in difficulties for a in adapters.values()]
    integer(instances_per_gym, "instances per gym", 0, 512)
    integer(instance_seed, "instance seed", 0, 2**63-1)
    if instances_per_gym:
        for gym_id,a in adapters.items():
            if gym_id in FAMILIES:
                parents=generate_instances(a,seed=instance_seed,count=instances_per_gym,difficulties=difficulties,profile=profile)
                queues.append([s for p in parents for s in split_quiz(p,max_questions=section_size)])
    candidates = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    max_job = cost(prices, input_cap, output_cap)
    available = budget_microusd - reserve_microusd
    n = len(candidates) if max_job == 0 else min(len(candidates), available//max_job)
    selected = candidates[:n]
    require(bool(selected), "budget cannot reserve even one request")
    jobs = [_job(p, input_cap, output_cap, prices) for p in selected]
    plan = {"schema_version": "distill.batch_plan.v2", "library_version": VERSION,
            "purpose": "reference_evaluation", "training_admission": False,
            "rates": prices, "budget_microusd": budget_microusd, "reserve_microusd": reserve_microusd,
            "authorized_max_microusd": sum(j["max_cost_microusd"] for j in jobs),
            "profile": profile, "difficulties": list(difficulties), "section_size": section_size,
            "candidate_count": len(candidates), "omitted_count": len(candidates)-len(jobs), "jobs": jobs,
            "instance_seed": instance_seed, "instances_per_gym": instances_per_gym,
            "instance_candidate_sections": sum("instance" in p for p in candidates),
            "instance_selected_sections": sum("instance" in p for p in selected)}
    plan["plan_id"] = digest(plan)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    packets = out/"packets"
    packets.mkdir()
    for p in selected:
        write_json(packets/(p["packet_id"]+".json"), p)
    write_json(out/"plan.json", plan)
    return plan

def load_plan(out):
    out = Path(out)
    p = read_json(out/"plan.json")
    base_keys={"schema_version", "library_version", "purpose", "training_admission", "rates", "budget_microusd",
                    "reserve_microusd", "authorized_max_microusd", "profile", "difficulties", "section_size",
                    "candidate_count", "omitted_count", "jobs", "plan_id"}
    modern=p.get("schema_version")=="distill.batch_plan.v2" and p.get("library_version")==VERSION
    legacy=p.get("schema_version")=="distill.batch_plan.v1" and p.get("library_version")=="0.2.0"
    require(modern or legacy,"unsupported plan version")
    if modern:
        base_keys |= {"instance_seed","instances_per_gym","instance_candidate_sections","instance_selected_sections"}
        integer(p.get("instance_seed"),"instance seed",0,2**63-1)
        integer(p.get("instances_per_gym"),"instances per gym",0,512)
        integer(p.get("instance_candidate_sections"),"instance candidates")
        integer(p.get("instance_selected_sections"),"instance selected")
    object_keys(p,base_keys,"plan")
    require(p["purpose"] == "reference_evaluation" and p["training_admission"] is False, "invalid plan")
    require(p["plan_id"] == digest({k:v for k,v in p.items() if k!="plan_id"}), "plan digest mismatch")
    integer(p["budget_microusd"], "budget", 1)
    integer(p["reserve_microusd"], "reserve", 0, p["budget_microusd"])
    require(isinstance(p["jobs"], list) and bool(p["jobs"]), "empty jobs")
    require(len(p["jobs"]) + integer(p["omitted_count"], "omitted") == p["candidate_count"], "candidate count mismatch")
    jobs = {}
    instance_sections=0
    for j in p["jobs"]:
        rid = j["request_id"]
        require(isinstance(rid, str) and len(rid)==64 and all(c in "0123456789abcdef" for c in rid), "invalid request id")
        require(rid not in jobs, "duplicate request")
        packet = read_json(out/"packets"/(rid+".json"))
        _packet(packet)
        require(packet["library_version"]==p["library_version"],"packet library version mismatch")
        if "instance" in packet:
            require(modern and packet["instance"]["seed"]==p["instance_seed"],"instance seed mismatch")
            instance_sections+=1
        require(packet["packet_id"] == rid and packet["profile"] == p["profile"] and packet["difficulty"] in p["difficulties"], "packet-plan mismatch")
        require(1 <= len(packet["questions"]) <= p["section_size"], "section cap mismatch")
        require(j == _job(packet, j["input_token_cap"], j["output_token_cap"], p["rates"]), "request binding mismatch")
        integer(j["input_token_cap"], "input cap", 1)
        integer(j["output_token_cap"], "output cap", 1)
        jobs[rid] = j
    if modern:
        require(instance_sections==p["instance_selected_sections"]<=p["instance_candidate_sections"]<=p["candidate_count"],"instance section accounting mismatch")
        require(p["instance_candidate_sections"]-instance_sections<=p["omitted_count"],"instance omission mismatch")
    total = sum(j["max_cost_microusd"] for j in jobs.values())
    require(total == p["authorized_max_microusd"] and total <= p["budget_microusd"]-p["reserve_microusd"], "plan exceeds budget")
    return p, jobs
