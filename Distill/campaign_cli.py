"""Offline campaign CLI. All live submission is external to this package."""
import argparse
import json
from .batch import dollars, plan_campaign, rates
from .ledger import Ledger
from .registry import inventory, load_registry

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--budget-usd", required=True)
    p.add_argument("--reserve-usd", default="0")
    p.add_argument("--input-usd-per-million", required=True)
    p.add_argument("--output-usd-per-million", required=True)
    p.add_argument("--fee-usd-per-request", default="0")
    p.add_argument("--profile", choices=("quick", "full"), default="full")
    p.add_argument("--difficulty", nargs="+", choices=("foundation","standard","frontier"), default=["foundation","standard","frontier"])
    p.add_argument("--section-size", type=int, default=5)
    p.add_argument("--input-cap", type=int, default=6000)
    p.add_argument("--output-cap", type=int, default=5000)
    p.add_argument("--gym", nargs="+")
    p.add_argument("--instances-per-gym", type=int, default=0)
    p.add_argument("--instance-seed", type=int, default=20261127)
    for name in ("status","reserve","settle"):
        c = sub.add_parser(name)
        c.add_argument("--campaign", required=True)
        if name != "status":
            c.add_argument("--request-id", required=True)
        if name == "reserve":
            c.add_argument("--counted-input", type=int, required=True)
        if name == "settle":
            c.add_argument("--input-tokens", type=int, required=True)
            c.add_argument("--output-tokens", type=int, required=True)
            c.add_argument("--provider-receipt", required=True)
    a = parser.parse_args()
    if a.command == "inventory":
        result = inventory()
    elif a.command == "plan":
        adapters = load_registry()
        if a.gym:
            if len(set(a.gym)) != len(a.gym) or not set(a.gym) <= adapters.keys():
                parser.error("unknown or duplicate gym")
            adapters = {k: adapters[k] for k in a.gym}
        plan = plan_campaign(a.out, prices=rates(dollars(a.input_usd_per_million), dollars(a.output_usd_per_million), dollars(a.fee_usd_per_request)),
                             budget_microusd=dollars(a.budget_usd), reserve_microusd=dollars(a.reserve_usd),
                             profile=a.profile, difficulties=a.difficulty, section_size=a.section_size,
                             input_cap=a.input_cap, output_cap=a.output_cap, adapters=adapters,
                             instances_per_gym=a.instances_per_gym, instance_seed=a.instance_seed)
        result = {k:v for k,v in plan.items() if k != "jobs"}
        result["planned_requests"] = len(plan["jobs"])
        result["planned_question_slots"] = sum(j["question_count"] for j in plan["jobs"])
    else:
        ledger = Ledger(a.campaign)
        if a.command == "status":
            result = ledger.status()
        elif a.command == "reserve":
            result = ledger.reserve(a.request_id, counted_input_tokens=a.counted_input)
        else:
            result = ledger.settle(a.request_id, input_tokens=a.input_tokens, output_tokens=a.output_tokens, provider_receipt=a.provider_receipt)
    print(json.dumps(result, sort_keys=True, indent=2))

if __name__ == "__main__":
    main()
