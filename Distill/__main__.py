"""Offline CLI, from the repository root: python -m Distill --help."""
import argparse
import sys
from .core import compile_quiz, evaluate, load_adapter, read_json, render_prompt, validate_response, write_json
from .checkers import example_checkers
from .native import grade_native
from .instance_oracle import grade_instance
from .instances import compile_instance
from .registry import load_registry

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--adapter", required=True)
    c.add_argument("--profile", choices=("quick", "full"), default="full")
    c.add_argument("--case")
    c.add_argument("--difficulty", choices=("foundation", "standard", "frontier"), default="standard")
    c.add_argument("--out", required=True)
    i = sub.add_parser("instance")
    i.add_argument("--gym", required=True)
    i.add_argument("--seed", type=int, required=True)
    i.add_argument("--ordinal", type=int, default=0)
    i.add_argument("--difficulty", choices=("foundation","standard","frontier"), default="standard")
    i.add_argument("--profile", choices=("quick","full"), default="full")
    i.add_argument("--out", required=True)
    p = sub.add_parser("prompt")
    p.add_argument("--packet", required=True)
    for name in ("validate", "grade"):
        v = sub.add_parser(name)
        v.add_argument("--packet", required=True)
        v.add_argument("--response", required=True)
        if name == "grade":
            v.add_argument("--out", required=True)
            group = v.add_mutually_exclusive_group()
            group.add_argument("--toy-checkers", action="store_true")
            group.add_argument("--native-checkers", action="store_true")
            group.add_argument("--procedural-checkers", action="store_true")
    args = parser.parse_args()
    if args.command == "compile":
        packet = compile_quiz(load_adapter(args.adapter), profile=args.profile,
                              public_case=read_json(args.case) if args.case else None, difficulty=args.difficulty)
        write_json(args.out, packet)
    elif args.command == "instance":
        write_json(args.out, compile_instance(load_registry()[args.gym],seed=args.seed,ordinal=args.ordinal,difficulty=args.difficulty,profile=args.profile))
    elif args.command == "prompt":
        print(render_prompt(read_json(args.packet)))
    else:
        packet, response = read_json(args.packet), read_json(args.response)
        if args.command == "validate":
            print(f"Valid format: {len(validate_response(packet, response))} answers; correctness ungraded")
        else:
            if args.procedural_checkers:
                receipt = grade_instance(packet,response)
            elif args.native_checkers:
                receipt = grade_native(packet,response)
            else:
                receipt = evaluate(packet,response,checkers=example_checkers(packet) if args.toy_checkers else None)
            write_json(args.out, receipt)
            if any(row["exact_check"] == "checker_error" for row in receipt["checks"]):
                return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
