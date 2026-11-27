"""Offline CLI, from the repository root: python -m Distill --help."""
import argparse
import sys
from .core import compile_quiz, evaluate, load_adapter, read_json, render_prompt, validate_response, write_json
from .checkers import example_checkers

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--adapter", required=True)
    c.add_argument("--profile", choices=("quick", "full"), default="full")
    c.add_argument("--case")
    c.add_argument("--out", required=True)
    p = sub.add_parser("prompt")
    p.add_argument("--packet", required=True)
    for name in ("validate", "grade"):
        v = sub.add_parser(name)
        v.add_argument("--packet", required=True)
        v.add_argument("--response", required=True)
        if name == "grade":
            v.add_argument("--out", required=True)
            v.add_argument("--toy-checkers", action="store_true")
    args = parser.parse_args()
    if args.command == "compile":
        packet = compile_quiz(load_adapter(args.adapter), profile=args.profile,
                              public_case=read_json(args.case) if args.case else None)
        write_json(args.out, packet)
    elif args.command == "prompt":
        print(render_prompt(read_json(args.packet)))
    else:
        packet, response = read_json(args.packet), read_json(args.response)
        if args.command == "validate":
            print(f"Valid format: {len(validate_response(packet, response))} answers; correctness ungraded")
        else:
            receipt = evaluate(packet, response, checkers=example_checkers(packet) if args.toy_checkers else None)
            write_json(args.out, receipt)
            if any(row["exact_check"] == "checker_error" for row in receipt["checks"]):
                return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
