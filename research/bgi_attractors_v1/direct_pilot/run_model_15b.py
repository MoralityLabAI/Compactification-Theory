"""BGI-01E: model-scale-only replay of the frozen BGI-01D experiment.

Uses the original BGI-01D evaluator, fixtures, parser, decoder settings, prompt
cache and controller unchanged. Only model ID/revision and output path differ.
These scenarios were inspected after BGI-01D and are exploratory for 1.5B.
"""
from __future__ import annotations

import sys
from pathlib import Path

import run_model

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "775b11afaf83e0dc75bd5abaf90133e47b3ec082"


def main():
    # The QwenProposer resolves MODEL_ID and MODEL_REVISION at instantiation.
    # Configuration override is confined to this process; the BGI-01D source
    # and original received artifacts remain byte-for-byte unchanged.
    run_model.MODEL_ID = MODEL_ID
    run_model.MODEL_REVISION = MODEL_REVISION
    default_output = Path(__file__).resolve().parent / "results" / "qwen15b_direct_policy_v1.json"
    if not any(arg == "--output" or arg.startswith("--output=") for arg in sys.argv[1:]):
        sys.argv.extend(["--output", str(default_output)])
    run_model.main()


if __name__ == "__main__":
    main()
