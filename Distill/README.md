# Distill v0.3.0: first-principles research examination library

A CPU-only, standard-library Python module that turns a gym's public design into a reproducible written exam. It composes 18 common questions, five-field step-by-step questions, domain packs, gym-specific probes and optional concrete public examples. Quick mode keeps eight common questions and every step, domain and gym-specific question. Full mode is the default. Version 0.2 adds three difficulty levels, independently answerable sections, fourteen byte-pinned public adapters, two native exact checks and a provider-neutral offline budget planner with transactional reservations. See [CAMPAIGN.md](CAMPAIGN.md). Version 0.3 expands the registry to 25 public design adapters across ten source repositories and adds ten families of reproducible, exactly graded finite mathematical instances. See [PROCEDURAL.md](PROCEDURAL.md). DISTILL-04 adds an opt-in free-tier Inkling evaluation runner, durable response journal and independent semantic-review forms. See [INKY_FREE.md](INKY_FREE.md). DISTILL-05 adds the research-specific v0.4.0 decision, interpretation, experiment design and falsification suite plus an exact synthetic action-ranking curriculum. See [RESEARCH.md](RESEARCH.md).

The model supplies visible, checkable derivations, explanations, examples and certificates. A valid JSON shape is not a correctness certificate. Exact answer checks and explanation review are recorded separately. No automatic model call, training, admission, provider spend or GPU initialization occurs. All receipts default to reference-only custody, including for Anthropic responses; review applicable provider terms and obtain required rights before any training use. Treat provider-reported identity in the response as claimed, not independently authenticated.

## Layout

- core.py: adapter validation, deterministic packet IDs, prompts, strict answer contract, evaluator callback.
- packs.py: reusable foundations, recurrence, proofs, geometry, strategy, algebra, research and authenticated-computation modules.
- catalog.json: legacy five-adapter v0.1 catalog, retained for compatibility.
- registry.json and registry_snapshots/: 25 byte-pinned adapters across ten repositories, including nine exact public sidecar snapshots and eleven source-pinned adapters in registry_new/.
- instances.py and instance_oracle.py: seeded mathematical case generation and independent finite-answer grading.
- native.py: reviewed source-pinned Pluto and COE native verifier wrappers.
- batch.py, ledger.py, campaign_cli.py: section planning, integer-cost budget envelope and persistent reservations.
- checkers.py: independent small public Pluto and COE toy-case checks; these are not the native gym scorers.
- rust/compile_exam.rs: optional standard-library Rust subprocess bridge; Python remains the canonical validator.
- tests/test_distill.py: offline integration and negative tests.

## Offline usage (from repository root, Python 3.10+)

    python -m unittest discover -s Distill/tests -v
    python -m Distill compile --adapter research/pluto_gym/distill_exam.json --out /tmp/pluto-exam.json
    python -m Distill prompt --packet /tmp/pluto-exam.json
    python -m Distill validate --packet /tmp/pluto-exam.json --response /tmp/pluto-answer.json
    python -m Distill grade --packet /tmp/pluto-exam.json --response /tmp/pluto-answer.json --toy-checkers --out /tmp/pluto-receipt.json

All JSON output files must be new. Windows paths can replace /tmp. A model runner can submit the rendered prompt externally and store its raw response in a separate journal. The v0.2 campaign CLI handles offline planning and local reservation accounting; live provider orchestration is still external.

    rustc Distill/rust/compile_exam.rs -o /tmp/distill-compile
    /tmp/distill-compile research/chromatic_observer_gym_v1/distill_exam.json /tmp/coe-exam.json

Any language can produce and consume JSON without linking PyTorch or Lean. Rust uses the Python CLI at present, not a second native implementation.

## Add a gym

Copy the nearest distill_exam.json and give it the native gym ID/version, exact repository/commit/path/blob source lock, public design objective, ordered steps, domain packs and gym-specific questions. Use only explicitly public development material; no sealed tests, private source, hidden labels or credentials. The source lock identifies the reviewed design; it does not by itself authenticate a live source file or confer permission to send that source to a provider. The adapter's native_verifier field is descriptive; the core never imports or executes a verifier from a string.

In a different repository use a pinned checkout of Compactification-Theory and add it to PYTHONPATH (or call its CLI in that checkout with an absolute adapter path). Place the adapter adjacent to the gym. No training-loop change is required.

    from Distill import compile_quiz, load_adapter, evaluate

    packet = compile_quiz(load_adapter("distill_exam.json"))
    # Call a model separately, bind actual metadata in the host, save raw response.
    # Only the gym owner supplies a trusted callable for exact grading:
    # receipt = evaluate(packet, response, checkers={"case.solve": own_checker})

A checker receives (public_case, answer_fields) and returns a bool for its own check. The host must authenticate the scorer, preserve raw outputs, bind source and model identity, and keep referee inputs outside the model-facing packet. Caller-selected custom cases must have the adapter's example kind and an explicit public_development classification. The key blacklist is a defense-in-depth check only; it cannot prove that arbitrary prose or values are public. Do not send evaluation labels or secret material.

## Evidence and versioning

See WIRE.md. The packet ID is SHA-256 over sorted-key compact ASCII JSON excluding packet_id, including the adapter digest, source pins, questions, public case, profile and library versions. The receipt binds the response bytes in that same canonical representation; raw API bytes must be stored separately if needed. Exact fixture checks only validate the requested final answer; explanation review remains pending. A self-reported model identity and an unsigned digest are not authorization. Freeze prompts and case lineages before comparison. Keep original frozen gym source and evidence unchanged.

The first five adapters are for IRC-PG-01 (Pluto), COE-01, ARG-01, AEG-01 and SCM-23. Nine additional pinned adapters cover DataGoblin, BPD and LoopedTransformers. Eleven further adapters cover EvalMath, MathCan, HybridTRMLDT, SelfModels, Control-Harness and GPTStoryworld. The generated cases are independent toy exercises inspired by these design sources, not native scorer replacements. The current release does not patch model training loops, score prose automatically, export SFT datasets, or call a paid model.
