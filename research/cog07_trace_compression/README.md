# COG-07: finite compression of verified proof-step examples

**Evidence:** finite verified example; generic supporting lemmas checked in Lean 4.35.0.
**Run date:** 2026-11-26.
**Source:** [DataGoblin COG-02](https://github.com/MoralityLabAI/DataGoblin/pull/77), commit `ae0dbecffc1d246778ee47d8f3d8ec59b5948f2b`.

Five labeled examples preserve every hypothesis consistent with the 11,836 training examples in this experiment. The declared class has 6,562 hypotheses. Exactly one survives both samples. An executable decoder reconstructs it from a 13-byte packet, given the shared feature extractor and class definition.

The training sample contains 46 distinct feature/label constraints. Encoding those unique constraints takes 59 bytes, which supplies the relevant deduplication baseline. The five-example result concerns this finite symbolic class. The experiment measures candidate classification in synthetic Horn proof worlds.

## What runs here

`compression.py` reads the COG-02 ranking corpus and computes eight public Boolean features for each proposed rule application. A useful candidate has all premises available, adds a new conclusion, and lies on a syntactic dependency path to the goal. The target is exactly the conjunction of these first three features. The extractor computes backward ancestry from the public rule graph.

The hypothesis class contains every conjunction with a positive, negative, or absent literal at each of eight positions, plus constant false. Its size is `3**8 + 1 = 6562`. The selector greedily covers all hypotheses excluded by the complete training sample. It accepts training examples only. The decoder chooses the consistent hypothesis with the fewest literals, breaking ties by the fixed enumeration order.

Selection retains two positive and three negative examples from two training roots. Both the full sample and the decoded packet leave hypothesis 3159, which requires bits 0, 1, and 2. Evaluating every possible eight-bit input confirms equality with the curriculum's specified target formula.

## Measured results

| Representation | Examples or constraints | Encoded bytes | Training version space preserved |
|---|---:|---:|---|
| Full training sample | 11,836 | 13,323 | Yes |
| Deduplicated sample | 46 | 59 | Yes |
| Greedy cover | 5 | 13 | Yes |
| Random equal-count sample, eight seeds | 5 each | 13 each | No |
| Positive examples from the cover | 2 | 10 | No |

The wire packet includes a seven-byte header and six bytes for five nine-bit feature/label values, with zero padding. The decoder consumes values directly. Provenance indices stay in the separate 1,531-byte audit sidecar. Shared source code and protocol sizes are measured separately in [report.json](results/report.json).

| Classifier | Ordinary errors / 10,875 candidates | Deeper-world errors / 33,237 candidates |
|---|---:|---:|
| Full training sample | 0 | 0 |
| Five-example greedy cover | 0 | 0 |
| Random five examples, range over eight seeds | 1,193 to 5,006 | 5,830 to 11,462 |
| Positive examples only | 9,944 | 31,120 |

Each test partition contains 64 source roots. Candidate rows within a root share the same world. Every random control makes at least one error on every test root. These controls show how retaining constraints affects this decoder; they do not establish a population effect for learned proof systems.

## Meaning of the certificate

For a fixed class `H`, let `V(S)` be the hypotheses agreeing with every labeled example in sample `S`. If `C` is a subset of `S` and each hypothesis excluded by `S` is also excluded by `C`, then `V(S) = V(C)`. Consequently, any fixed decoder that depends only on the version space returns the same hypothesis on both inputs.

[FiniteCompression.lean](FiniteCompression.lean) proves the consistency equivalence and its implications for version-space equality and a fixed decoder. Its assumptions explicitly include subset membership and coverage of excluded hypotheses. The concrete enumeration over 6,562 hypotheses runs in Python. Lean checks the generic connection between that certificate and the preserved decoder. The native audit reports only `propext`, `Classical.choice`, and `Quot.sound` as dependencies.

The greedy procedure removes at least one previously uncovered excluded hypothesis per selected example. This gives a finite size bound of `|H| - |V(S)|`, equal to 6,561 here. Five is the measured cover size; the code does not claim it is the smallest possible cover. [THEORY.md](THEORY.md) states the assumptions and encoding costs.

## Scope and timing

The shared feature extractor supplies substantial mathematical structure. Since the known target is a conjunction already in the class, recovering its unique hypothesis gives zero error on all 256 feature patterns analytically. Generalization scores here exercise the pipeline and authenticate the data path.

A class index would need only 13 bits when both parties already share the class ordering. The sample packet preserves the full training version space and carries selected observations, which defines the question this experiment answers. Neural parameter counts and proof discovery on research mathematics remain outside its measured scope.

All source worlds and seeds are public. The original protocol is retained in [protocol_initial.json](protocol_initial.json). The amendment in [protocol.json](protocol.json) binds the published source and input hash, and adds the deduplication and analytic-target diagnostics after the first run. Selection and scored data stayed fixed. The [power diagnostic](results/preflight.json) is retrospective and records `REDESIGN_OR_PILOT_ONLY`. Matched shams are absent, so the planner declines a power calculation. Random sample controls measure decoder sensitivity. The exact finite certificate supplies the primary evidence.

## Reproduce

Python 3.11 or later suffices for the experiment and audits. Pytest is needed for the tests. From this repository root, with the pinned DataGoblin checkout in `../DataGoblin`:

```bash
python ../DataGoblin/src/datagoblin/training_gym/rlt_gym/cogentic_ledger.py \
  --out /tmp/cog02-corpus --per-split 64 --seed 20261126
python research/cog07_trace_compression/compression.py \
  --ranking /tmp/cog02-corpus/ranking.jsonl \
  --output /tmp/cog07-replay
python research/cog07_trace_compression/audit.py \
  --results /tmp/cog07-replay --lean /path/to/lean-4.35.0/bin/lean
python research/cog07_trace_compression/audit_provenance.py \
  --tasks /tmp/cog02-corpus/tasks.jsonl \
  --traces /tmp/cog02-corpus/traces.jsonl \
  --ranking /tmp/cog02-corpus/ranking.jsonl \
  --selected /tmp/cog07-replay/selected.json \
  --output /tmp/cog07-provenance.json
python -m pytest -q research/cog07_trace_compression
```

The finite audit checks 15 conditions. A separate verifier authenticates all 256 source worlds, replays 2,965 teacher attempts, and checks the 66,893 candidate labels and features at 2,453 reached states before enumerating the selected certificate. Tests cover the exhaustive two-bit case, malformed packets, nontraining selection, unrealizable samples, altered observations, and forged provenance. Receipts include SHA-256 hashes for their inputs and code.
