# Algernon Algebraic Reasoning Gym | ARG-01

A runnable CPU benchmark combining EvalMath's independent exact finite formal-series verifier with **frozen, previously trained** RRC-19 recurrent and feedforward class-ranking checkpoints. A separate JSONL interface accepts proposals from an external Qwen3.5 / looped transformer checkpoint, with its model revision and depth recorded.

**Executed evidence:** 6,240-parameter RRC-19 balanced GRU and skewed/shuffled controls, 6,235-parameter RRC-19 MLP, certified class-size ordering, and complete exhaustive exact search on a newly seeded EvalMath roster. This is a pretrained *task-specific* GRU, not a foundation-model Qwen graft. No new model training, GPU, internal Qwen recurrence, or hidden-state intervention occurs in ARG-01.

## Ownership and exact mathematical boundary

EvalMath owns `spiral_ln.algebraic_reasoning_oracle`, its tests, and a deterministic 228-case reference receipt. The `evalmath_pin.json` locks the oracle source SHA-256 and the merged EvalMath commit `88bb6cf8d2fe172e37d25964dfb2c285d9033841` (EvalMath PR #31). The workflow checks out that revision explicitly. `gym.py` imports that module from a separate checkout at the pinned revision. It never vendors the EvalMath verifier or substitutes a neural predictor for an exact equality check.

Two public demonstrations give integer coefficients through degree 12, with a fresh query input. Graph programs contain three symbols from `I,N,S,R`, operating on four supplied graph topologies. One paid graph query computes both observed examples and records exact coefficient agreement and the first mismatching degree. A maximum of twelve candidates is allowed for each learned/static restricted policy. The complete certified class and 64-syntax searches receive explicitly larger budgets. Future degree-13–32 query output is held back until independent grading.

The 36-case calibration set is generated and receipt-hashed but unused in the benchmark model selection. The new 168-case positive roster spans seven source generators; rational, squared, sign-twisted product, and lacunary are test-only relative to the three RRC-19 training-family classes. Twelve inconsistent demo cases and six matched future-collision pairs provide negative controls. All source-family and topology labels remain absent from the learned feature encoder other than the public topology identity.

## Reproduce on CPU

Use the EvalMath revision in `evalmath_pin.json` in a sibling checkout. The bytewise oracle source SHA-256 is checked on every local run. The pinned revision is merged and the bytewise source checksum is checked on every run. From Compactification-Theory root:

```sh
PYTHONPATH=.:../EvalMath/src python -m unittest discover -s research/algernon/algebraic_reasoning_gym -p 'test_gym.py' -v
PYTHONPATH=.:../EvalMath/src python -m research.algernon.algebraic_reasoning_gym.gym --check
```

No inference model is downloaded: the four existing RRC-19 checkpoint JSON files under `research/rrc_compactification/results/` are used as frozen weights, with pinned SHA-256 verification. The RRC-18 exact class compiler is an explicit shared algorithmic dependency. `results/arg01_reference.json` is a source-hashed, deterministic receipt; timing lives in a separate machine-specific profile.

## External pretrained model protocol

ARG-01 can export test questions for a real Qwen or looped model, while keeping target programs and future coefficients inside EvalMath:

```sh
PYTHONPATH=.:../EvalMath/src python -m research.algernon.algebraic_reasoning_gym.external \
  --export-public /tmp/arg01_public.jsonl
```

A model runner returns one JSON object per line with these mandatory fields:

```json
{"case_id":"signed:signed:00001","public_sha256":"<64-hex-input-sha256>","depth":2,"program":["I","R","N"],"model_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
```

`depth` may be 0, 1, 2, 4, or 8. A null program is an explicit abstention; missing records are reported separately. Inference time and token/forward-call counts are optional extra fields. All records must refer to a single pinned model revision. Responses containing future labels or unrecognized fields fail closed. The scored receipt is **not** evidence of actual internal latent loops without an independently recorded model hook/forward-pass trace. A 5080 experiment should pin exact model weights, site, hook and decode settings separately before use.

```sh
PYTHONPATH=.:../EvalMath/src python -m research.algernon.algebraic_reasoning_gym.external \
  --score-proposals /tmp/qwen_arg01.jsonl --out /tmp/qwen_arg01_scored.json
```

## ARG-02

[`arg02/`](arg02/README.md) evaluates Qwen3.5-0.8B-Base on this roster with explicit recurrent depth and matched-compute controls, through the same public-only export and EvalMath grading. Protocol: [`arg02/PROTOCOL.md`](arg02/PROTOCOL.md).

## Research interpretation

The primary signal is executable-program *future* accuracy per full measured computation, including class compilation, predictor inference and exact candidate verification. Aggregate correctness should be split by known/novel source family; source-family holdouts in RRC-19 itself are public, so ARG-01 is another deterministic research evaluation rather than an independently blind study. An RRC-19 GRU forward pass processes the thirteen coefficient steps of its input **once**; additional candidate-verifier calls are external computation, not new GRU refinement steps. A true recurrence-depth comparison requires the external checkpoint and actual latent forward traces.

A success claim about Algernon's joint processor requires a model advantage beyond the strongest exact symbolic control at comparable resources, causal loop-depth ablations, and retained base skills on a separate evaluation. ARG-01 establishes the interface for that experiment.
