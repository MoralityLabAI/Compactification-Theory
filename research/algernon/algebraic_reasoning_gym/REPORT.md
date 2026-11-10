# ARG-01: frozen recurrent algebraic coprocessor pilot

**Executed:** November 10, 2026, CPU-only. EvalMath supplies a separately implemented exact finite-series interpreter. The recurrent model is the **previously trained RRC-19 6,240-parameter GRU**, whose full JSON checkpoint is loaded without training or modification. It is a domain-specific model, rather than a pretrained Qwen or a latent-layer graft. No new GPU model executed in this pilot.

## Verified future predictions

The first 13 coefficients of two public demonstration pairs are the entire candidate-selection evidence; a separate degree-32 target scores extrapolation. All learned and static twelve-check arms have equal exact candidate budgets, though they differ in inference work. Complete certified-class and complete syntax searches are higher-budget references.

| Source family (24 cases) | Frozen GRU | Frozen FF | Class-size, 12 checks | Complete class search |
|---|---:|---:|---:|---:|
| Signed, seen generator | 22 | 19 | 14 | 24 |
| Partition product, seen | 24 | 23 | 11 | 24 |
| Sparse signed, seen | 15 | 14 | 13 | 19 |
| Rational reciprocal, novel | 15 | 13 | 16 | 23 |
| Convolution square, novel | 17 | 18 | 16 | 21 |
| Sign-twisted product, novel | 14 | 16 | 16 | 24 |
| Lacunary cubic substitution, novel | 22 | 19 | 16 | 24 |
| **All positives /168** | **129** | **122** | 102 | **159** |
| **Novel sources /96** | **68** | **66** | 64 | **92** |

The exact independent 64-syntax search recovers **157/168** correct future predictions, while verifying all 168 observed prefixes; the complete *certified-class* order recovers **159/168**. Different accepted programs can agree through degree 12 and diverge at degree 32, so exhaustive search is a finite **acceptance** upper control, not a universal future-prediction upper bound. The accepted programs are never treated as proofs of infinite identities.

All twelve selected invalid demonstration cases fail complete finite-DSL search. Six paired late-future mutants preserve identical observed inputs and candidate traces, producing six unavoidable future errors under complete search. Reusing the same frozen GRU on source-family shifts yields only a 2-case gain over feedforward on the 96 novel examples. That is a small exploratory sample; no robust generalization or internal-loop advantage is claimed.

## Computation

The pretrained GRU contributes 6,240 FP32 parameters and a 32-number recurrent state, with an estimated 53,216 dense multiply-accumulates per query (13 coefficient timesteps). The FF has 6,235 FP32 parameters and an estimated 6,120 MACs. The neural method makes zero unsampled candidate-graph executions before proposing an order. Each paid exact query then executes both public demonstrations and charges a separate integer-loop work proxy. The 12-check GRU executes 909 exact two-demo queries across the 168 positive cases, compared with 1,017 for the FF and 1,242 for the twelve-check certified class-size order.

**Observed host profile (one single-process warm pass):** the GRU uses ~582 ms across all 168 positive cases (inference plus exact verification), the FF ~90 ms, the restricted class-size order ~58 ms and the complete certified class search ~69 ms. These are machine-specific Python 3.13 wall-clock measurements, excluding independent future grading, prior training, and cold setup. The same run measured ~189 ms to build the exact four-topology class map and ~7 ms to deserialize and pin the four checkpoints. Shared compiler initialization and model loading are recorded separately.

The model remains slower than complete symbolic search on this CPU while recovering fewer correct futures. The ongoing benefit of the architecture is a falsifiable model-to-exact-verifier interface and a reproducible set of model baseline numbers, not a demonstrated compute-efficient reasoner.

## Checks and provenance

- EvalMath oracle: 12 tests, source-hashed deterministic 228-case reference manifest, merged in EvalMath PR #31.
- Algernon harness: 14 tests, source-hashed receipt, exact budget and negative/collision tests.
- Cross-check: EvalMath's independent arithmetic agrees with the existing RRC interpreter on 512 randomly generated graph/topology/input cases.
- Frozen model: checkpoint SHA-256 is verified for four RRC-19 arms before deserialization. No gradient training or model selection occurred.
- Tests explicitly replace the hidden graph/future while holding visible demonstrations fixed, ensuring the model's proposed class order remains invariant.
- Timing measurements are stored in `results/arg01_reference_profile.json`; the deterministic receipt in `results/arg01_reference.json` excludes elapsed time.

## Next executed-model gate

Provide a local Qwen3.5-0.8B (or pinned LOTUS/looped checkpoint) and inference implementation with inspectable per-loop latent/forward traces. The included `external.py` can export public prompts and score records at depths `0/1/2/4/8`, requiring a resolved checkpoint revision, actual loop timing and a separately fixed test roster. Compare against the frozen 6,240-weight GRU, 6,235-weight FF, 12-check class-size prior and complete symbolic search. All candidate operations must still be verified independently by EvalMath; a model-written explanation is optional and carries no correctness credit.
