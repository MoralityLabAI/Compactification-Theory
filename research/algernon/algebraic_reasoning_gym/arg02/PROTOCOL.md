# ARG-02 protocol: recurrent depth in a pretrained Qwen on the Algernon algebraic gym

**Registered 2026-11-10, before any Qwen output on the development or test rows existed.** The only model outputs produced before registration are the CPU parity check in `results/arg02_checkpoint_check.json` (two development prompts, depth 1 and one extra block pass, no grading) and the tiny random-model smoke runs used by the tests.

## Question

Does additional internal recurrent computation make a pretrained model recover algebraic programs that its first forward pass misses, with an advantage over exact symbolic search and over neural controls that spend the same compute without recurrence? This is a falsification test: every arm and every negative outcome is reported.

## Model and recurrence

- Inventory (`config/checkpoint_lock.json`): no looped-layer Qwen checkpoint survives on this machine. LoopedTransformers `experiments/recurrent_recovery_laco_v1` (Qwen3-0.6B with a LaCo recurrent block) kept manifests only; its `trainable.pt` and tuned teacher are gone. HybridTRMLDT has no recurrent language model. `tomg-group-umd/huginn-0125` is natively recurrent but is not a Qwen model and is not used.
- Model: `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, bf16, SDPA, transformers 5.17.0, torch 2.11.0+cu128, loaded as `Qwen3_5ForCausalLM` (752,393,024 text parameters). File SHA-256 values are in the lock; the runner refuses a snapshot that differs.
- Qwen3.5-0.8B has no native recurrence. ARG-02 adds an explicit, **untrained** block loop (`recurrence.py`): decoder layers `[a, b)` form a block B applied `d` times to the residual stream with the pretrained weights unchanged. `x = layers[0:a](embed)`, `h_1 = B(x)`, `h_t = B(h_{t-1})`, readout `lm_head(norm(layers[b:](h_t)))`. Depth 1 is the unmodified model; the CPU check measured a maximum absolute log-probability difference of 0.0 against the HF forward on the real weights.
- The 24 layers repeat a 3 linear-attention (Gated DeltaNet) + 1 full-attention pattern. Candidate blocks are four-layer groups: `[12,16)`, `[8,12)`, `[16,20)`, `[4,8)`.
- The external iterative controller (below) is a separate, explicitly external arm. Repeated prompting is never described as latent recurrence.

## Splits (fixed, `splits.py`, audit in `results/arg02_splits_audit.json`)

| Split | Rows | Source | Use |
|---|---:|---|---|
| test | 192 | ARG-01 EvalMath roster (168 positives over 7 generators, 12 inconsistent negatives, 6 future-collision pairs) | final run, once |
| calibration | 36 | ARG-01 calibration roster (seen generators) | development selection, probe hyperparameters |
| dev | 96 | ARG-02 roster, seed 2026111102, seen generators only | development selection, probe training |
| shots | 4 | ARG-02 roster, seed 2026111103, one per topology | in-context examples when shots = 4 |

Development = calibration + dev (132 rows). All non-test rosters are signature-disjoint from every EvalMath roster case. The test rows are byte-identical to `external.public_records()`. The four novel generators (rational, squared, twisted product, lacunary) appear only in test.

## Model input and answers

- The runner reads only exported public JSONL (`topology`, two demonstrations through degree 12, query, digest). It never builds the roster; a test patches `oracle.roster` to raise during every runner stage. Source graphs and future coefficients are used only by the grader. The query series is public but not rendered: it carries no information about the program. Case identifiers are not rendered.
- The prompt states the four operators and four topologies (the public DSL), the two demonstrations, and ends with `\nProgram:`. Operators ` I`, ` N`, ` S`, ` R` are single tokens.
- Each prompt is scored as 16 rows (prompt + every operator pair), giving the full-vocabulary log-probabilities of all 64 programs and the unconstrained greedy decode. **Constrained proposal**: the 64 programs ranked by joint log-probability. **Greedy proposal**: argmax tokens; invalid when any is not an operator token (the invalid-program rate).
- Every proposal is checked by `oracle.search` on the two public demonstrations; future correctness is read only inside `oracle.grade`. Budgets: 1 check (top-1) and 12 checks (the ARG-01 budget).

## Arms

| Arm | Definition | Compute charged |
|---|---|---|
| base = loop(1) | unmodified forward | 1 forward |
| loop(d), d = 2,4,8,16 | `h_t = B(h_{t-1})`; every step 1..16 is also read out for the depth curve | prelude + d blocks + readout, per CUDA-event timing |
| open(d) | residual-feedback ablation: `h_t = h_{t-1} + (h_1 - x)`. The block sees only x; its single update is re-applied d times, never recomputed from the current state | prelude + 1 block + readout |
| shuffle(d) | shuffled intermediate state: `h_t = h_1 + D_t`, D_t = `h_t - h_1` of a donor case's own loop (the previous test row, cyclic), right-aligned by position and matched by candidate row | recipient + donor loop to d |
| ensemble(m(d)) | repeated-forward control: mean joint log-probability over the first m(d) prompt variants (canonical, demos swapped, space separators, both), depth 1; m(d) = round((24 + (d-1)(b-a)) / 24) capped at 4, i.e. matched layer evaluations | m forwards |
| ff2b | matched-compute feedforward: `Qwen/Qwen3.5-2B-Base` at `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, single pass, same prompt and tokenizer; 2.68x the 0.8B forward MACs, between loop(8) (2.16x) and loop(16) (3.49x) | 1 forward |
| controller | external iterative controller: depth-1 model, after each rejected candidate the prompt gains "Rejected: a b c (example k differs at x^n ...)" from a public-demo check; next proposal is the best untried program; up to 12 rounds | one forward per round + public checks |
| perturbation | loop to 16 with a Gaussian kick of 0.25 x RMS(h_1) added after step 1 (seeded per row); readouts at 1,2,4,8,16; distance to the clean trajectory per step | as loop(16) |
| patching | at depth p = max(2, d*), the answer-site states (last 5 positions: `\n`, `Program`, `:`, two candidate tokens) are replaced by the donor's at step 1, ceil(p/2) or p, then the loop continues | 3 loops to p + donor |
| symbolic | ARG-01 arms re-scored per case through the unmodified `gym.score_case`: class-size order (12 checks), complete certified class search, exhaustive 64-program search, frozen RRC-19 GRU and FF | CPU time + verifier work |

The base model is the parameter-matched feedforward control for every loop depth (identical weights). ensemble and ff2b are the equal-compute controls.

## Development selection (run before the test stage)

1. Shots: k in {0, 4} at depth 1 on the 132 development rows; choose the larger development future-correct count at 12 checks; tie -> 0.
2. Block and depth: with the chosen k, each candidate block's loop trajectory on the development rows; choose (block, d*) with d* in {2,4,8,16} maximising development future-correct at 12 checks; tie -> smaller d, then the block order above. The selection is the one most favourable to recurrence that development data supports. It is written to `selection.json`, whose SHA-256 enters every test part key.
3. No other setting is tuned. Sigma, ensemble variants, patch steps, controller rounds and budgets are fixed here.

## Measurements

- Per arm on the 168 positives: future-correct and accepted at 12 and at 1 check, greedy validity, greedy acceptance, rank of the source program, by generator family and known/novel split; negatives accepted (must be 0); collision pairs with identical traces (identical public input must give identical proposals).
- Depth: per-step curves 1..16, first step at which top-1 is future-correct, programs found only after step 1 and whether they survive to step 16, answers lost or gained between step 1 and each depth, relative residual norm `|h_t - h_{t-1}| / |h_{t-1}|` per step and its per-step reduction ratio.
- Perturbation recovery: future-correct and top-1 agreement with the clean run per readout, and distance to the clean state over the kick norm per step.
- Patching: top-1 kept, moved to the donor's answer, changed; KL of the 64-program distribution to the clean run.
- Probes (`probe.py`): linear probes on the block output at the last prompt token, step 0 (block input) to 16. Labels from the canonical certified class representative (RRC-18 class map) of the source program: its outer operator (4 classes) and whether it uses R. Trained on the 96 dev rows, PCA size and L2 chosen on the 36 calibration rows, tested on seen-generator and novel-generator test positives, with the majority baseline and a 200-permutation label null.
- Compute: CUDA-event GPU time per segment (a depth-d latency is prelude + d block steps + its readout inside the combined trajectory), analytic MACs (`flops.py`), tokens, layer evaluations, forward passes, verifier checks, ARG-01 integer-loop proxy, verifier CPU time, p50/p95 end-to-end latency, peak allocated VRAM per stage. Integer-loop work and neural MACs are never summed.

## Statistics and decision rules

Primary metric: future-correct at 12 checks on the 168 test positives. Paired differences use exact McNemar tests and a 95% cluster bootstrap (clusters: source family x topology, 28 cells, 10,000 resamples, seed 2026111104).

- **H1 depth**: loop(d*) - loop(1) > 0, cluster-bootstrap lower bound > 0 and McNemar p < 0.05.
- **H2a feedback**: loop(d*) - open(d*) > 0, lower bound > 0.
- **H2b state**: loop(d*) - shuffle(d*) > 0, lower bound > 0. A depth gain that survives replacing the case's own recurrent displacement with another case's is not evidence of case-specific recurrent reasoning.
- **H3a repeat**: loop(d*) - ensemble(m(d*)) > 0, lower bound > 0.
- **H3b feedforward**: loop(d*) - ff2b > 0, lower bound > 0.
- **H3c efficiency**: future-correct per end-to-end second of loop(d*) above loop(1) and ensemble(m(d*)).
- **H4 symbolic**: loop(d*) future-correct >= complete certified class search and more future-correct per end-to-end second.
- A useful recurrent mechanism is claimed only if H1, H2a, H2b, H3a and H3b all hold. Every other outcome is reported as the negative result it is, with the failing rule named.

## Stop conditions

The runner stops on a checkpoint hash mismatch, a public row with any non-public field, a prompt whose tail tokenises differently, operator tokens that are not single tokens, or parts written under different keys. It exits 75 at a part boundary when the GPU is power-capped (`vramorama run --pause-exit 75` restarts it; finished parts are reused only under an identical key). The analysis stops if ARG-01's `external.py` scorer disagrees with ARG-02's top-1 grading.

## Reproduction

`results/arg02_answer_tape.jsonl.gz` holds every model answer record (log-probabilities rounded to 1e-6 before grading). `analyze replay` regrades it on CPU against `results/arg02_receipt.json` with no model or GPU.
