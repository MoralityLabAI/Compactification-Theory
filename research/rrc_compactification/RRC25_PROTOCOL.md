# RRC-25: executed CPU protocol for delayed composite-branch search

**Date:** 2026-11-10. Exploratory, one fixed training seed. Previous RRC-24 models, source designs and outcomes were known when the new grammar was chosen. The split and controls were implemented during the experiment; this is not a blind preregistered or multi-seed claim.

## Question

A first nonzero delayed residual supplied enough information to reconstruct RRC-24's sole suffix. This experiment adds two independently chosen hidden unary transforms whose presence cannot be determined from either of the first two delayed onsets. The question is whether a learned nine-step recurrent classifier can rank the remaining sixteen suffix pairs more efficiently than an MLP and exact algebraic previews, under the same exact candidate-verification budget.

## Fixed roster and information boundaries

- Teacher input: two independent integral-unit formal-series input/output demonstrations, observed degrees 0–192, plus a public four-base header decoder. Future query output through degree 256 is grading-only.
- Training: 720 unique cases, 240 each from independent signed, sparse signed, and unrestricted arithmetic-product sources. Development: 144 distinct cases, 48 per training source family. The model input contains only nine coefficient positions from each of two residual windows, normalized as signed log; the hidden suffix labels \((U,V)\) are used **only as supervised training targets**. Teacher supplies the generating header for constructing training examples.
- Test: 192 fresh cases, 32 each from signed, sparse, product, reciprocal sparse-polynomial, convolution-square sparse-polynomial, and late-onset sources. Last three generator families are excluded from all training and development. Tests also include 12 selected observed-output mutations, outside the declared finite grammar, and six base/future-mutant pairs (12 cases). Train/test observed-prefix duplicates are forbidden except the deliberate pairs.
- The late-onset source family is retained as a strong ambiguity negative control. The learned models see a normalized near-zero early signal there, and some U/V effects lie beyond observation degree 192.

## Models and methods

- Recurrent classifier: GRU input width 8, hidden width 12, nine sequential steps, sixteen-output class head, exactly **1,000 FP32 parameters**. Matched MLP: complete 72-feature input, hidden width 11, sixteen-output head, **995 parameters**. A shuffled-label GRU uses the same model and training-data count.
- Each model uses 400 single-thread CPU Adam updates with batch size 96 and learning rate .005. Identical minibatch indices and a fixed independent seed are used for each arm. Held-out data are constructed after training and are unavailable to the optimizer.
- Before any model, the public exact decoder tries all four candidate base operators, detecting allowed delay, spacing and amplitude by first-fault arithmetic across the two demo pairs. It supplies each plausible header to the template-ordering mechanism. Every base evaluation is metered; this is an expressly **interpreter-assisted** student.
- Strict comparison: at most **four full template checks** per case for recurrent, feedforward, shuffled, fixed lexical ordering, and exact short-window preview ordering. Additional recurrent depth budgets of one and sixteen are reported separately. Exact preview ordering evaluates all sixteen short candidate pairs first; its preview operations are charged separately from its full four-check allowance. The complete factorized reference may use up to 64 full template checks, so it is a higher-budget upper control for observed-prefix acceptance.
- Every full submitted candidate executes the two degree-192 demonstrations **independently**, even if the first fails. The result counts only if both match exactly. No model can use the future query, source family, hidden program, or exact candidate outputs prior to issuing its ranked list.

## Acceptance and failure gates

1. Verify the full coefficient identity with an independent coefficient-by-coefficient sum and show two suffix templates that agree at both delayed onsets but later differ on a concrete input.
2. The original source program must pass the exact full-prefix checker for every positive case. The complete factorized search must accept every positive case, rejecting the 12 incompatible public demonstration mutations.
3. A public model input and candidate trace must be invariant when hidden program labels, identifiers and future output are replaced. Every full verifier call and shortened preview must be counted; no unique candidate may be executed twice in a search trace.
4. Audit the multiplicity of simultaneously verified template programs for every held-out case. Report observed-acceptance and future-correct scores separately, especially on the intentionally ambiguous late-onset family.
5. Every six constructed future-collision pairs must produce identical candidate sequences, and complete search must yield exactly six future false acceptances.
6. The neural superiority criterion requires higher **future-correct executable-program recovery** than both the matched feedforward model and the public exact preview search under four full candidate checks, accounting for header decoding and extra previews. A negative result is retained.
7. Freeze three model JSON files, source SHA-256 values, all roster hashes, per-method trace hashes, exact-check counts, and a deterministic full JSON receipt. A separate physical-profile file contains machine-specific times and must not be asserted byte-for-byte in CI. The training reproduction must produce byte-identical checkpoint files.

## Reproduce

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_compound25.py' -v
python research/rrc_compactification/bench25.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_delay24.py' -v
python research/rrc_compactification/bench24.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_scale23.py' -v
# Optional actual CPU profile (timing varies by host):
python research/rrc_compactification/bench25.py --profile --cases 4 --rounds 2 --out /tmp/rrc25_cpu_profile.json
# Optional PyTorch CPU training:
python research/rrc_compactification/train25.py --out-dir /tmp/rrc25_retrained
for kind in recurrent feedforward shuffled; do
  cmp /tmp/rrc25_retrained/rrc25_${kind}.json research/rrc_compactification/results/rrc25_${kind}.json
done
```

CI uses only the standard library for exact algebra, portable frozen-model inference and replay. No model weights or sources are fetched at CI runtime. The supplied short-window exact preview and full factorized search may use more mathematical computation than a four-check neural proposal; that difference is part of the experiment.
