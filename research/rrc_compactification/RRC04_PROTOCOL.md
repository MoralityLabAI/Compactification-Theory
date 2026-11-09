# RRC-04 protocol: raw coefficients to executable product programs

**Run date:** 2026-11-09. Status: executed CPU pilot, frozen checkpoints,
standard-library replay. This protocol describes the version and split
implemented by the source-bound receipt. The model was not tuned after the
initial test evaluation. Two indexing/assertion mistakes in the original
benchmark's collision audit were corrected, then the same checkpoints were
replayed. RRC-01 source and its Git blob identity remain unchanged.

## Estimand and testable target

The learner sees only a normalized version of exact integers
`(f_0,...,f_64)` from an unknown partition family. It proposes an
*entire* program `(cap, period, residue mask)`, using caps unlimited,
1 or 2 and periods `1..12`. No Euler transform, source-program label,
local gap, family name, source code or future coefficient enters the
model features. The fixed feature transform is the coefficient log,
its neighboring difference and normalized public degree. The fully
observable comparison receives the same raw coefficients in flattened
form. Network logits become discrete programs by argmax for cap/period
and zero-threshold bits for the mask.

The checker directly executes the proposed product to degree 64. If any
coefficient disagrees, it rejects and the result is **abstention**.
Accepted programs are subsequently tested on *all* degrees 65–128.
The checker is shared across all models, has no access to those degrees
before final scoring and supplies the executable finite certificate.

The exact reference first computes the unique Euler exponents from the
prefix and enumerates all cap/period skeletons in the bounded class.
Its shortest verified program is the class-conditional upper control;
its complexity and interpreter are charged separately. A third, clearly
labeled **hybrid** grants a neural-proposed `(cap,period)` access to the
same exact Euler inverse to fill in the mask. Report hybrid outcomes
separately from wholly neural programs.

## Fixed roster and information boundaries

- Optimization: 1,500 unique prefixes from product generators with
  moduli `{2,3,4,6,8,9,10,12}` and caps `{∞,1,2}`, seeded `20261210`.
- Development: 180 independently sampled and prefix-deduplicated product
  cases from the same moduli, seed `20261311`. Hyperparameters are fixed
  for this pilot; development loss is diagnostic.
- Unseen product moduli: 108 independently sampled cases from `{5,7,11}`
  (seed `20261412`). They share product syntax with training and constitute
  a **modulus-disjoint** test, not a new source-family test.
- Fully held-out generator families: 32 gap-restricted partitions for
  gaps 1–4 and minima 1–8; 15 nonperiodic allowed-part series from
  squares, triangular numbers, primes, powers of two and Fibonacci
  numbers, with all three caps.
- Collision controls: eight period-five bases and eight future mutants
  formed by division by `(1-q^K)` at degrees
  `K∈{65,73,87,96,103,112,119,127}`. Both halves of each pair
  share the exact prefix and must induce the same model proposal.
- Every result reports incidental prefix aliases between evaluation and
  training data. The generator *families* remain disjoint, and the
  intentionally duplicated collision pairs are recorded separately.

All training labels are canonical exact programs obtained from the
**training prefix**. The original source program is excluded from the
learner's training features and target. A supervised objective combines
cap cross-entropy, period cross-entropy and masked residue-bit logistic
loss. The held-out local-gap/nonperiodic/defect generators are never
constructed during training (`include_test=False`).

## Models and budget

- Raw sequential GRU, hidden 24, followed by one shared latent refinement
  with a residual update. Exactly 3,363 FP32 parameters.
- Flattened raw-prefix feedforward control, hidden 21 and one output
  layer. Exactly 3,345 FP32 parameters, a 0.54% difference.
- A 3,363-parameter GRU trained with a fixed random permutation of labels.
  All models use 650 Adam updates, batch size 128, learning rate 0.008,
  CPU single-thread PyTorch, deterministic seeds. Training uses only
  the training roster; developmental loss is logged, no checkpoint
  selection by test accuracy is performed.
- Recurrent latent refinements `0/1/2/4/8` run with *one fixed* model
  trained at depth 2. Feedforward and shuffled controls are evaluated
  at their registered depth 2. Shared weights across depths prevent
  depth-dependent parameter increases. Hidden state is reset per case.
- Inference MAC counts include dense recurrent gates, optional refinement
  and output layer. They exclude nonlinearities, source coefficient
  generation, feature transforms, exact verification, data loading and
  Python overhead. Source training cost and FP32 parameter/live-state
  bits are reported separately.

## Acceptance and failure gates

1. Independent recursive enumeration agrees with polynomial product
   coefficients for small degrees and every cap. Recovered Euler mask
   agrees exactly with source membership on registered product samples.
2. The enumerator must accept every in-class product case through degree
   64 and predict degrees 65–128. It must abstain on the registered
   nonperiodic and out-of-class gap examples.
3. The two Rogers–Ramanujan local-gap cases must reconstruct their
   period-five product masks. All other gap samples have explicit receipts.
4. Both members of every planted collision pair must yield identical
   prefix-only proposals. The finite exact solver must disclose the
   induced future false acceptances, rather than certify an identity.
5. Neural direct-program accuracy, accepted-future accuracy, model
   abstention, hybrid cap/period accuracy and exact baseline are reported
   by held-out family and development split. A successful recurrent
   advantage requires improved **direct** held-out program forecasts
   over the matched feedforward model at equal evidence, and evidence
   of a depth effect. Failure is a valid research outcome.
6. A standard-library replay recomputes per-case trace SHA-256, every
   aggregate score and source/checkpoint hashes. Checkpoints are
   verified for shape/parameter budget. CI never launches training.

No finite coefficient result proves an infinite identity. Source program
recovery is unidentifiable under the documented exact equivalences, and
late unseen defects remain observationally ambiguous by construction.
