# RRC-08: executed CPU protocol and declared boundaries

**Status:** executed 2026-11-09. This document records the implemented
protocol and frozen run, with source hashes and independent replay. It was
written during execution, **after the first development of the code**. It is
not a claim of an externally preregistered or multiply replicated experiment.

## Task and split

- Learn an operator from exact **training-only** `f`, `b` and `a` coordinates
  up to degree **16**. Teacher `b` comes from an independent forward recursion;
  teacher `a` comes from the RRC-01 Euler transform. The learner sees 192 unique
  periodic-product sources and 3072 local supervised rows per head.
- Fit eight free real coefficients with full-batch FP64 Adam for 1000 updates,
  learning rate .06, a fixed zero initialization and one CPU thread. Student
  program is the nearest integer in `[-2,2]` for every learned weight if all
  distances are ≤.125. Otherwise the compiler **abstains**.
- A fresh 40-case development product roster shares the training modulus
  family. In the final evaluation the following are withheld **generator
  families**: 60 product instances with moduli 5/7/11/13/17/19; 12 squared
  products; 12 reciprocal products; 12 signed `q→-q` series; 20 local gap
  constraints; 15 irregular part sets; 12 large-period products; and six
  base/mutant pairs (12 items), for **155** held-out cases in all.
- The learner receives the raw integer prefix `f_0..f_64` and its compiled
  program. Neither teacher state nor exact error/certificate can influence
  inference; the exact verifier runs afterward.
- The source-family holdouts above were excluded from training through
  `roster(include_test=False)`. The four exposed operations (convolution,
  division, previous-state access and proper-divisor sum) are part of the
  supplied vocabulary in every split. Their construction is **not** learned.

## Comparators

- One eight-parameter raw-local feedforward fit sees `f[n]`, `f[n-1]`,
  `f[n-2]` and public degree but has no previous-state convolution or divisor
  graph. This is parameter-matched but differs in computational primitives.
- A label-shuffled operator-fit control must pass the same fixed integer
  compilation gate to count as an executable program.
- The exact symbolic Euler+grammar synthesizer uses one direct algebraic
  inversion and is the correct class-conditional upper comparator.
- Parallel Jacobi evaluation of the **same** learned operator at depths
  0/1/2/4/8/16/32/65 measures how the hand-built dependency graph propagates
  exactness. The serial topological scan is separately costed.

## Gates and evidence

1. A trained operator can be compiled only through the stated integer-snapping
   rule; the frozen true weights are not injected into the interpreter.
2. The compiled circuit must reproduce the exact log-derivative `b` and Euler
   exponent `a` traces for the observed prefix on all tested sources. It must
   pass the coefficient residual and independent product execution before any
   periodic grammar is accepted.
3. The held-out 65–128 coefficient span grades predicted programs, including
   all six late-defect pairs. Exact-prefix validity is never reported as a
   proof of an infinite identity or perfect future generalization.
4. Out-of-program-grammar sources may have an exactly certified Euler trace
   while producing **abstention** from the restricted cap/period/residue DSL.
5. Each compiled program's syntax bits are quoted with the interpreter and
   training computation excluded. Recurrent passes and exact verification
   receive their own compute/resource accounting.
6. A source- and checkpoint-hashed full replay must equal the checked JSON
   receipt byte-for-byte. A mismatch fails CI. Training is optional in CI;
   independently repeated CPU training must reproduce checkpoint bytes to
   support deterministic evidence.

## Leakage and limitations

One local-gap prefix aliases the **RRC-08** training examples. Across older
RRC-06/RRC-07 training rosters, one unseen-period prefix aliases each old
training roster. Past model results are therefore known, and this is a new
seeded *instance and generator family test*, not a fully blind sequence-wide
novelty claim. The compiled universal recurrence generalizes by mathematical
identity once its weights are recovered, so the positive result is strongly
explained by its public bias. Learning arbitrary algebraic primitives,
operator graphs, or new equivalences remains untested.

## Reproduce

From the repository root, with Python 3.10+ standard library:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_operator08.py' -v
python research/rrc_compactification/bench08.py --check
```

Optional independent training (CPU PyTorch available):

```bash
python research/rrc_compactification/train08.py --out-dir /tmp/rrc08-retrained
```

Compare the three JSON files from `/tmp/rrc08-retrained` with the files under
`research/rrc_compactification/results/`. The source-hash check deliberately
fails if the protocol's executable code changes after freezing the receipt.
