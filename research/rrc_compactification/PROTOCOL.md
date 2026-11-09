# RRC-01: test contract and claims boundary

## Fixed deterministic split

- Positive families: adjacent-part gap `2`, minimum part `1` and `2`.
- Trainer input: local-recurrence counts `f_0..f_64`. The trainer has no access to the closed-form `mod 5` target or to coefficients 65–256.
- Algorithm search: smallest modulus in `[1,12]` with a binary Euler exponent pattern that repeats on at least two observed part values in *every* residue class.
- Test: produce all coefficients from the inferred product through degree 256. Report separately the fit errors in `[0,64]` and holdout errors in `[65,256]`.
- Independent exact controls: q-series coefficients and small brute-force partition enumerations.
- Negative controls: gap 3 under the same fit cap; one-step cyclic shift of the first learned residue mask; deliberately mutated held-out target in a unit test.

## Success and failure gates

- Exact equality of the local and staircase generating functions through degree 256 is mandatory. The staircase combinatorial derivation explains this agreement for every degree; the test catches implementation mistakes.
- For each RR family, `a_m` must match the known `mod 5` mask and the recovered rule must predict every held-out coefficient exactly.
- The gap-3 control must either reject a short binary periodic mask on training or expose a concrete held-out failure. It must never be promoted to an infinite identity from a finite fit.
- A shifted mask must fail on the held-out span. Any verified equality there causes the experiment to fail rather than silently discard the control.
- The receipt contains a SHA-256 hash of the exact source file. `--check` recomputes and compares the entire JSON object.

## Resource accounting

- Program bits: Elias-gamma for modulus, then `m` literal membership bits. Exclude the shared rule decoder, library code, and training samples. This number is a syntax-length proxy, not an information-theoretic bound on the original partition family.
- Data bits: full coefficient vectors are unbounded integers, and are accounted for separately; one coefficient may cost many bits.
- CPU complexity: local and product coefficient dynamic programming cost `O(N²)` for fixed gap, while Euler inversion costs `O(N²)` naive arithmetic operations. Big-integer cost depends on coefficient size.
- No model, neural inference, external GPU, probabilistic seed, or LLM-generated results are involved in this phase.

## Interpretation

The infinite Rogers–Ramanujan equalities are known theorems; this experiment is a **positive-control symbolic identification demonstration**. Perfect prediction on 192 withheld coefficients is evidence that the learned periodic grammar recovers the expected finite behavior. It cannot establish equivalence beyond the tested truncation, nor does it demonstrate general-purpose representation compression or self-discovered new mathematics.

## Proposed RRC-02 extension

Build a predeclared pool of constrained partition grammars and a blinded train/dev/test split by *grammar family*. Include invalid periodicity hypotheses and examples sharing a long initial coefficient prefix but diverging later. Require a student to output an executable verifier and program length, with exact holdout checking. Compare against bounded enumerative synthesis at matched CPU/parameter budgets, with recurrent steps `0/1/2/4/8` and randomized residue-label controls. Make the withheld grammar families inaccessible during training and model selection. Register acceptance and false-positive rates before training.
