# RRC-05: algebra-assisted recurrent Euler inversion

**Run status, 2026-11-09:** CPU training and finite-degree evaluation executed. This is a **positive control for an explicitly wired algebraic recurrent circuit**, with a decisive boundary: the learned student was given the exact log-derivative \(b_n\) and divisor graph, so direct discovery from raw coefficients remains unachieved. The parameter-free exact Jacobi procedure gives the same full-depth accuracy.

## What was learned

The two-scalar recurrent student updates its 65-element working state with
\(x^{t+1}=x^t+\alpha(c-x^t-\beta Px^t)\), where \(c=b_n/n\) and the
proper-divisor matrix \(P\) are provided by an exact nonlearned preprocessor.
It learned \(\alpha=0.9987553358\) and \(\beta=0.9999978542\) after
**300 Adam steps** on 384 unique periodic-product training prefixes. The
matched two-parameter, one-pass coordinatewise control learned gains
\(0.3090599775,\ 0.5673363209\), and the shuffled-target recurrence learned
\(0.0461007729,\ 0.9359089136\).

Each model has two trainable FP32 scalars, or **64 bare weight bits**. The
recurrent live exponent vector contains 65 FP32 values, excluding the
supplied coefficients, exact integer arithmetic, divisor graph, result
caches, grammar search, verifier and program decoder. The historical
RRC-04 student controls are much larger and receive only raw coefficient
features. The controls do not equalize external symbolic computation or
inference FLOPs.

## Exact held-out measurements

The six test groups are source-disjoint for the new products and generator
families, with the explicit collision pairs intentionally sharing inputs.
One of 40 local-gap prefixes aliases a training input, and is disclosed in
the receipt; no unseen-periodic prefix aliases training.

| Variant at 7 passes (or natural one-pass output) | 120 unseen-modulus products | 40 local-gap cases | 15 nonperiodic cases | 18 out-of-grammar cases | 12 collision cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact bounded symbolic synthesizer | 120 / 120 | 3 / 40 | 0 / 15 | 0 / 18 | 12 / 12 |
| Exact divisor-graph Jacobi | 120 / 120 | 3 / 40 | 0 / 15 | 0 / 18 | 12 / 12 |
| **Two-scalar trained recurrence** | **120 / 120** | **3 / 40** | **0 / 15** | **0 / 18** | **12 / 12** |
| One-pass two-scalar feedforward | 5 / 120 | 0 / 40 | 0 / 15 | 0 / 18 | 0 / 12 |
| Recurrent, proper-divisor edges erased | 1 / 120 | 0 / 40 | 0 / 15 | 0 / 18 | 0 / 12 |
| Recurrent, shuffled training targets | 0 / 120 | 0 / 40 | 0 / 15 | 0 / 18 | 0 / 12 |
| Frozen raw RRC-04 feedforward | 0 / 120 | 0 / 40 | 0 / 15 | 0 / 18 | 0 / 12 |

Cells are **prefix-verified executable program counts**. For unseen-modulus,
local-gap and other ordinary cases, all accepted programs also predict every
future coefficient at degrees 65–128. **Six of the twelve collision
acceptances are false future predictions**, equally for the exact solver
and trained recurrence. The test deliberately contains six base/mutant
pairs with identical observed prefixes, so at most one member of each pair
can be predicted correctly from that observation. At 7 passes the trained
recurrence also recovers the exact intermediate Euler vectors on *all* 205
non-development test cases, including 70 cases with no in-class finite
periodic grammar, as well as the six adversarial collision pairs. Abstention at the grammar boundary
remains independent of exact Euler inversion.

## Loop-depth ablation: 120 withheld periodic products

| Passes | Exact parameter-free Jacobi accepted | Trained recurrence accepted |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 2 | 21 | 22 |
| 4 | 108 | 113 |
| 6 | 120 | 120 |
| 7 | 120 | 120 |
| 8 | 120 | 120 |

The trained gains allow a few additional coefficients to round correctly
at intermediate depths, but the exact circuit catches up by six on this
roster. Seven is the theorem's uniform bound for **arbitrary** formal
series of degree at most 64, and tests include a degree-64 chain witness
where six exact Jacobi passes are insufficient. Thus iterative computation
helps this supplied algebraic solver, while the *trainable parameters* have
not demonstrated an advantage over the optimal hard-coded circuit.

## Interpretation and cost

The full computational pipeline is:

1. Exact integer log derivative \(f\mapsto b\), requiring quadratic arithmetic
   in the prefix length with big-integer costs depending on the input.
2. Prewired proper-divisor graph (216 directed edges at degree 64), supplied
   independently of learning.
3. Two learned scalar gains reused across up to eight passes over a
   65-coordinate working state, plus integer rounding.
4. Exact integer divisor/series residual checks, bounded grammar search,
   complete prefix replay, and separate future-coefficient evaluation.

The exact solver in step 4 is also a full standalone baseline and can
recover the product class without the trained circuit. The successful
RRC-05 training is therefore evidence that two learned numerical gains can
*imitate the fixed algebraic correction step* across fresh mathematical
inputs. It is not end-to-end neural symbolic discovery or a proof of a
new Rogers–Ramanujan identity.

## Reproduction

Requires only Python 3.10+ standard library to replay:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_rrc05.py' -v
python research/rrc_compactification/bench05.py --check
```

Optional CPU PyTorch retraining:

```sh
python research/rrc_compactification/train05.py --steps 300 --batch-size 96
python research/rrc_compactification/bench05.py --check
```

The reported registered run uses `torch==2.10.0+cpu`, one CPU thread and
deterministic algorithm settings. The saved report
`results/rrc05_pilot.json` binds each source file and checkpoint by SHA-256,
all frozen roster digests, per-case decision hashes, collision checks and
held-out outcomes. The tests require exact JSON equality on replay. CI has
no PyTorch dependency or training job.

**RRC-06 gate:** Remove access to exact log-derivative \(b_n\) at inference;
let a raw-prefix recurrent student generate a compressed approximate \(b\)
trace, optionally using *predicted* divisibility information. Retain an
independent exact verifier only for scoring and final acceptance, freeze a
new holdout before experimenting, and compare against both exact symbolic
synthesis and matched-information feedforward/unrolled controls.
