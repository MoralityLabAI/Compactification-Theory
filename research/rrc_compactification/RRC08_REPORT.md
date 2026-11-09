# RRC-08: frozen CPU operator-induction result

**2026-11-09, one seed.** The learned eight-real-weight operator fit a 3072-row
training design from exact local supervision at degrees 1–16. After a fixed
integer-code compilation gate it emitted

```
b = [ 1, -1, 0, 0 ]
a = [ 1, -1, 0, 0 ]
```

in the explicitly supplied convolution/divisor basis. The compiled recurrence
runs directly on held-out *raw coefficient prefixes* and exactly reconstructs
Euler exponents through degree 64. This closes the **intermediate exactness**
gap exposed in RRC-06/07 under a stronger algorithmic prior. It is an
induced recurrence with teacher-provided basis, rather than a general learned
symbolic reasoner or a purely neural program generator.

## Measured outcomes

| Method | Euler coordinates / 12,480 | Verified finite traces /195 | Accepted restricted product programs /195 | Correct through degree 128 /195 |
|---|---:|---:|---:|---:|
| Learned, compiled recursive operator (8 FP64 learned weights) | **12,480** | **195** | **115** | **109** |
| Eight-parameter raw-local feedforward fit | 2,807 | 0 | 0 | 0 |
| Exact symbolic Euler+grammar solver | **12,480** | **195** | **115** | **109** |

There are **40 development** and **155 held-out** cases in this table.
On the 155 held-out cases alone, the compiled operator and symbolic reference
accept **75** finite product programs and make **69** correct future
predictions; six accepted collision mutants diverge after degree 64.
The shuffled-supervision control and the raw-local fit fail the registered
small-integer operator compilation threshold, leaving no executable program
under that specific gate. The feedforward model's raw coordinate predictions
are evaluated even though its compiler rejected its weights.

| Held-out generator family | Cases | Exact Euler trace | Product programs accepted | Correct future |
|---|---:|---:|---:|---:|
| Unseen moduli 5/7/11/13/17/19 | 60 | 60 | 60 | 60 |
| Squared product series | 12 | 12 | 0 | 0 |
| Reciprocal product series | 12 | 12 | 0 | 0 |
| Signed `q→-q` transforms | 12 | 12 | 0 | 0 |
| Local-gap partitions | 20 | 20 | 3 | 3 |
| Irregular part sets | 15 | 15 | 0 | 0 |
| Period > 20 products | 12 | 12 | 0 | 0 |
| Six finite-prefix collision pairs | 12 | 12 | 12 | 6 |

The three accepted local-gap cases include the unrestricted distinct-part
partition function and both Rogers–Ramanujan identities. Product-program
abstentions are correct **restricted-grammar** outcomes, even though every
case has an exact Euler-exponent representation.

### Same learned weights, different execution depths

| Simultaneous correction passes | Correct Euler coordinates / 3,840 | Complete Euler traces /60 |
|---:|---:|---:|
| 0 | 1,881 | 0 |
| 1 | 1,881 | 0 |
| 2 | 322 | 0 |
| 4 | 479 | 0 |
| 8 | 930 | 1 |
| 16 | 1,713 | 4 |
| 32 | 2,677 | 17 |
| 65 | **3,840** | **60** |

The zero/one-pass accuracy includes many true zero exponents. Early passes
introduce correct and incorrect nonzero coefficients, so coordinate accuracy
need not increase monotonically. The proven upper bound is 65 simultaneous
passes for 64 degrees; a serial ascending-degree execution solves the same
recurrence in **one topological scan**. The loop-depth success is therefore
architectural/algebraic, rather than independent evidence that repeated
neural thought discovers the correction law.

## Learning, leakage, and costs

Both four-column training designs have full rank. Their condition numbers are
4.189 and 4.979. After 1000 FP64 full-batch Adam steps the learned operator's
largest distance to a snapped integer is about `1.04e-17`; each head's
normalized MSE is below `2e-33`. There was no gradient-based inference-time
student training or unseen-family adaptation. The direct local control has
eight weights and greater per-case raw-feature simplicity, yet is missing the
operator basis essential to this result.

Each program consists of eight `[-2,2]` coefficients, occupying **24 bits**
under the deliberately fixed three-bit signed encoding. The interpreter,
hand-coded convolution/divisor graph, teacher traces, raw input integers,
training, proof checker and coefficient buffers are excluded from that figure.
The trainable real parameter file stores eight FP64 scalars (512 bare bits).
The serial interpreter performs `O(T²)` integer convolution work; the 65-pass
Jacobi ablation repeats it many times. The exact symbolic reference is at
least as accurate and naturally performs a direct triangular solve.

RRC-08 train/development/unseen-period input prefixes are mutually disjoint.
A single local-gap test prefix aliases RRC-08 train. One unseen-period prefix
also aliases each of the earlier RRC-06 and RRC-07 **training** rosters; this
qualification rules out a claim that the entire research sequence is
strictly unseen at the raw-input level. All 36 held-out transformed-operator
cases are disjoint from RRC-08 training.

## Audit and next bottleneck

The standard-library test suite checks formal-series identities on signed
integer sequences, independent local gap and unrestricted product oracles,
invalid-program rejection, source-only inference, parallel depth,
future-collision impossibility, frozen checkpoints and family splits.
`bench08.py --check` replays the saved JSON receipt with source and checkpoint
hashes. Training replay requires PyTorch only when explicitly requested.

**Decision:** The circuit is exact once its basis is supplied and its weights
are identified. Future experiments should *withhold a needed primitive or
operator graph*, require the student to compose it, and compare against an
exact sparse symbolic search with all interpreter costs charged. Success on
that task would be evidence about operator discovery beyond supervised
coefficient identification.
