# RRC-07: first-fault correction, oracle cost, and information limits

## Lower-triangular first-fault equation

Write `F(q)=Σ_{n≥0} f_n q^n` with `f_0=1`, and let `a_n` be the uniquely determined formal Euler exponents. For an arbitrary integer proposal `x_n`, define

\[
R_n(x)=n f_n-\sum_{k=1}^{n}f_{n-k}\sum_{d\mid k}d x_d.
\]

Suppose `m` is the first index with `R_m(x) ≠ 0`. The lower-triangular equations imply `x_d=a_d` for all `d<m`. At degree `m`, the only contribution of `x_m` to the residual is `-m x_m` (since `f_0=1`). Therefore

\[
R_m(x)=m(a_m-x_m),\qquad x'_m=x_m+R_m(x)/m=a_m.
\]

The quotient is integral for any integer-series prefix with integral Euler exponents. Replacing `x_m` keeps all earlier equations correct, and moves the first bad degree strictly forward. A complete residual query locates the first fault *and* supplies its exact correction. The implementation charges a complete verification on every pass and at termination.

### Inductive consequence

If the initial proposal has `E` incorrect exponent coordinates, first-fault correction in ascending error order makes the complete Euler prefix correct in exactly `E` edits, using `E+1` residual-vector verifications under the registered terminal-check policy. With budget `B`, it fixes `min(B,E)` incorrect coordinates; the next first-error location advances beyond all edited positions. Since `E≤64`, budget 64 always completes the truncated Euler vector when it has integral exponents. The feedback algorithm uses a powerful exact arithmetic oracle, even though it never calls `euler_exponents` internally.

## Complexity and source of advantage

A naïve evaluation of the vector `R(x)` costs `O(T²)` integer arithmetic operations at prefix degree `T`, plus divisor sums. Repeating it `E` times costs `O(E T²)` arithmetic in the reference implementation. The independent exact Euler-series inversion needs `O(T²)` arithmetic once, and the bounded grammar interpreter/replay adds its own finite cost. Any speed or accuracy improvement from **exact residual feedback** is therefore a change of computational interface and must be compared with this analytic solver. The raw-only student has no such interface.

The learned GRU correction at each scan performs approximately `T × [3H(I+H)+3H+3H]` dense multiply-accumulates, with `T=65`, `H=32` and `I=9`, plus nonlinearities and softmax. RRC-06's frozen base contributes its own ~245,180 MACs. Model weight, cached feature and intermediate-state bytes are separate from the syntax bits of a learned periodic rule.

## Irreducible future ambiguity

For any `K>64`, `G(q)=F(q)/(1-q^K)` has precisely the same `f_0,...,f_64` as `F(q)` and differs first at degree `K`. Both the raw-only model and the verifier-feedback policy see identical inputs and residual histories through degree 64 on the pair. Under a uniform prior on each pair, no single deterministic full-future forecast can be correct for both members. This is an information limit and holds even after the truncated Euler vector is recovered exactly.

These statements are elementary properties of formal power series. Computational unit tests verify declared finite cases; there is no novel infinite-identity proof or Lean-kernel certification claim.
