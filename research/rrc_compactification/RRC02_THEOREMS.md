# RRC-02 mathematical audit: finite Euler inverses and prefix nonidentifiability

These are elementary, known formal-power-series results used to bound the
interpretation of the neural pilot. They are written as exact proofs. There is
no novelty claim or external proof-assistant certificate.

## 1. Uniqueness of the finite Euler exponent vector

Let `F(q)=1+sum_{n>=1} f_n q^n` in `Z[[q]]`. In the field `Q[[q]]`, there is a
unique sequence `a_1, a_2, ...` in `Q` such that

`F(q)=prod_{m>=1}(1-q^m)^(-a_m)`.

To see this, formally differentiate and multiply by `q/F`:

`q F'(q)/F(q) = sum_{n>=1} b_n q^n`, with `b_n=sum_{d|n} d a_d`.

Comparing coefficients of `qF'=F sum b_n q^n` gives

`b_n=n f_n - sum_{k=1}^{n-1} b_k f_{n-k}`.

Then Möbius/divisor inversion gives

`a_n=(b_n-sum_{d|n, d<n} d a_d)/n`.

Both expressions use only coefficients through degree `n`; induction proves
uniqueness. If `F` is a finite product of factors `(1-q^m)^(-a_m)` with
integer `a_m`, inversion recovers those integers. A generic integral series
need not have integral Euler exponents; RRC-01's `euler_exponents` checks
integrality explicitly.

**Consequence:** Input coefficients through degree `T` determine the Euler
exponents through `T`. A periodic binary rule on that finite window can be
checked exactly. A theorem asserting the rule for every future exponent
requires mathematical information beyond the finite window.

## 2. Prefix-indistinguishability theorem

Fix any integer `T>=0`, any partition-counting generating function
`F(q)=sum f_n q^n` with `f_0=1`, and any `K>T`. Define

`G(q) = F(q)/(1-q^K) = F(q) (1+q^K+q^(2K)+...)`.

Then for every `n<K`, `g_n=f_n`. At degree `K`, `g_K=f_K+1`.

**Proof.** Multiply `F` by the geometric series. Terms involving a positive
power of `q^K` have degree at least `K`. At degree `K`, the only new
contribution is `q^K f_0`, which equals one. QED.

**Combinatorial interpretation.** `G` counts the old configurations together
with an additional distinguishable, unrestricted part type of weight `K`.
All coefficients are nonnegative integers. In RRC-02, `K` is 65, 73, 96,
or 112, and the period-5 baseline and mutated families have identical
coefficients through the entire 64-degree training window.

**Decision-theoretic corollary.** Any deterministic algorithm that sees only
`(f_0,...,f_T)` outputs the same prediction when given `F` or `G`. It is
wrong at coefficient `K` for at least one of those two admissible targets.
Under an equal prior on the pair, any randomized predictor has exact
`g_K`/`f_K` accuracy at most `1/2`, unless it receives additional evidence.
Abstention is the appropriate response when a prediction must be guaranteed
for arbitrary unknown future perturbations.

## 3. Finite-period certification and its scope

For `m<=M`, suppose the Euler exponents `a_1,...,a_T` are constant in
`{0,1}` within each residue class modulo `m`, and every residue class
appears at least twice. Let `R` be the residues assigned value one.
By uniqueness in Result 1,

`F(q) = prod_{j>=1: j mod m in R} (1-q^j)^(-1)  (mod q^(T+1))`.

This is an exact **finite congruence** of generating functions. Conversely,
if that product has the same coefficients through `T`, its exponent vector
matches through `T`. The second observation per residue is a model-selection
regularizer, preventing a period larger than the data from simply encoding
all observed exponents once. It does not prove future periodicity.

The trained RRC-02 student proposes candidate residues after seeing an exact
Euler inverse and six summary features per residue. The independent
prefix-certificate checks this finite congruence with an unrestricted-part
DP. A certificate is therefore reproducible evidence about a *fixed finite
truncation*, not a new proof of a Rogers–Ramanujan identity.
