# PEF-06: Exact coalitional memory and Boolean-noise response

The three selected **OpenAI mathematics preprints** are indexed in
[PEF06_SELECTION.md](PEF06_SELECTION.md). The new finite statements are
derived here, rather than imported from the papers as assumptions.

This exercise connects the themes of the papers:
- information retained by a noisy binary summary;
- expected single-bit sensitivity of Boolean threshold rules;
- collective/block interventions that a singleton probe can miss.

The general results of those preprints are substantially deeper
than the finite examples established here.

## 1. Joint finite object

For any integer m>=1, define a Boolean *coalition latch* on 2m bits:

    F_m(x) = XOR_{j=0}^{m-1}(x_(2j) AND x_(2j+1)).

The input word is considered oldest-first, and bit positions use
zero-based indexing. The distinguished intervention baseline
is the all-zero word, although we also evaluate uniform X.

The function's **single-bit sensitivity at zero is exactly 0**:
flipping only one input leaves every AND gate closed.

There are **m disjoint two-bit-sensitive blocks**,
B_j={2j,2j+1}, which each change F from 0 to 1.
In fact the local block sensitivity at zero is exactly m.
For the upper bound, any sensitive intervention set must
contain at least one *entire* AND pair. Disjoint sensitive
sets must contain different complete pairs, and there are
only m available. The m exhibited pair blocks attain
this upper bound.

Every individual bit is pivotal exactly when its partner is
1, independently of other pairs. For independent uniform
inputs, each coordinate's influence equals 1/2, giving

    AS(F_m)=sum_{i=0}^{2m-1} Inf_i(F_m)=m.

At the all-ones input, every coordinate is pivotal, so
the **global maximum sensitivity is 2m**. This example
has a strong *local* block-versus-singleton gap; it
does **not** reproduce the global superquadratic separation
proved by the third selected OpenAI paper.

F_m is also a polynomial threshold function. On 0/1 inputs,
define P(x)=-prod_j (1-2*x_(2j)*x_(2j+1)).
Its sign is +1 exactly when F_m(x)=1. The polynomial has
degree <=2m. The second paper's general average sensitivity
bound is consistent with the sharper explicit AS(F_m)=m;
PEF-06 derives the equality independently.

## 2. Exact noise polynomial and Möbius order

Let independent input bit-flip noise Z_i~Bernoulli(p), where
0<=p<=1/2, act on the all-zero source. Within a gate,
its two bits both flip to one with probability p², and
different gates are independent. The output is the parity
of m Bernoulli(p²) gate events.

Therefore the **exact all-zero activation probability** is

    Q_(m,0)(p) = [1-(1-2*p²)^m]/2.                  (PEF-06 A)

A general Boolean truth table f has the exact local
intervention/Möbius polynomial

    E[f(x XOR Z)] = sum_{S subset [n]} D_S f(x) p^|S|,

where

    D_S f(x)=sum_{T subset S}(-1)^(|S|-|T|)
                   f(x XOR indicator(T)).

Proof: expand the product Bernoulli law in multilinear
coordinates, use the Boolean lattice Möbius transform and
collect the coefficients of p^|S|. This is an identity
for every finite binary source word and every Boolean f.

For the latch at zero, the odd-degree coefficients vanish.
The even-order coefficient at order 2j is

    c_(2j)=(-1)^(j+1) 2^(j-1) binom(m,j), 1<=j<=m.

The first nonzero term is **m*p²**, reflecting the m
minimal two-bit coalitions. Thus average noise sensitivity
around zero is quadratic in small p, while local singleton
tests see zero. For m=3,

    Q_(3,0)(p) = 3p² - 6p⁴ + 4p⁶.

At p=1/4 this is the exact rational **169/1024**.
The implementation checks the general Boolean Möbius
identity for arbitrary small truth tables and exhaustively
compares Q with the independent Bernoulli-mask oracle.

## 3. Uniform-source noise response and information contraction

Now take X uniform on {0,1}^(2m), Y=X XOR Z with iid BSC(p).
A single AND gate changes value with probability

    q_gate = p - p²/2.

Consequently the sign correlation between its noisy and
original output is (1-p)². Independent XOR of m gates
multiplies the signed correlations:

    E[(-1)^F(X) (-1)^F(Y)] = (1-p)^(2m).

Therefore

    Pr(F_m(X) != F_m(Y)) = [1-(1-p)^(2m)]/2.      (PEF-06 B)

For fixed m the small-p behavior is **m*p**, unlike the
all-zero baseline's m*p². Source distribution changes the
leading order of observed noise sensitivity.

Write A=2^(-m) and C=(1-p)^(2m). For a,b in {0,1},
the *exact* full output-to-output joint law is

    Pr(F(X)=a, F(Y)=b)
      = [1 + (-1)^a*A + (-1)^b*A
           + (-1)^(a+b)*C]/4.                    (PEF-06 C)

It follows because the original and noisy parity signs
both have mean A, with signed product correlation C.
The implementation independently enumerates all x and
noise masks for m<=3, in exact Fraction arithmetic.

At m=3,p=1/4, the joint law with denominator 16384 is

| original F | noisy F=0 | noisy F=1 |
|---|---:|---:|
| F=0 | 5849 | 3367 |
| F=1 | 3367 | 3801 |

The uniform-source output disagreement is **3367/8192**.

The first OpenAI paper bounds I(F(X);Y), where Y is the
**full** noisy Boolean input, by the information retained
by a single noisy fair coordinate. We separately compute
I(F(X);F(Y)), with a further lossy one-bit observer.
The data-processing relation is

    I(F(X); F(Y)) <= I(F(X);Y) <= 1-h2(p).

The exact distributions used before entropy conversion
are rational; only final log2 computations use floating
point. For m=3,p=1/4, direct enumeration gives
approximately 0.0196532 bits, 0.0331960 bits,
and the single-coordinate upper curve 0.188722 bits,
respectively. This is a sanity check of the selected
paper's statement. PEF-06 does not reprove the general
Courtade–Kumar theorem.

## 4. What changes in the looped-transformer evaluation

The PEF-03/04/05 neural control work measured single
old-input counterfactuals and a train-only mean signed
history margin. A nonlinear recurrent state can also
contain interactions between different history bits.
PEF-06's companion frozen checkpoint gym, on
[LoopedTransformers PEF-06](https://github.com/MoralityLabAI/LoopedTransformers/tree/research/pef06-frozen-coalition-probes-20261115/research/pef06),
systematically measures all 15 two-bit older-history
interventions on the 256-point finite source cube for
every previously frozen PEF-05 model.

The neural measurement is posthoc, and the latch is
a mathematical calibration. No learned model is
assumed to compute F_m; the experiment records actual
task-class changes and mixed hidden-state/logit
finite differences.

## 5. Reproduce the exact proof oracles

    python -m unittest discover -s research/precision_entropy_frontier -p 'test_pef06_coalition.py' -v
    python research/precision_entropy_frontier/pef06_coalition.py --suite --out /tmp/pef06_exact.json

The independent Lean companion is in
patrickdugan/MathCan/MathCan/CoalitionNoise.lean,
including finite m=3 bit-intervention and exact BSC
probability certificates. General all-m probability
and Möbius formulas are mathematical proofs in this
note, with bounded exhaustive Python differential tests.
