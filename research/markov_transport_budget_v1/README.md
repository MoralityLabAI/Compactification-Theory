# MTB-01: Geometry and target-shift budgets for recurrent monitors

New mathematical campaign, November 14, 2026, pairing previously
unused OpenAI Mathematics results:

- [#327, Markov type characterizes superreflexivity](https://github.com/openai/math/blob/main/lean/docs/327.md)
- [#374, Sharp one-third stability of Brenier maps](https://github.com/openai/math/blob/main/lean/docs/374.md)

[Theorems and proof boundaries](THEORY.md) |
[Executed exact results](RESULTS.md) |
[Original preregistration](PROTOCOL.md) |
[Exact affine and cube reference](reference.py) |
[Independent directed counterexample](nonreversible.py)

## Main connection

Let a stationary reversible process X_t live on a
uniform compact convex source, and let T_mu,T_nu
be its quadratic optimal transport maps into
two target measures. If a post-transport
representation A is L-Lipschitz into a space
with Markov type p∈(1,2], its time-and-target
shift satisfies

    Lp_drift(t)
      <= L * C * Wasserstein2(mu,nu)^(1/3)
       + MarkovTypeConstant * t^(1/p) * Lp_one_step_energy.

This is a conditional synthesis of the two
source theorems. A separately measured
decision margin converts this into a
probability bound on changed decisions.

For Hilbert representations, type 2 has
constant one. An exact affine Brenier/
uniform-refresh chain computes every
term as a rational number. A reversible
hypercube example shows l1^n representation
constants necessarily grow with dimension
for any p>1, despite identical binary
state information. A separate stationary
directed-square control shows why
reversibility cannot be dropped.

## Reproduce

Python 3.10+ standard library, from root:

    python -m unittest discover -s research/markov_transport_budget_v1 -p 'test_*.py' -v
    python research/markov_transport_budget_v1/reference.py --out /tmp/mtb01_exact.json
    python research/markov_transport_budget_v1/nonreversible.py --out /tmp/mtb01_nonreversible.json

All finite probabilities, variances and
source-state transition calculations use
exact Fraction arithmetic. Affine Brenier
transport is verified through independent
moment formulas. Hypercube Hamming moments
are checked with a count-state Markov chain
and full bitmask enumeration for small n.

[Verified GitHub Actions](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43592117700)
ran 23 tests and saved both full JSON receipts.

The separate [MathCan MTB-01 companion PR](https://github.com/patrickdugan/MathCan/pulls)
contains a finite Lean/Init proof of the
four-bit cube's exact moments, the
affine two-step drift/decision numerator,
and a sixteen-state nonreversible
directed-square discrepancy.

## Application to ongoing mechanistic interpretability

This is a controlled test suite for
recurrent semantic-state monitoring:
whether a norm/feature basis permits
dimension-independent drift budgets,
whether the activation process is
reversible, and how a target-distribution
change affects decision stability.

The next checkpoint-level promotion
must freeze model weights, an independently
specified feature geometry, the
transition model, optimal transport
assumptions and an operational
decision margin. The current campaign
does not evaluate a trained transformer.

General source #327/#374 proofs are imported
conditionally; the exact affine family does
not reproduce source #374's sharp
three-atom counterexample. The directed
square was added as a clearly versioned
postprotocol negative control.
