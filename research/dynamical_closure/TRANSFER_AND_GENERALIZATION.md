# Recoverability under distribution and intervention shift

Companion to [TASK_CLOSURE.md](TASK_CLOSURE.md), based on PR #28 head
`01fa566e3072bac0d4ce88a965f0121243d2a7ed` and the [canonical
intervention-preserving quotient](https://github.com/MoralityLabAI/EvalMath/blob/main/research/interventional_compactification/RESEARCH.md).
This note separates a **proved transfer statement** from a deliberately small
finite-state **empirical falsifier**. Conditional-expectation projections and
covariate-shift inequalities are classical; no priority is claimed.

## A quantitative transfer bound for a fixed target

Let X be finite, C:X→Z be a deterministic code, and Y:X→R a **fixed** scalar
future target (for example Y(x)=h(F_w(x)) with w fixed). For a probability
measure p on X define R_p(C;Y)=inf_g E_p[(Y-g(C(X)))²]. Let μ be the
measurement measure and ν the deployment measure. Assume 0<m≤ν(x)/μ(x)≤M<∞ on every positive-mass state, and that both
measures have the same support. Then

    m R_μ(C;Y) ≤ R_ν(C;Y) ≤ M R_μ(C;Y).             (10)

The upper bound alone needs only ν(x)≤M μ(x). For a fixed finite family W
of target words, let R_{p,W}(C)=max_{w∈W}R_p(C;Y_w). The same two-sided
inequality holds after taking maxima. Along nested deterministic codes C_k,
let k_p*(δ)=min{k:R_{p,W}(C_k)>δ}, with min(∅)=∞. Then

    min{k:R_{μ,W}(C_k)>δ/M} ≤ k_ν*(δ)
                         ≤ min{k:R_{μ,W}(C_k)>δ/m}.  (11)

If deployment changes the target to Y', and
σ²=E_ν[(Y'(X)-Y(X))²], then

    sqrt(R_ν(C;Y')) ≤ sqrt(M R_μ(C;Y)) + σ.          (12)

**Proof.** Take g_μ to be a minimizer of the finite-state training risk,
choosing arbitrary values on μ-null fibers. By the upper density-ratio assumption,
R_ν(C;Y) ≤ E_ν[(Y-g_μ(C))²] ≤ M E_μ[(Y-g_μ(C))²].
For every candidate g, E_ν[(Y-g(C))²]≥m E_μ[(Y-g(C))²]; taking infima gives
the lower half of (10). Taking maxima over the same W preserves both bounds.
If R_{ν,W}(C_k)>δ then R_{μ,W}(C_k)>δ/M; if
R_{μ,W}(C_k)>δ/m then R_{ν,W}(C_k)>δ. Monotonicity along nested codes
therefore gives (11), including infinite thresholds. For (12), distance in
L²(ν) to the linear subspace of C-measurable functions is 1-Lipschitz in its
target. Apply the triangle inequality to Y' and Y, then (10) to Y. The same proof applies to a fixed, predeclared distribution
over intervention words if the joint density ratio for (x,w) is bounded and
the predictor may use both C(x) and w. □

In particular, if R_μ(C;Y)=0, absolute continuity (ν≪μ) alone preserves
exact sufficiency for the *same* target even without a finite M. A quantitative
small-to-small guarantee requires finite M, and a failure-transfer guarantee
requires m>0. The threshold bracket is useful only when it is narrow enough
to distinguish compression levels. With μ=(1,0), ν=(1/2,1/2),
C constant, Y=(0,1), risk moves from 0 to 1/4: the support condition matters.
With μ=ν uniform and targets Y=(0,0), Y'=(0,1), risk likewise moves from 0
to 1/4: the bound says nothing about an unrelated intervention when σ is
uncontrolled. The fixed-target monotonicity result in TASK_CLOSURE.md is
correct, but **a measured k* cannot be advertised as intervention-universal**.
Even a small σ bounds only the *optimal* risk; estimating the predictor from
finite data additionally needs a generalization bound, not supplied here.

## A finite test of prediction and its limit

The eight microscopic states are x=(y,b,c)∈{0,1}³, with current task h(x)=y
and uniform initial-state distribution μ over all eight states. The declared
one-step actions are identity, flip_y, flip_b, reveal_b (replace y by b), and
reveal_c (replace y by c). The nested codes are C₀=(y,b,c), C₁=(y,b), C₂=y,
C₃=constant. Each code is a deterministic coarsening of the previous one.
The loss threshold uses the *same scalar task scale and measure* at every k:
R(C_k;Y)>δ with δ=1/16. It is a population Bayes-risk diagnostic, not a
finite-sample confidence claim.

The observational training word flip_b then flip_y has zero task risk for C₂;
C₂ even has exact autonomous transitions under identity, flip_y and flip_b.
We fit per-action transition tables by exact rational conditional means over
all eight one-step training states for identity, flip_y, flip_b and reveal_b;
no tables are fitted on held-out words or on reveal_c. The readout of a
nonconstant code is its y-coordinate. A constant code predicts training mean
1/2. Both evaluation sequences below are unseen **compositions of fitted
one-step actions**. This avoids claiming to infer an entirely unseen action's
transition from data. The model class and full state coverage make estimation
error zero where the transition is exactly closed.

| Representation (k) | Pilot reveal_b Bayes MSE | Held-out flip_b→reveal_b fitted MSE | Held-out flip_y→reveal_b fitted MSE | Unseen reveal_c Bayes MSE |
| --- | ---: | ---: | ---: | ---: |
| (y,b,c) (0) | 0 | 0 | 0 | 0 |
| (y,b) (1) | 0 | 0 | 0 | 1/4 |
| y (2) | 1/4 | 1/4 | 1/4 | 1/4 |
| constant (3) | 1/4 | 1/4 | 1/4 | 1/4 |

The pilot reveal_b gives k*=2 and predicts failure at k=2 for both held-out
compositions, because the same hidden b matters. C₁ is dynamically closed
under all fitted one-step actions and its composed predictions remain exact.
C₂ is sufficient for the observational task, yet its fitted and irreducible
OOD losses coincide at 1/4. However, for the **unseen** reveal_c, k*=1: C₁
already fails because c was discarded. This is a direct counterexample to
using a threshold from b-sensitive interventions as a universal certificate
for c-sensitive interventions. The reveal_c column is an oracle Bayes-risk
calculation, **not** a claimed fitted-model score for an unseen action.

The fixed target, intervention family, state support, and predictor class all
matter. The matched 1/4 loss is a controlled demonstration, not evidence of a
phase transition in a neural model. Real representations may be stochastic,
continuous, and not closed on cached states. The necessary next measurement
is whether a compression path's threshold for a preregistered intervention
family transfers to held-out compositions and related perturbations, while
failing or being flagged for new latent dependencies.

## Reproduce

From this repository's root, using standard-library Python on CPU:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/dynamical_closure -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python research/dynamical_closure/recoverability_experiment.py
```

The runner emits exact rational risks and an explicit distinction between
fitted held-out composition error and oracle error on the unseen action.
No network, randomness, GPU, model API, or external dependency is needed.
