# Algernon: belief geometry, evidence cubical complexes, and stopping

**Evidence class:** exact finite CPU reference and proposed 27B mechinterp tests. No GPU or Goodfire credits consumed. All finite-model likelihoods, hypotheses, and observation costs are known by construction.

## Mathematical statements

### 1. Evidence is a commutative semigroup action in log-odds coordinates

For a positive prior `b` on a finite hypothesis set and conditionally independent observed evidence events `e`, let `D_e=diag(P(e|h))`. Unnormalized posterior weights are `D_e b`; their projective normalization leaves a probability distribution. Diagonal matrices commute, hence the resulting posterior is invariant under reordering the **same multiset of observations**. A decision policy may still depend on collection costs and adaptive availability.

In coordinates `ell_i = log(b_i/b_ref)` we have `ell' = ell + v_e`, with `v_e[i] = log P(e|h_i) - log P(e|h_ref)`. A proposed low-dimensional neural representation `W(h_{t+1}-h_t) ~= v_e` should be tested with held-out evidence families and with controls for chronology, input wording, and decoder collapse. For a linear readout `W` and a matrix `V` of target increment vectors, exact representation of all columns requires `rank(V) <= rank(W) <= hidden_subspace_dimension`. This is necessary, not sufficient, for faithful computation.

### 2. Discrete curvature and a path-independence bound

Let `g(S,e)` be an Abelian-vector-valued increment for adding unseen event `e` to evidence subset `S`. For each square face indexed by distinct unseen `e,f`, its curvature is

`Omega(S;e,f)=g(S,e)+g(S+e,f)-g(S,f)-g(S+f,e)`.

If every square has zero curvature, all orders of any fixed subset have identical accumulated increment. **Proof:** adjacent transpositions connect any two permutations; one swap changes the path integral by one square curvature. More generally, if all squares have norm at most `epsilon`, the difference between two paths is at most `inversions(path1,path2)*epsilon` by the triangle inequality. This bound is exact for the **edge-accumulation** representation. A hidden-state recurrence with future nonlinearity requires separate stability or congruence assumptions before transferring the bound to neural activations.

The filled hypercube is contractible, so this is a discrete integrability test rather than evidence of a nontrivial first cohomology class. Inconsistent *observed edge assignments* appear as nonzero two-dimensional curvature, not a topological hole. The earlier three-cycle sheaf obstruction in `../algebraic_machine/` is a different construction on an unfilled graph cycle.

### 3. Stopping policy has exact alpha-vector geometry

For a finite-horizon, known finite POMDP with state-independent per-action costs and terminal rewards, every conditional plan is an `alpha` vector of expected rewards per hidden hypothesis. Choosing the plan with maximum expectation yields `V(b)=max_alpha alpha.b`, a convex piecewise-affine function on belief space. `model.FiniteHorizon` enumerates these plans exactly using rational arithmetic, allowing each observation to select a different downstream plan.

The stop/continue advantage `G(b)=max_{alpha: first_action != STOP} alpha.b - max_{beta: STOP} beta.b` is piecewise affine, with sign-changing decision boundaries where plans tie. This identity holds for fixed available actions and remaining horizon; changing the action mask changes the set of alpha vectors.

**Decision-value counterexample:** two independent fair bits `X,Y`, reward for predicting X. Probe A reveals Y perfectly. Probe B reveals X with 25% error. Both cost `1/20`. Mutual information about `(X,Y)` is 1 bit for A, about 0.1887 bits for B. Yet terminal values are **STOP 1/2**, **A 9/20**, **B 7/10**. The best costly probe is B, illustrating how entropy reduction can diverge from terminal utility.

### 4. Approximation certificate connecting log-odds to stopping

For a finite posterior, softmax's Jacobian has induced `L_infinity -> L_1` operator norm at most 1. Therefore if all **log-odds coordinates** including a fixed zero reference have error at most `delta`, then the decoded posterior error is at most `delta` in `L_1` (indeed at most `2*tanh(delta/2)` by global ratio bounds, but our local bound suffices).

If a plan has payoffs in `[m,M]` for every hidden state, the expectation error under two distributions is at most `(M-m)*TV(b,b_hat)`, and `TV=||b-b_hat||_1/2`. The max over plans is also Lipschitz with that bound. Thus the stop/continue advantage estimate differs by at most `(M-m)*delta` when the same alpha plan family is applied to `b_hat` and `b` (twice the half-bound). A verified stopping-margin larger than this threshold has a certified sign. When model likelihoods or cost contracts are wrong, this certificate need not apply.

A constant decoder would satisfy evidence-order commutation trivially. Strong neural claims therefore require posterior correctness, noncollapsed state information, and held-out intervention effects alongside commutator tests.

## Executable reference

```sh
python -m unittest discover -s research/algernon/belief_geometry/tests -v
python -m research.algernon.belief_geometry.run_reference --out /tmp/belief_geometry_reference.json
```

`model.py` uses only the Python standard library, with `fractions.Fraction` to avoid floating-point false positives. The fixed reference receipt is in `results/reference.json`. Do not substitute source-free synthetic fixture results for mechanistic evidence on Qwen.

## Qwen3.5-27B mechanistic experiment (GPU phase, not executed here)

Source model: [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), 64 language layers with interleaved Gated DeltaNet and attention blocks. Lock full HF revision and tokenization before a paid run; verify Silico supports activation capture on this checkpoint.

- Generate held-out *hypothesis-system families* with exact Bayesian likelihood tables and cost-weighted terminal decisions. Freeze 256 tasks as a discovery collection and at least 128 independent final cases, balancing positive/negative stopping advantage at similar entropy and including near-boundary decisions. Preserve source parameters and exact rational labels.
- Present fixed multisets of independent observations in multiple permutations and matched paraphrases. For each ordered sequence, extract hidden states at six candidate language layers, with explicit tracing of residual sites and token boundaries. Fit linear probes for log-odds increments, stopping advantage, and policy region. Use disjoint families for fit and final evaluation.
- Measure approximate square curvature and the path-distance bound on *decoded log-odds increments*; compare to exact Bayesian invariants and token-order/position baselines. A good prediction score without a causal change is an observational correlate.
- Choose mechanistic directions on calibration only. Patch/ablate on held-out matched task pairs with fixed action budgets and cost contracts. Score change in decision regret, stopping errors, numerical probability calibration and total metered resources. Random/norm-matched, label-shuffled and constant-state controls are mandatory.
- Run one batch script against cached task/state manifests and reuse extracted activations. Account for GPU-hours, input/output tokens, extraction bytes and RU pricing. A single bounded run should produce content-addressed receipts and a concise final report. Store activation arrays out of Git.

## Scientific boundaries

This module establishes mathematical properties of a known finite Bayesian model. It makes **no empirical claim** about Qwen, neural circuit existence, SAE sparsity, transfer to the Ryzen NPU, or terminal utility improvements. Mathematical continuation value and information-gain objectives are different quantities. A 27B activation experiment would test whether the model approximates these structures and whether they causally matter.

## 5. Information geometry: flat evidence increments, curved decision surfaces

Let `theta` be a categorical natural-parameter vector. Its log-partition is `A(theta)=log sum_i exp(theta_i)`, with gradient `softmax(theta)` and Hessian `diag(p)-p p^T`, the Fisher information of the categorical family. The Hessian has a zero mode along the all-ones vector because adding a common scalar to every logit leaves probabilities unchanged. In the dimension `n-1` reference-gauge log-odds coordinates the Fisher metric becomes positive definite for strictly positive posteriors.

**KL identity:** `KL(p_theta||p_eta)=A(eta)-A(theta)-∇A(theta)·(eta-theta)`, a Bregman divergence. Thus the additive evidence translations in natural coordinates coexist with a generally non-Euclidean Fisher metric. Tests in `tests/test_info_geometry.py` check the gradient, Hessian, gauge invariance and KL identity numerically.

For two fixed terminal policies with alpha vectors `alpha,beta`, their decision boundary in natural coordinates satisfies `sum_i (alpha_i-beta_i) exp(theta_i)=0`, after cancellation of the positive partition function. It is generally **curved** in log-odds coordinates, though affine in mixture belief coordinates. For three states and `alpha=(1,1,0), beta=(0,0,1)`, fix `theta_3=0`: the boundary `e^x+e^y=1` has `y(x)=log(1-e^x)` and `y''(x)=-e^x/(1-e^x)^2 < 0` for `x<0`.

**Mechanistic implication:** a linear readout for additive log-likelihood updates and a nonlinear readout for action advantage can both be valid. A curved stopping feature in activation space alone does not falsify flat Bayesian evidence accumulation. Source model and exact activation site must be pinned before interpretability measurements.

## 6. Sharp Hodge certificate: local curvature controls global evidence inconsistency

On the filled n-dimensional Boolean cube, let `B` be the oriented vertex-edge incidence matrix, `C` the oriented edge-face incidence matrix, and `g` a real-valued observation increment on each directed positive edge. They obey `B C = 0`. The globally consistent fields are gradients `Bᵀ phi`. Define the closest global posterior potential by least squares, `phi* = argmin_phi ||g-Bᵀ phi||₂`, and set residual `r=g-Bᵀ phi*`. The face curvature is `curl(g)=Cᵀg`.

For the **unweighted, oriented** standard cube and `n>=2`, the sharp bound is

`||g-Bᵀ phi*||₂ <= (1/2) ||Cᵀg||₂`.

**Spectral derivation.** The edge Hodge Laplacian is `L1=BᵀB+CCᵀ`. By the tensor decomposition of n interval cochain complexes, its eigenvalues are `2k` with multiplicities `n*binomial(n-1,k-1)`, for `1<=k<=n`. The gradient sector has the nonzero vertex-graph eigenvalue `2k` with multiplicity `binomial(n,k)`. Since the fully filled cube has trivial H¹, the orthogonal complement of gradients is `im(C)`, with strictly positive eigenvalues `2k` for `k>=2`. Thus `⟨r,CCᵀr⟩>=4||r||₂²`. Using `Cᵀr=Cᵀg`, the claimed factor `1/2` follows. It is sharp: a circulation on the (0,1) coordinate square repeated across other coordinates saturates it.

This creates a **global consistency certificate from local face measurements**. It is specific to the full unweighted cube and additive decoded evidence increments; missing faces, dependent measurements, or nonlinear state evolution alter the complex and its spectral gap. It is distinct from proving Bayesian numerical correctness, which also needs an anchored prior and correct per-event likelihood increments.

`hodge.py` numerically verifies the Hodge spectrum and sharpness for n=2..5 and tests random corrupted fields, using optional NumPy. Python unit tests are finite evidence for the implementation; the spectral derivation is a mathematical argument, not a Lean kernel proof here.
