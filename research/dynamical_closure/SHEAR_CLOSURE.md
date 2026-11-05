# Shear exposes the missing state in cyclic addition

This finite-state companion extends the [cyclic-addition note](CYCLIC_GROKKING.md) and the [task-closure framework](TASK_CLOSURE.md). EvalMath owns the general intervention-preserving quotient. The prior CPU modular-addition training runs did not grok; this exact benchmark **does not** test or claim advance warning of grokking.

## Theorem: present output versus joint present-and-future closure

Let p=31, X=Z_p², h(a,b)=a+b=s, and S(a,b)=(a,a+b). Then h(S(a,b))=2a+b=s+a. The map

    (a,b) ↦ (h(a,b),h(S(a,b)))=(s,s+a)

is bijective, with inverse a=h(S(a,b))−h(a,b) and b=2h(a,b)−h(S(a,b)), mod p. Consequently the all-S-words behavioral quotient for h has all p²=961 states: if two states are equivalent, equality at word lengths zero and one already forces them equal. The identity partition is closed and task preserving, so it is the unique coarsest such quotient. Its exact fixed-length state code requires **ceil(log₂ 961)=10 bits**. Present-output prediction alone has p=31 classes and requires **ceil(log₂ 31)=5 bits**. *Post-shear output alone also needs only five bits*: ten is the joint requirement for present and post-shear outputs on the same code, not for the second output in isolation.

The canonical phase χ_j(x)=exp(2πijx/p), 1≤j<p, gives an explicit exact closed representation C(a,b)=(χ_j(s),χ_j(a)) with transition G(u,v)=(uv,v). Its readout recovers s from u, since j is invertible mod prime p. By contrast χ_j(s) alone is not S-closed: states with the same s but different a have different next phases. This separates task sufficiency at time zero from autonomous evolution under shear. The mathematical phases use ideal complex numbers; storing their p² distinctions still needs at least 10 bits of total fixed state precision, or an equivalent external table. The result is a direct finite-state congruence calculation, not a claim about a trained network.

For n≥0, S^n(a,b)=(a,b+na) and h(S^n(a,b))=s+na. Thus the empty and one-step words already settle equivalence for the entire monoid. Fixed-target one-hot squared Bayes risk for the **same code** is allowed to differ between time zero and one here: shear changes the target using hidden a, rather than merely permuting h by a state-independent shift.

## Exact nested risk curves

Under the uniform measure on all 961 states, use Y₀=e_s, Y₁=e_(s+a)∈R^31 and the fixed squared Euclidean target scale. For any integer code C, its Bayes risk is E||Y−E[Y|C]||², computed exactly by conditional class counts (no fitted decoder or sampled estimate). Let L=5.

A *present-first* nested B-bit reference code for B≤L is C_B=floor(2^B s/p). For B=L+j, 0≤j≤L, use (s,floor(2^j a/p)). These are nested partitions; each uses at most 2^B values, though some bit patterns are unoccupied. For 0≤B≤5,

    R₀(C_B)=1−min(2^B,p)/p,       R₁(C_B)=1−1/p.             (14)

For B=5+j, 0≤j≤5,

    R₀(C_B)=0,                   R₁(C_B)=1−min(2^j,p)/p.   (15)

*Proof.* s and a are independent uniform elements under the bijective change of variables (a,b)↔(s,a). Given a bin with m distinct equally likely target labels, the conditional one-hot loss is 1−1/m. A partition into q nonempty bins therefore has risk Σ_bins (m/p)(1−1/m)=1−q/p. For B≤5, conditioning on a bin of s leaves a uniform, so s+a remains uniform and R₁=1−1/p. For B≥5, s is known and each a-bin contains distinct s+a labels, giving (15). At j=5 all 31 a-values are singleton bins. □

This present-first path is **not globally optimal for joint loss at intermediate B**. For a task-balanced second nested reference, allocate ceil(B/2) bits to dyadic bins of s and floor(B/2) bits to dyadic bins of t=s+a. Since (s,t) are independent uniform coordinates, its exact risks are

    R₀=1−min(2^ceil(B/2),p)/p,
    R₁=1−min(2^floor(B/2),p)/p.                             (16)

It reaches the same ten-bit identity quotient. We do not claim that this balanced path globally minimizes any scalarization over all B-bit codes. Both paths are analytically derived references, not learned compressors.

| B bits | Present-first R₀ | Present-first R₁ | Balanced R₀ | Balanced R₁ |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 30/31 | 30/31 | 30/31 | 30/31 |
| 5 | 0 | 30/31 | 23/31 | 27/31 |
| 8 | 0 | 23/31 | 15/31 | 15/31 |
| 10 | 0 | 0 | 0 | 0 |

`shear_exact.py` enumerates every state and recomputes the risks with exact `Fraction` arithmetic; the expressions above are asserted against the enumeration at every B=0,…,10. The tests verify the inverse map, one-round refinement to singletons, nesting, and the entire present-first curve.

## Matched-bit controls, without network training

`shear_controls.py` defines three *code families* over the same 961 states and ideal total budgets B=0,…,10. Each four-coordinate family allocates successive bits round-robin across coordinates, uses nested dyadic bins in a fixed label-independent support interval, and shares the **same unrestricted conditional-mean decoder** used for both references. It reports attained codewords as well as exact risk for the realized integer partitions. The Fourier bank is task engineered, (cos s,sin s,cos a,sin a); the PCA and random-input bank is (cos a,sin a,cos b,sin b). Three additional random orthogonal projections of the Fourier bank compare orientations with the **same source features**. Thus code budgets and evaluation are matched, but *feature construction and encoder complexity are not*. At uniform full-domain sampling, raw input-phase covariance is 0.5I: PCA directions are genuinely degenerate, and canonical coordinate axes are the declared tie-break. A different basis in that tied eigenspace can change low-bit quantization; random rotations expose this sensitivity.

| Code (B=10) | Attained codes (of ≤1024) | Present risk | Post-shear risk |
| --- | ---: | ---: | ---: |
| Exact joint quotient, either nested reference | 961 | 0 | 0 |
| Task-engineered Fourier features, 4-axis quantizer | 312 | 0.1613 | 0.6381 |
| Raw-input PCA with canonical tied axes | 312 | 0.6381 | 0.6632 |
| Random Fourier rotations, seeds 0–2 | 206–219 | 0.5134–0.6499 | 0.6731–0.6820 |
| Random raw-input rotations, seeds 0–2 | 206–219 | 0.6731–0.6820 | 0.7417–0.7645 |

Full exact rational values, the B=0,…,10 curves, counts, seeds and quantizer rules are in [shear_results.json](shear_results.json). At B=10 the task-engineered four-axis phase **quantizer** is not injective, despite the *unquantized* phase representation being exactly closed. The ten-bit theorem is a necessary capacity bound and is attained by a suitable discrete code, not a guarantee for every 10-bit allocation. Random projections are checked at equal budgets; finite projection/quantization collisions explain much of their residual risk. Python tuples are logical packed-bit codes, not claims about their physical in-memory size. The Bayes decoder's **form** is matched across methods, but its attained lookup-table size varies with codewords; these are population recoverability bounds, not equal-cost fitted predictors.

Numerical cosines, PCA tie checks and random rotations use float64 to produce *integer code assignments*. Once assigned, the conditional risks are exact rationals for those codes, not rigorous interval certificates for ideal real-valued phase quantization. Tiny boundary perturbations could change a realized code; the source, seeds, feature order, and bin rule are recorded for replay. This exact finite-state benchmark should precede any trained-network shear experiment. For a separate grokking-warning claim, first select a configuration with independently demonstrated grokking and freeze frequency selection, budgets, intervention targets, and scoring before new confirmation runs.

### Reproduce

From the repository root, standard-library Python for theorem checks and `apy` (CPU NumPy) for controls:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s research/dynamical_closure -p test_shear_exact.py -v
PYTHONDONTWRITEBYTECODE=1 apy -m unittest discover -s research/dynamical_closure -p shear_control_test.py -v
PYTHONDONTWRITEBYTECODE=1 apy research/dynamical_closure/shear_controls.py --output research/dynamical_closure/shear_results.json
```

No GPU, model API, trained checkpoint, or stochastic simulation is used. The historical `receipt.json` and source hashes for the earlier package are unchanged; the separate shear receipt records this extension.
