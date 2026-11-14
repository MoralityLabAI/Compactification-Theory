# SCM-13: Affine intervention disks, topological transition radii and a nonabelian obstruction

**Evidence class:** mathematical proof for exact continuous planar configuration-space curves, an exact-rational sufficient certificate, and synthetic CPU regression. **No trained RMP activations or certified source Hessian bounds have been measured.** The topology is relative to one frozen planar readout chart. All relative braid conclusions concern that chart's collision-free configuration space.

## 1. The affine-disk model

Let an intervention coefficient be `u = r (cos theta, sin theta)` in a real 2D disk. For each of `n` named tokens, let its projected state be

    z_i(theta) = a_i + B_i u(theta) + e_i(theta),

where `a_i in R^2`, `B_i in R^(2x2)`, the residual `e_i(theta)` is continuous and periodic, and `||e_i(theta)||_2 <= eta_i` for **every** angle in the circle. `a_i` and `B_i` must describe the *ideal exact-real source model* or another stated continuous target. The program accepts rational values as premises; no numeric finite difference or sampled maximum is accepted as a proved `eta_i`.

For each pair `i<j`, set `a_ij=a_i-a_j`, `B_ij=B_i-B_j`. When `det B_ij != 0`, the affine pair collides at exactly one coefficient in the 2D intervention plane:

    u*_ij = - B_ij^(-1) a_ij,    r*_ij = ||u*_ij||_2.

The pair difference on the intervention circle is an oriented ellipse:

    q_ij(theta) = B_ij (u(theta)-u*_ij).

Its pairwise winding about zero is `0` for `r<r*`, or `sgn(det B_ij)` for `r>r*`. The equality case has a collision and is rejected. The sign here is winding of labeled `z_i-z_j` with counterclockwise theta; SCM-01's crossing convention uses the opposite sign for braid linking.

## 2. Nonabelian null theorem

**Theorem.** Assume each pair matrix `B_ij` is invertible, `r < min_ij r*_ij`, and the true loop `z` is homotopic to its affine boundary circle without pair collisions (as certified below). Then the entire labeled pure braid of `z` is the identity up to the free-loop basepoint convention. In particular, its faithful Artin action is trivial and its oriented closure is an unlink.

**Proof.** Define a radial deformation of the affine strands

    H_i(s,theta)=a_i + s r B_i(cos theta,sin theta),  s in [0,1].

A pair could collide only when `sr u_unit(theta)=u*_ij`. This requires `sr = r*_ij`, impossible because `sr<=r<r*_ij`. Thus `H` is a map from the whole disk into `Conf_n(R^2)` and contracts the affine boundary loop to the constant configuration `a`. If the true boundary `z` is collision-free homotopic to the affine boundary, their free-loop classes agree. QED.

**Corollary: winding-invisible nonlinear witness.** With all `B_ij` invertible and a proved small remainder, if every pair winding is zero then every affine collision preimage is outside the intervention disk. The full braid is null. Consequently the SCM-03 commutator `[sigma_1^2,sigma_2^2]`, with zero pair linking and nonidentity Artin action, cannot be realized within the certified affine-approximation regime. This is a **sufficient obstruction to an affine response model**, not by itself evidence that a neural network has such a commutator.

## 3. Robust affine boundary certificate

For each full-rank pair, put `U=B_ij^(-1)` and `d=|r-r*|`. The affine ellipse has pair clearance

    m_ij = min_theta ||B_ij (r u_unit(theta)-u*)||
         >= sigma_min(B_ij) d
         >= d / ||U||_F.

The last inequality is exact: `||U||_2 <= ||U||_F`. The true pair differs from the affine pair by at most `eta_i+eta_j`. The straight boundary homotopy remains collision-free whenever

    d > (eta_i+eta_j+margin) ||U||_F.

The code evaluates the equivalent *squared inequality* using exact Fractions, avoiding floating-point square roots. Each critical radius is bounded by outward rational square-root endpoints using `continuous._sqrt_bounds`. If the declared circle touches the enclosing critical interval or fails the strict clearance test, the result is **inconclusive**. A rank-deficient pair is also inconclusive under this initial full-rank certificate.

For a fully certified circle with an enclosed collision in the *affine* disk, the code reports the pair winding, preserving the continuous loop's free-homotopy class under the boundary homotopy. The affine source response may carry a nontrivial higher-order pure braid when several collision preimages lie inside the disk. No arbitrary zero-winding-to-null inference is used outside the full-rank affine regime.

## 4. Taylor/Hessian bound that would connect to RMP

Let `F_i:R^2 -> R^2` be the ideal projected token response at fixed model, task, injection site, visit and frozen chart, and let `B_i=DF_i(0)`. If an independently proved uniform operator bound `||D^2 F_i(u)|| <= H_i` holds on `||u||<=r`, then

    ||F_i(u)-F_i(0)-B_i u|| <= H_i r^2 / 2.

The function `taylor_errors_from_hessian` converts exact certified `H_i` and `r` into `eta_i`. A frozen checkpoint, calibration tokens, interval activation enclosures and a correct Hessian-norm verifier are required to supply such bounds. SCM-06 through SCM-11 currently implement first-order and clean-state controls; they have **not** produced these trained-checkpoint Hessian certificates.

For a composition `F_k o ... o F_1` with `||DF_i||<=L_i`, `||D²F_i||<=H_i`, an elementary conservative Hessian recurrence is

    G_0=0,  A_0=1,
    G_i = H_i A_(i-1)^2 + L_i G_(i-1),
    A_i = L_i A_(i-1).

Tighter tokenwise and source-local bounds are preferable for the RMP eight-visit sequence, where products of global block norms inflate rapidly.

## 5. Sampled first-harmonic diagnostic and its strict status

`scm13_diagnostic.py` reads existing SCM-04 cyclic JSONL captures without changing them. It fits a planar first harmonic `a_i+B_i(cos theta,sin theta)` on the observed angles and reports the **maximum sampled residual only**. It may compare predicted pair winding to the observed polygon. Every result is stamped `empirical_only_unverified_full_circle_error`; it cannot claim a continuous-braid certificate or actual causal feature localization.

A curve can make unobserved turns between samples, including the four-turn circle in the unit tests. The claimed Taylor remainder requires the continuum disk and the exact-real model, plus separately certified error from floating-point capture into the ideal chart. Data-derived Fourier coefficients are chart diagnostics, not source-certified Jacobians.

## 6. Reproducible test plan and validity conditions

- Preserve SCM-12's 64 group-disjoint selection-half calibration tasks with 16 from each family, the original LOOP-T s1 and FF-U s3 C3 cells, feature pair, two fixed readout charts, 32/64 angles and initial radius 0.05. No changes to historical RMP G0 or sealed results.
- First run `scm13_receipt_integrity.strict_compare_files` on the four-arm capture. It checks consistent source type over all tasks, frozen per-model hashes and features, exact cross-model token identities, and complete real RMP capture shape; it then invokes the existing SCM-12 comparisons. Actual source authenticity additionally depends on SCM-12 frozen-manifest verification.
- Run sampled-only first harmonic screening on **selection-half** traces and report null or unresolved results. A later, separately registered radius sweep can test the predicted `r*` only after proving source-level Jacobian/Hessian and numerical chart error enclosures.
- Separate a conditional affine topological theorem from a measured neural phenomenon. Any claim of changed reasoning capability requires matched budget causal prediction outcomes on a held-out, registered dataset.

## 7. Synthetic evidence and boundaries

The `results/scm13_synthetic_reference.json` fixture contains 80 seeded affine pairs and checks the exact rational certificate against independent sampled polygon winding. It also checks a winding-invisible three-strand commutator, which the rank-two affine theorem cannot explain in its certified regime. This reference establishes instrument correctness on controlled examples; it is **not** an empirical result about RMP checkpoints, compactification gains, or the four-dimensional Seifert surfaces of Hayden–Kim–Miller–Park–Sundberg.
