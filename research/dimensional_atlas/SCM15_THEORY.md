# SCM-15: Causal dimensional atlas and Atlas-Transport VPD

**Research note (2026-11-14).** This note follows the SCM-01–14 Seifert/mechinterp campaign, especially [SCM-14 PR #134](https://github.com/MoralityLabAI/Compactification-Theory/pull/134). The initial gym below is a new, independently runnable research track. All tests below use synthetic math tasks until the separately pinned EvalMath and RMP artifact dependencies become available. No knot in a trained neural representation, compression gain, or causal attribution is claimed.

## 1. Local dimension versus causal dimension

For recurrent hidden state `h^(v)(x;u)` and a frozen intervention map `A:R^k -> R^d`, form `F_v(u)=h^(v)(x;u)` at a specified token/site. After quotienting known channel-constant LayerNorm gauge directions, define `J_v=D F_v(0)` and `H_v=D^2 F_v(0)`. An epsilon-rank is a *thresholded numerical diagnostic*, not an algebraic rank certificate. If `sup_{||u||<=r} ||D^2 F_v(u)|| <= H` has been independently certified, singular-value perturbation yields

    sigma_j(J_v(0)) > H*r  =>  rank D F_v(u) >= j on the disk.

If `J_v` has constant rank, let `P_v` project onto its tangent image and define `II_v(a,b)=(I-P_v) H_v[a,b]`, the extrinsic second fundamental form for the immersion. It must be evaluated after fixing the input metric and chart. Principal angles between tangent images across visits describe subspace transport; a visit-index label alone does not prove a new computation.

## 2. Atlas-Transport VPD (AT-VPD)

The baseline is a strictly weight-tied transformer block `B_theta` applied for `T` visits. A shared top-k SAE `D` learns local feature coordinates from the baseline's residual states, on discovery training samples only. A new **visit-conditioned parameter-fiber adapter** augments a named residual/MLP site:

    a_v(x) = TopK_SAE(h_v(x))
    g_v(x) = tanh(W_route a_v(x)) * tanh(g_visit[v])
    Delta_v(h) = U diag(g_v(h)) V^T h
    h_(v+1) = B_theta(h_v) + Delta_v(h_v).

Here U,V are shared rank-r factors; the sparse coordinate router and visit gates implement transport of a **shared parameter fiber** across repeated use. The base model and SAE are frozen while fitting AT-VPD. The adapted system is explicitly **visit-conditioned** and is no longer an identical effective block at each visit; evaluation always includes the pure tied baseline and a visit-gate-permuted control. The module is inspired by VPD-style low-rank weight decomposition, not a reproduction of Goodfire's fitted VPD algorithm or a claim of a recovered true parameter decomposition.

The initial objective is exact-oracle supervised math correctness plus a small adapter penalty:

    L = L_program + lambda_adapter * ||Delta||^2 + lambda_sae * L_reconstruct.

Sparse reconstruction is trained in a separate phase, and the adapter is subsequently fitted on frozen SAE codes. Later stages may add a causal alignment objective comparing held-out interventions with exact intermediate variables. An optional head-orthogonal constraint can be added only after measuring readout-space rank and ensuring that the constraint does not destroy causal utility; VPD v012 shows head-orthogonal search can matter.

## 3. Dimensional atlas record

For each visit, task family and checkpoint revision, report a chart-fixed tuple:

    (effective Jacobian singular values, tangent principal angles,
     normal curvature, SAE support/feature drift, AT-VPD gate use,
     exact verifier pass, heldout behavior, token/forward costs).

SCM-05/13/14 supply sufficient collision-free homotopy and Taylor-bound conditions for planar braids. The gym's initial finite differences/SVDs are **empirical diagnostics only**, not continuous topology certificates. No property of a chosen 2D chart is asserted as intrinsic to R^d.

## 4. EvalMath-driven frontier generation

Owner of the exact formal-series oracle: `MoralityLabAI/EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py`; use the existing ARG-01 pin at `research/algernon/algebraic_reasoning_gym/evalmath_pin.json`. A paid candidate check returns exact observed-prefix acceptance and first-fault degree. Selection of candidate programs is complete before future coefficients are revealed for grading.

On a discovery-only set, define an **empirical failure ridge** of tasks on which the model assigns high confidence to a candidate that the independent prefix checker rejects late, or the independent future grader falsifies after a prefix acceptance. Search neighborhoods by fresh source-family seeds, graph and topology perturbations; deduplicate public signatures, group siblings, and create a separate generated-eval roster with a private gold sidecar. The phrase 'local maximum' refers to a measured maximum of a *specified proxy under a finite neighborhood*, not a theorem about the model's full loss surface. Avoid reusing public EvalMath tests as new final evaluations.

The synthetic finite-ring source in `gym/tasks.py` exercises the entire pipeline offline. Production use imports EvalMath's pinned oracle and reports source revision, calibration/discovery/test access, exact query budget, independent future grading, and resource count. The official EvalMath test roster is read only for the frozen evaluation and never mined for training or task selection.

## 5. Falsifiers and gates

* A topology/curvature metric changes with random chart but fails causal interventions: reject a mechanistic claim.
* SAE reconstruction improves while intervention-change fidelity or verified math accuracy declines: report the regression.
* AT-VPD beats the tied baseline only at greater FLOPs or a reused discovery score: compare matched computation and fresh family-disjoint test blocks.
* A generated task repeats an existing public signature, differs only by ID, or exposes gold to model input: reject generation.
* A model's high-confidence success on a finite prefix disagrees with future exact coefficients: record the finite-prefix ambiguity instead of assigning skill.
* Existing EvalMath receipts, RMP G0, and sealed splits remain immutable.
