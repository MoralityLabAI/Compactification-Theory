# SCM-17: local margin fibers, transport and the nonlinear head

Let a fixed RMP input `x` produce residual `h_v(x)` at recurrent visit `v`, and let `m(h) = ℓ_i(h)-ℓ_j(h)` be a chosen final logit margin, where `ℓ(h) = W_embed LN_final(h)`.

## Local decision-null fiber

For calibration contexts `x`, compute a matrix of final-state margin gradients `G` whose rows are `∇m(h_T(x))^T`. Let the top `r` right singular vectors be columns of `Q`, and define `P=QQᵀ`. If all nonzero singular directions are retained, `G(I-P)=0`; with rank truncation the residual is generally nonzero. Source-defined future evaluation margins can have additional gradients outside this span. Report the energy ratio `sum_{k<=r} σ_k² / sum σ_k²`; it is a reconstruction metric, not a guarantee of label accuracy.

At any one calibration anchor, for a perturbation `δ` with `∇m(h)ᵀδ=0`, Taylor's theorem gives

    |m(h+δ)-m(h)| <= (H/2) ||δ||²

provided `||D²m||<=H` along the segment. This first-order null is **local**. The exact linear kernel in SCM-16 was specific to its intentionally rank-deficient linear output. RMP's LayerNorm introduces curvature and makes finite patch effects possible even in an instantaneous local fiber.

## Recurrent transport into a decision variable

For a patch at visit `v<T`, with subsequent recurrence `Φ_(v→T)`, the leading margin change is

    Δm ≈ ∇m(h_T)^T DΦ_(v→T)(h_v) δ.

A direction chosen in the final readout's local nullspace need not lie in the nullspace of the composed gradient, so a fiber perturbation can become decision-visible later. Define an empirical mediation ratio from symmetric plus/minus patches on fixed calibration tokens; report terminal and earlier slopes separately. The same RMP weights and input re-injection are used for both arms. A large earlier effect establishes a causal path for that perturbation under those tasks; it is not automatically evidence of improved reasoning skill.

## SAE / visit-conditioned VPD

A fixed D-shared dictionary supplies normalized TopK codes at each `context.after`. Gates vary by invocation while read/write factors are tied. The decomposition `Δ = PΔ+(I-P)Δ` is mathematically exact for the chosen `Q`. This decomposition is **relative to selected task-margin gradients**, not an intrinsic decomposition of weight space. Causal controls use visible-only, fiber-only, reversed visit gates and zero codes without refitting.

The score uses token-correct cross entropy and exact task accuracy from RMP. The initial training provides no geometric braid certificate: SCM-05 through SCM-14 require continuous chart, intervention disk, and verified numerical error bounds, which are a separate evaluation stage. EvalMath local-maxima training and results remain in SCM-16; SCM-17 tests transport on RMP's native pointer and Sudoku tasks, so claims of cross-domain transfer require another controlled experiment.
