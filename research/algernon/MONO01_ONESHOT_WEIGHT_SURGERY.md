# MONO-01: conditional-leverage one-shot weight surgery

**Mathematical workorder, 2026-11-04.** Scope: Algernon H-W sparse receiver repair in Compactification-Theory. Source experiment: [Ayush Bodade, Policy Surgery](https://www.latentpaths.com/writings/policy-surgery), which adapted a 4,996-parameter LunarLander actor through GPT-OSS-120B-proposed weight operations. The source's crosswind success-objective seed-0 arm improved by 15/256 paired episodes on an independent final set; other approved edits had weaker intervals. This workorder is a new research plan, not a reproduction of either the LunarLander actor or the Qwen checkpoint.

## Mathematical contract

At a chosen linear receiver `y=Wx`, obtain fault-associated activations `K` from a frozen calibration set and *validated* desired intervention outputs `V` from causal activation patching. Let `C=E_reference[xxᵀ]+λI` be a positive-definite **uncentered** second moment (λ>0) of independent reference activations. The minimum reference disturbance for an exact local correction is

```math
min_Δ tr(Δ C Δᵀ), subject to ΔK=V,
Δ*=V(Kᵀ C⁻¹ K)⁻¹ Kᵀ C⁻¹.
```

Assuming independent constraint columns, this uniquely minimizes the positive-definite quadratic cost and has rank ≤ `rank(V)` ≤ number of keys. For one key, the edit is a rank-one LoRA. It satisfies a **local linear-layer correction**, while end-to-end downstream behavior requires independent measurement.

Given an allowed column support `S` and one key, the optimal constrained disturbance is `||v||²/q(S)`, where `q(S)=k_Sᵀ C_SS⁻¹ k_S`. With a new column `j`, Schur-complement inversion gives

```math
q(S∪{j})-q(S) =
(k_j-C_jS C_SS⁻¹ k_S)² /
(C_jj-C_jS C_SS⁻¹ C_Sj).
```

This is an exact marginal conditional-leverage score, but **greedy support selection has no general submodular guarantee**. For `C=[[1,0,0],[0,1,.99],[0,.99,1]]` and `k=(1.5,1,-1)`, budget-2 greedy selects (0,1), yielding q=3.25, while the optimal correlated pair (1,2) yields q=200, a >61-fold relative preservation-cost penalty. Compare beam and small-width exhaustive searches.

Memory optimization: given reference cache `X[N,n]`, use `C=λI+XᵀX/N` and Woodbury

```math
C⁻¹k=λ⁻¹[k-Xᵀ(NλI+XXᵀ)⁻¹Xk].
```

For selected support, compute `C_SS` from X's selected columns and avoid allocating full `C[n,n]`.

Exact actuator row swap: `d=e_i-e_j`, `ΔW=-d(dᵀW)` for an output-affine matrix with bias absorbed into one extra input. This is rank one. Reported Qwen p50c surgery is likewise a rank-one outer product **per selected down-projection matrix**, with `ΔW=-6u(mask⊙uᵀW)`; this amplifies and reverses the targeted projection to -5 times its original component. Weight touched count, algebraic rank and BF16-effective bitwise support are separate quantities.

For a simple real recurrent eigenmode `Ar=μr`, `ℓᵀA=μℓᵀ`, an idealized linear recurrence repair is `ΔA=(μ_target-μ)rℓᵀ/(ℓᵀr)`. Highly nonnormal systems and nonlinear recurrence require eigenvector conditioning, Jacobian and trajectory checks; do not assert stability from a local eigenvalue shift alone.

## Local CPU reference delivered in conversation

A reproducible MONO-01 ZIP archive was produced with `monoshot.py`, `test_monoshot.py`, `scale_smoke.py`, README, and checksum-verified synthetic receipts. It is an **attachment in the originating ChatGPT conversation**, not currently checked into GitHub. The 15 local NumPy tests passed. A constructed 12-input layer selected two columns; its rank-one correction cost was 0.0003858779 versus 0.0030286972 median across 51 eligible random supports. A second synthetic 1024×3584 experiment with N=64 and 64 selected columns satisfied the ideal linear constraint to 6.94e-18. BF16 merging changed only 23,784 of 65,536 targeted elements and produced 0.00537 maximal realized output-correction error. No pretrained weights, Gym rollouts, held-out task gains, or GPU experiments were run in MONO-01.

## Next implementation gate

1. Import the ZIP's tested source into `research/algernon/mono01/` and rerun all tests and SHA-256 checks; avoid rewriting these proven algebraic pieces before establishing parity.
2. Pin Policy Surgery's source and run its offline `verify`/`reproduce`. Use author-defined development, selection, approval and test episodes; never fit to final test. Rebuild exact rank-one thruster swap on real actor weights, then measure the sensor and output-column edits from retained crosswind proposals.
3. Develop causal activation patching and pair-aware or beam support selection. Compare against same-support/random-direction edits, same-norm sparse edits, one-step gradient, and matched simulator-budget PPO. Check reversibility, retained original-dynamics behavior, paired uncertainty, and multiple-comparison control.
4. Connect only successful, verified repairs to Algernon H-W, leaving H-C/H-R/H-J gates untouched. For recurrence, include transient-growth controls and local Jacobian calibration. For Qwen3.5-0.8B, first reproduce the published frozen edit against base/edited pinned checkpoints and track BF16-effective support, cut rate and loop outcomes.
5. Freeze budgets, candidate counts and train/calibration/test roster hashes before expensive runs. Preserve raw manifests and failed trials. No automatic cloud GPU rental.

See existing [Algernon protocol](PROTOCOL.md), [external-baseline pivot](PIVOT_EXTERNAL_BASELINES.md), and [W track description](README.md) in this directory. Distinguish algebra, synthetic CPU evidence, and actual policy or LLM outcomes in every report.
