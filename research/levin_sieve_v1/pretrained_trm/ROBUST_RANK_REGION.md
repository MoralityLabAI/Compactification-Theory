# LSV-14: Positive-radius full-ambient goal-history rank certification

**2026-11-04.** Mathematical statement: sound interval forward-mode automatic differentiation can turn LSV-13's point rank into a *uniform* rank certificate throughout a 3,888-dimensional input region. The trusted computational base is the same pinned `python-flint==0.9.0` ball backend used in LSV-13. The frozen pretrained checkpoint and selected columns, Sudoku roster, stage-one-state interpretation and goal digit pairs remain those of LSV-13. This note initially specifies the proof protocol; successful cases must be justified by completed CI receipts.

## Fixed goal-history map and full-space region

For the exact-real recurrent block `F`, the stage-one hidden state `z_1`, the final predicted output digit `a`, second-ranked output digit `b`, and selected cell `j` from LSV-13, define the smooth scalar margin `g(z) = (W_a-W_b) z_j + b_a-b_b` and the four-component goal history

`H(z)=(g(z), g(F(z)), g(F^2(z)), g(F^3(z)))`.

The input region is a **full** `3888`-dimensional box, `U_r={z: ||z-z_1||_infinity <= r}` with *every* hidden coordinate allowed to vary, for a registered positive rational radius `r`. Select the **same four input coordinates** `I` as in LSV-12. Use outward-rounded interval inputs for every coordinate of `U_r`, with four identity tangent directions at `I`.

## Theorem RR-01: Interval minor certifies uniform full-space rank

Suppose rigorous interval differentiation of the real operation graph yields balls `B_ij` such that `dH_i(z)/dz_{I_j} in B_ij` for every `z in U_r` and all sixteen entries, and suppose interval evaluation of the determinant polynomial on `B` yields a real interval `D` with `0 not in D`. Then

`for every z in U_r: rank(DH(z))=4`.

**Proof:** Exact-real GELU, linear layers and LayerNorm with strictly positive epsilon are smooth; their derivative rules use sums, products, exponentials, the error function and positive square roots. The input box contains every allowed input point and the interval arithmetic encloses each primitive operation and derivative at every input. Induction through the computation graph bounds each of the sixteen derivatives simultaneously (though it discards correlations). The determinant, a polynomial of the entries, belongs to its independent interval evaluation for every `z`. A determinant interval disjoint from zero implies the selected `4x4` minor is invertible at every `z`. There are precisely four output components, therefore `rank(DH(z))=4` throughout `U_r`. QED.

## Corollary RR-02: Robust obstruction to smooth compression

On any open neighborhood inside `U_r` where `H=G∘π` with differentiable maps `π:R^3888 -> R^k`, `G:R^k -> R^4`, the chain rule gives `rank DH <= k`. Thus each of the certified neighborhoods requires `k>=4` smooth real coordinates to reproduce the exact four real-valued margins throughout that neighborhood. This is a **local differential obstruction**. It does not establish storage bits, long-horizon goal convergence, or an optimal neural encoder.

## Distinguish preservation of the chosen margin from predicted digits

The goal margin uses a fixed pair `(a,b)`. The rank result remains meaningful if the logits' ordering changes in the box. Separately, the runner checks if final output-logit intervals preserve the top-ranked and second-ranked digits everywhere. Those class-order results carry a different interpretation and are never inferred from the determinant alone.

## Reproduction and evidence

```
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install python-flint==0.9.0
python -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p test_robust_minor.py -v
python research/levin_sieve_v1/pretrained_trm/run_robust_minor.py --fetch --radius 1e-12 --max-cases 64 --out runs/lsv14/r1e-12.json
```

Every result stores sixteen serialized Arb derivative enclosures, a rigorous Arb determinant, and the result of an independent standard-library exact-rational Leibniz determinant replay. A per-case positive answer is only issued when the rational determinant excludes zero. The runner checks that the text serialization encloses each original Arb ball. The source-locked evaluation uses the original LSV-13 case selection to prevent post-hoc selection of easier coordinate directions.

**Scope:** The exact-real lift uses binary32 checkpoint weights and a binary32 observed initial state, and adopts exact analytic erf-GELU and LayerNorm. The certificate assumes the trusted source pin, Arb's outward rounding, and correct implementation of the operation graph. It does not certify binary floating-point hardware, arbitrary initial states outside each region, or Sudoku-task recovery.

## LSV-14 pilot, complete frozen roster

The first `64`-case full-state-ball scan at `r=1e-12` completed on [GitHub Actions 41825389885](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41825389885), passing six implementation tests and all runtime-integrity checks. **0/64** determinants excluded zero under natural interval dependency propagation. Separately, the exact-real class-order bound preserved top-1 for **62/64** cases and preserved both top classes for **29/64**. This is a negative certification result for that method/radius, not evidence that true Jacobian rank dropped. The interval enclosure for the first case's determinant included zero despite a nonzero center point certificate. We therefore test smaller radii with the identical source, case roster, and selection. The original failed receipt is retained as a CI artifact rather than replaced.

The standalone [rational proof replay](verify_robust_receipt.py) uses the 16 serialized derivative intervals per case and matches every selected goal pair and coordinate column to the immutable LSV-13 receipt; it rejects tampering with a determinant, column selection, radius, or source hash. Its checks remain conditional on the intervals having been constructed by the trusted Arb execution.
