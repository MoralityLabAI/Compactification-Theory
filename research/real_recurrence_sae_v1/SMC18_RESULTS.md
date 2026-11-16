# SMC-18: Clean task gradients predict most finite SAE selective-kill effects

**Executed November 16, 2026.** This source-locked gradient experiment was prospectively registered at [commit 70dc6a3](https://github.com/MoralityLabAI/Compactification-Theory/commit/70dc6a3e48097aeed02ccf1a10623a0cbbd624e5) *before* generating any fresh SMC-18 heldout state. It follows [SMC-17 PR #179](https://github.com/MoralityLabAI/Compactification-Theory/pull/179), whose registered gate remains **1 of 4** pairs, with original results unchanged. Its pure-mathematics companion is [MathCan OLM-18 PR #102](https://github.com/patrickdugan/MathCan/pull/102).

The [complete source-pinned GitHub Actions run #43807953506](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43807953506) passed the pilot unit/regression suite, full four-pair × two-family × 64-fresh-puzzle evaluation, and independent **standard-library receipt audit**. The original 131-candidate outcome-blind sham search and original source model/SAE/artifact SHA hashes were preserved. SMC-18's scientific success gate passed in **3 of 4 receiving teacher/SAE pairs**, evaluated on *both* new cohorts.

## Experimental boundary

Source pretrained recurrent teacher A and SAE71, independent recurrent teachers B501/B701 and their SAE71/89 models are frozen. All seven original SMC-13C clue-and-position-conditioned Hungarian feature mappings are reconstructed and verified on the original calibration split **before any new test state exists**. New noncanonical exact-unique Sudoku families use grid seeds **202618030000..202618030063** (30 blanks) and **202618040000..202618040063** (34 blanks), with full solutions and inputs disjoint from all prior SMC-13/15/16/17 cohorts. Each family contributes 64 independent puzzle clusters; eight visible-mask-selected blank goals are scored per puzzle.

For an eligible goal, the target matched atom is naturally among the top-four SAE features. Reuse the exact minimum-norm SMC-16 active-feature erasure \(k\) and SMC-17's label-blind 131-candidate inactive-preactivation-matched tangent sham \(s\), verifying equal physical L2 dose, preserved active-feature codes, and rank-48 previously inactive 92×48 encoder geometry before reading outcomes. Both branches modify one of the 81 stage-one token states, followed by the original three FP64 recurrent blocks.

The primary finite effect is \(D=m(h-s)-m(h-k)\), where \(m\) is the *correct Sudoku digit* minus the receiving teacher's strongest **clean** competing digit logit. Clean FP64 autograd gives the gradient \(g\) with respect to the target goal token's original stage-one 48-dimensional state. The first-order predictor \(L=g^\top(k-s)\) and finite nonlinear remainder \(R=D-L\) are scored on the exact same eligible goal and source-pinned checkpoint.

The preregistered gate per pair requires at least **24 eligible goals in 20 independent puzzles**, a positive lower 95% puzzle-bootstrap bound on mean D, at least 24 nonzero sign-comparable goals with a positive lower bound on sign agreement minus one-half, and \(\sum|D-L|/\sum|D|\le0.50\). Both fresh puzzle families must pass; 2,000 seeded resamples cluster by puzzle ID.

## Registered task-gradient result

| Receiving SAE | New 30-hole: finite D [95% puzzle-bootstrap] | Eligible goals / puzzles | New 34-hole: finite D [95% puzzle-bootstrap] | Eligible goals / puzzles | Both-family gate |
|---|---|---:|---|---:|---|
| B501 SAE71 | **+1.3835 [0.6980, 2.0419]** | 36 / 30 | **+1.2941 [0.7367, 1.8363]** | 40 / 31 | **PASS** |
| B501 SAE89 | +0.6936 **[−0.6649, 2.0623]** | **22 / 18** | +0.9728 [0.0134, 1.7744] | 24 / 20 | **FAIL (novel30)** |
| B701 SAE71 | **+1.4042 [0.5794, 2.1925]** | 45 / 36 | **+0.9063 [0.2404, 1.7071]** | 41 / 32 | **PASS** |
| B701 SAE89 | **+1.8707 [1.2464, 2.4603]** | 50 / 35 | **+1.4068 [0.8255, 1.9594]** | 58 / 39 | **PASS** |

The B501 SAE89 new-30 condition has 22 eligible goal tokens across 18 puzzles, below the two frozen minimums; its mean D interval also crosses zero. It therefore fails the overall pair gate. The linear remainder ratio and raw sign agreement are reported but cannot revise the registered coverage/finiteness result.

## How predictive is the clean task gradient?

| Receiving SAE | 30-hole gradient L | 34-hole gradient L | \(\sum|R|/\sum|D|\), 30/34 |
|---|---:|---:|---:|
| B501 SAE71 | +1.2415 | +1.2545 | 0.145 / 0.149 |
| B501 SAE89 | +0.4945 | +0.9917 | 0.147 / 0.129 |
| B701 SAE71 | +1.2772 | +0.8748 | 0.176 / 0.163 |
| B701 SAE89 | +1.7138 | +1.3798 | 0.164 / 0.142 |

Across all **316 eligible goal/pair/family observations**, the clean gradient had the same finite-effect sign on **310/316 (98.1%)**. The pooled total absolute first-order remainder was **15.48% of the total absolute finite effect**, with the eight registered per-family ratios ranging **12.89%–17.64%**. This is a strong local predictor of the measured finite task-margin response on the *selected eligible subset*. Each of the four teacher/SAE comparisons shares architecture, generator and some checkpoint dependencies, so 316 goal observations do not constitute 316 independent model replications.

### Second-order curvature diagnostic, explicitly secondary

The evaluator also ran four additional FP64 continuations per eligible goal, probing \(h\pm0.05k\) and \(h\pm0.05s\). Finite central curvatures \(q_k,q_s\) yield the second-order predictor \(Q=L-(q_k-q_s)/2\). The pooled total absolute prediction error decreased from **104.62** for L to **70.44** for Q, a **32.67%** reduction. Condition-specific outcomes are heterogeneous: the B701 SAE71 novel34 case worsened slightly under the local second-order approximation. The quadratic diagnostic is calculated at small radius and **does not certify** the third-order remainder along the full original feature-kill amplitude.

## Fundamental inactive-feature dual and representation ambiguity

For every one of the **316 eligible cases**, the 92×48 *originally inactive* physical SAE encoder block \(B\) had **full column rank 48**. The minimum observed singular value across all cases was **0.2758**. The frozen clean physical task gradient \(g\) was reconstructed as

\[
\lambda_{\min}=B(B^\top B)^{-1}g,\quad
B^\top\lambda_{\min}=g,
\]

with a maximum observed FP64 reconstruction L2 residual **1.88×10⁻¹⁴**. This mathematically valid inactive-coordinate representation covers **all** of the observed physical task-gradient directional effect. The associated Cauchy–Schwarz ceiling \(\|\lambda_{\min}\|\|B(k-s)\|\) had a median ratio of **4.26** to the magnitude of the observed linear prediction, so the bound is generally too loose to isolate a particular active feature's contribution.

The mathematical [OLM-18 theorem and exact-rational tests](https://github.com/patrickdugan/MathCan/pull/102) prove that **every** physical gradient admits such a minimum-norm inactive-feature dual when \(B\) has full column rank. The same program derives the sharp conditional equal-dose Hessian error \(|D-L|\le Mr^2\) and third-order residual \(Tr^3/3\). It gives a rational counterexample in which two task-margin maps share the same clean first and second derivatives yet have opposite finite edit effects. The tests check algebra and explicit rational witnesses, not universal Lean kernel formalizations or assumptions on these specific recurrent weights.

The observed 310/316 first-order sign agreement establishes a stable relationship between **a chosen physical intervention** and the receiving model's task gradient. Its complete representation in inactive encoder coordinates prevents claiming that the selected active SAE label is a uniquely identified semantic variable. The earlier SMC-17 low-coverage/negative gates remain part of the evidence.

## Source receipts, audit scope and reproducibility

- Original successful hosted pilot/full run: [Actions #43807953506](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43807953506), head **4ca2d831d0451f813bcb3c8d9212ebfd80cf6824**. The only post-implementation repair before this green run corrected a toy quadratic head's batched/unbatched shape in the test fixture; frozen models, algorithms, interventions and scientific results were unchanged.
- [Full raw intervention + independent audit artifact #14199088649](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43807953506/artifacts/14199088649); ZIP SHA-256 **4c2c19a9e745130173c66613948cfc11dbfb35eaf4ec6dc4c809d8c17127ab62**.
- Raw **25,811,375-byte full.json** SHA-256 **8d11799e73f4353b0130a1aa8edc899711bfd089a8897a2536b392b431b940de**; independent audit JSON SHA-256 **f06902c432d3a8fd15375fd25711c72da01bcdf6565201a9f4ef09abd1ed4257**.
- New-cohort fingerprint SHA-256: novel30 **f93a6f2e91848d7cb841fa2fb8440194a9d7e9fb0b5a4f316742bc086796c9d5**, novel34 **4d93a38eb8da8f38494ca7a6cb9e8f584e9dd2d854bfc9b0a21aab8bd6ea0b08**.
- [Machine-readable permanent eight-condition summary](SMC18_SUMMARY.json) includes registered gate outcomes, original artifact SHA hashes, all eight condition margins and confidence intervals, rank and gradient reconstruction receipts.

The independent standard-library auditor recomputes Sudoku validity and visible-goal selection, all seven original calibration assignment SHA-256s, SMC-17 sham geometry and equal-dose conditions, the 48-dimensional gradient dot products, 92-dimensional minimum-norm inactive duals, directional curvature arithmetic, 16,000 puzzle-cluster bootstrap draws and every registered gate. The auditor **does not** independently rerun the learned neural models or verify exact-real Hessian bounds. The result is reproducible source-locked FP64 numerical evidence on fixed-seed synthetic Sudoku cohorts.

## Next discriminating campaign

SMC-18 answers an important question about the **local task functional**: it usually predicts the finite margin effect of a frozen SAE-selected hidden edit. The full-rank inactive block shows that local gradient decomposition alone cannot assign that effect to a uniquely defined SAE feature.

A productive next experiment would test **representation gauge stability** by independently re-fitting SAEs, applying orthogonal physical-basis rotations that preserve the teacher's task function, and measuring whether matched causal selection and necessity claims survive reparameterization. Capability-matched independently trained teachers would reduce the sizable source/new Sudoku accuracy disparity. Any such experiment requires fresh data and its own frozen decision thresholds.
