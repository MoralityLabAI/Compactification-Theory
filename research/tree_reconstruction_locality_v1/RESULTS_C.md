# KSR-01C: frozen recurrent-state transfer audit

Date: 2026-11-14. Evidence labels: FROZEN_RRC25_COARSE_STATE_ASSUMPTION_AUDIT and EXACT_FINITE_ONE_STEP_REFINEMENT.

[Registered protocol](PROTOCOL_RRC25_C.md) | [New proofs](THEORY_C.md) | [Native runner](audit_rrc25.py) | [Refinement certificate](refinement.py)

Verified run: https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43551533462

## Frozen model and intervention

The original RRC-25 1,000-parameter GRU checkpoint was loaded through its existing training-source/shape gate. Its file SHA-256 is 54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b. Its nine-row native input was replayed independently with maximum final-logit error 0.0. Mutating the withheld future and hidden source-program label leaves the replay unchanged in the isolation test.

A parent label is the first-template U index of the highest-ranked sixteen-way native output. It has four possible values. Each continuation edge independently selects one of eight calibration feature rows and applies the original recurrent step to the full twelve-dimensional hidden state. The input-bank SHA-256 is 62edc50b5a4671be62458bda5d93b5cf91ac23a7f11cdf62f49ed82938aecbe8.

There are 36 calibration roots from three source families and 144 test roots from six families. The test roots never construct the input bank. Root IDs and public visible inputs are disjoint between these cohorts. The extra recurrent rows form an artificial continuation beyond the native nine-row task; they carry no claim of following the model's training distribution.

The discrete probability calculations are exact conditional on portable floating-point inference, this finite root cohort and the fixed bank. They are not real-arithmetic neural-network proofs. The original LoopedTransformers Q5 weights were not used.

## Transfer assumptions fail on the declared native-label process

| Diagnostic | Calibration roots | Test roots |
|---|---:|---:|
| Cohort size | 36 | 144 |
| Root label counts | 14, 11, 7, 4 | 53, 30, 47, 14 |
| Mean conditional sibling same-label excess Delta | 9077/177408 = 0.051165 | 7029445/120524544 = 0.058324 |
| Maximum row TV from best symmetric channel | 965/3696 | 1312547/4184880 = 0.313640 |
| Maximum TV between actual two-step channel and one-step matrix squared | 21/64 = 0.328125 | 26909587/167395200 = 0.160755 |

The child inputs are independent given the full hidden parent. Their label distributions vary among hidden parents sharing the same compressed label. KSR-7 identifies the resulting sibling covariance as Cov(p_H | parent label). A positive trace Delta is an exact finite-law violation of conditional sibling independence at the coarse label.

The labels also have a nonuniform prior and a transition matrix that is not symmetric. Squaring their averaged one-step matrix fails to reproduce two-step transitions. Accordingly, the original uniform symmetric broadcast floor is not applied. The code retains same-label parent pairs with different exact child laws as replayable witnesses.

The calibration least-squares symmetric coefficient is 199/231, while the test-cohort descriptive fit is 1519817/2092440. Substituting these coefficients into b*lambda^2 would give numbers above one; those numbers provide no reconstruction theorem for a process that fails the needed assumptions.

## KSR-8: recovery guarantees that retain the measured dependencies

For arbitrary root prior pi and count scores N_j, the new theorem gives

    P_MAP >= max(max_j pi_j,
                 sum_j pi_j^2 + sum_j Cov(1{root=j},N_j)^2 / Var(N_j)).

A zero-variance score contributes zero. This follows by projecting each posterior root probability onto its count score and applying Cauchy-Schwarz, then using max posterior >= sum of squared posterior probabilities. No tree symmetry or coarse sibling independence is needed.

The table refers to the test-root finite population, uniform over its 144 full hidden roots. Its largest prior label mass is 53/144 = 36.8056%.

| Depth | Observed leaf labels | Mutable count bits, upper bound | Dependence-robust accuracy floor | Exact finite-population count MAP |
|---:|---:|---:|---:|---:|
| 1 | 2 | 6 | 75.7673% | 87.5651% |
| 2 | 4 | 9 | 66.9088% | 86.7705% |

The count posterior is computed from the same finite root population being described. These are exact population diagnostics under that declared intervention law, not held-out accuracy of a separately trained decoder. The static posterior table, the input-bank enumeration, and the native model's state and weights are additional resources. Only the three mutable leaf counters are included in the stated bit bounds.

## KSR-10: minimum one-step state refinement

A refined parent label that retains the original four-way label and makes original child labels conditionally independent must distinguish every different child-law vector within each original class. KSR-10 proves that the exact minimum supported alphabet size is the sum of those distinct-kernel counts. The constructor D=(original label, child-law vector) attains it.

| Cohort | Distinct child laws per original label | Minimum refined labels | Minimum fixed-length code |
|---|---|---:|---:|
| Calibration | 4, 2, 1, 2 | 9 | 4 bits |
| Test | 6, 7, 2, 6 | 21 | 5 bits |

The original four-label representation uses 2 bits. On the test support, preserving the original label while restoring this one-step sibling independence requires at least 21 supported labels, hence 5 bits. The certified grouping has zero within-group child-kernel variance.

This result concerns original child labels for one fixed kernel and finite parent support. Recursive closure of the refined child process has not been established. Computing the refinement consumes child-law information; its static lookup table and computation cost remain separately charged.

## KSR-9: why checking a single-edge channel is insufficient

An independently verified finite counterexample preserves the symmetric channel q=3, b=4, lambda=1/2 along each edge, while copying one shared child-channel draw to all four children. At depth one the optimal recovery is 2/3. Applying the independent-child variance would incorrectly claim a 5/7 lower bound. The corrected shared-innovation recurrence gives V_1=4 and the valid lower bound 1/2.

In general, if siblings share a channel draw with probability rho at each vertex, independently between vertices, the moment recurrence is

    V_(L+1) = V_L/(b*lambda^2)
              + (b-1)/b * [1 + rho*(1-lambda^2)/lambda^2].

Exact histogram enumeration checks this formula for multiple alphabets, depths and rho values. This correlated model is a constructed control, separate from the native GRU experiment.

## Verification

KSR-01C adds 10 exact dependence tests, 4 refinement tests and 7 frozen-checkpoint integration tests. All 21 passed in the cited workflow. The original KSR-01/01B suites remain in their separate workflow and also passed at commit 6aa3aec75e7308d355f2375b778945dfd8de9133, for 44 tests across the two workflows.

CI retains the full audit JSON, minimal-refinement groups, input-bank data and source hashes. No source checkpoint was updated, no new recurrent model was trained, and no original KSR theorem was weakened to fit the neural observations.
