# SHAFT-COMP-14: Task-sensitive directional observability

**Registered 2026-11-11 before the new holdout was evaluated.** [Protocol](SHAFT_COMP_14_PROTOCOL.md) · [Executable CPU experiment](experiments/task_weighted_observability.py) · [Frozen numerical summary](results/comp14_summary.json)

**Outcome:** The preregistered task-weighted directional Gramian achieved held-out ROC-AUC **0.652604**, versus **0.481179** for the matched SHAFT-COMP-13 rank-deficit baseline. The paired within-model bootstrap interval for the AUC difference is **[0.059345, 0.272535]**. The study uses only two fresh GRU model seeds (505 and 606), and the effect varies considerably between them. The baseline is a learned-probe rank; the directional metric has access to derivatives of the network's native four-query readout, so it uses richer measurement information.

The unweighted native directional score achieved **0.654742**, slightly above the preregistered boundary-weighted score. The result supports further investigation of **state-contrast direction and task-output sensitivity**. It provides no evidence that the chosen margin weight adds predictive information.

## 1. Mathematical construction

Let two naturally reached hidden states be \(h_A,h_B\in\mathbb R^{24}\), with \(\Delta=h_A-h_B\). For each state, use a shared, independently sampled diagnostic five-token continuation \(x_0,\ldots,x_4\). Along the resulting recurrent trajectory,

\[
\Phi_{s,0}=I,\qquad \Phi_{s,t+1}=J_{s,t}\Phi_{s,t},\qquad
J_{s,t}=D_hF_{x_t}(h_{s,t}).
\]

Define the four native counterfactual response logits \(z_{s,t}(h)\) as the output of the frozen GRU's query-step cell and its actual nonlinear readout for each of the four synthetic query tokens. Let

\[
C_{s,t}=D_hz_{s,t}(h_{s,t})\in\mathbb R^{4\times24},
\quad
D_{s,t}=\operatorname{diag}
\left(\frac{1}{1+|z_{s,t,q}|}\right)_{q=1}^{4}.
\]

The registered primary feature is

\[
\boxed{
S_T^2=\Delta^\top
\underbrace{\left[
\frac12\sum_{s\in\{A,B\}}\sum_{t=0}^{5}
\Phi_{s,t}^{\top}C_{s,t}^{\top}D_{s,t}^{2}
C_{s,t}\Phi_{s,t}\right]}_{Q^{\rm task}_5}
\Delta.
}
\]

The matrix is positive semidefinite. Its quadratic form measures how strongly the **actual paired state difference** would affect native task logits under a local linearization. The independent outcome uses a different 32-token continuation and never enters these features. The directional score scales with perturbation magnitude; the study tests its predictive value on naturally occurring state differences rather than actively increasing them.

The unweighted control fixes \(D_{s,t}=I\). The COMP-13 matched baseline instead computes the singular-value rank of a separate learned four-output linear probe W along the same diagnostic schedules, using the *already registered* relative threshold \(10^{-3}\):

\[
S_R=24-\frac{\operatorname{rank}_{10^{-3}}(\mathcal O^W_{A,5})
+\operatorname{rank}_{10^{-3}}(\mathcal O^W_{B,5})}{2}.
\]

A rank counts linearly detectable state dimensions without specifying the direction \(\Delta\). The new directional quadratic form includes that direction. **The change in native versus learned probe readout is a competing explanation for observed gain** and remains to be separated in a further factorial ablation.

An [exact finite integer example](task_gramian_reference.py) uses a rank-one monitor of a two-dimensional state. Two equal-norm state directions have directional energies 1 and 0 even though the measurement rank is unchanged. [TaskObservability.lean](lean/Shaft/TaskObservability.lean) kernel-checks this finite example and monotonicity/zero-cost properties of nonnegative cumulative task energies. It does **not** claim a formal Lean proof of the real-valued neural Gramian's full spectral properties.

## 2. Registered data and causal separation

The same synthetic four-bit truth-table TaskMemoryGRU architecture as COMP-13 was used. Four independently trained 3,265-parameter GRUs with 24 hidden units were rebuilt from their known seeds. The learned four-output probe is trained on the native model's own counterfactual behavior, with goal IDs excluded from probe supervision. The base GRU itself was trained using explicitly supplied goal IDs.

**Calibration seeds 101/202; fresh held-out seeds 505/606.** The completed COMP-13 test seeds 303/404 were not reused.

Each seed is evaluated at recurrent depths 12 and 40 on 16 goals × 16 pairs, totaling 256 pairs per depth and seed. The states in a pair share their task but are reached through independent random input prefixes. Eligibility requires equality of current four-query *native binary* responses and equality of the four learned-probe binary outputs.

The diagnostic inputs are independently seeded and shared by the states within each pair. The outcome inputs are separately seeded, shared by both states, and 32 tokens long. \(Y=1\) means the two native four-query signatures differ **after** that continuation.

The analytic derivative of the complete GRUCell→tanh→linear native readout was independently compared with PyTorch autograd. The maximum absolute discrepancy across model seeds was **1.430511474609375×10⁻⁶**, below the registered \(10^{-5}\) tolerance.

The only primary test is the ROC-AUC of \(S_T\) on eligible held-out pairs, with the **higher-score-more-divergence** direction frozen in the protocol. The boundary weights, five-step diagnostic horizon, outcome length, rank threshold and holdout seeds were fixed in advance. Calibration outcomes are preserved without feature tuning.

## 3. Measured performance

| Score | Calibration AUC (101/202) | Held-out AUC (505/606) |
|---|---:|---:|
| **Task-weighted directional Gramian, registered** | **0.670026** | **0.652604** |
| Native directional, no boundary weight | 0.696794 | 0.654742 |
| Matched learned-probe rank deficit | 0.447497 | 0.481179 |
| Native current-boundary margin | 0.578085 | 0.613612 |
| Learned-probe current margin | 0.527185 | 0.572826 |
| Euclidean state separation | 0.552525 | 0.501990 |

Eligibility and prevalence:

| Split | Generated pairs | Eligible pairs | Divergence events |
|---|---:|---:|---:|
| Calibration | 1,024 | 819 | 60 |
| Held-out | 1,024 | 864 | 50 |

The registered task-weighted AUC improves on the matched rank baseline by **0.171425**. The paired, seed/depth-stratified 1,000-bootstrap interval for this difference is **[0.059345,0.272535]**. The 95% interval for the primary AUC itself is **[0.573592,0.731161]**. The 1,000-permutation, within-seed/depth two-sided test gives **p=0.014985**, centered on its stratified permutation-null average **0.567573**. That null average differs from 0.5 because different seed/depth strata can have different score distributions and event prevalences; the test evaluates association conditional on those strata.

These uncertainty calculations resample or permute **pairs within the two fixed test models**. They do not quantify variation between newly trained model seeds.

### Held-out model variability

| Model and depth | Eligible | Divergence events | Task-weighted AUC | Rank-deficit AUC |
|---|---:|---:|---:|---:|
| Seed 505, depth 12 | 256 | 7 | 0.686173 | 0.422834 |
| Seed 505, depth 40 | 171 | 13 | 0.688413 | 0.410906 |
| Seed 606, depth 12 | 256 | 14 | 0.473436 | 0.547078 |
| Seed 606, depth 40 | 181 | 16 | 0.619318 | 0.555114 |

Pooling within seed gives task-weighted AUC **0.742** for seed 505 and **0.575** for seed 606. The depth-12 result on seed 606 reverses the expected ranking. The total of 50 positive events makes model-level replication an important next priority.

### Exploratory operating point

The following operating-point comparison was computed **after** inspecting the held-out AUC and is not a registered confirmatory result. For each score, set the threshold to the 90th percentile of its *calibration negative* scores, and predict divergence only for scores **strictly greater** than that threshold.

| Score | Held-out false-positive rate | Held-out true-positive rate |
|---|---:|---:|
| Task-weighted directional | 9.83% | 22% (11/50) |
| Unweighted native directional | 10.69% | 26% (13/50) |
| Rank deficit | 10.93% | 8% (4/50) |

These operating points have similar, not identical, held-out FPRs. The low counts prevent strong claims about deployed monitors.

## 4. Frozen reproducibility and interpretation

The [full local result JSON](results/comp14_summary.json) was generated twice from the executable CPU script, with byte-identical output under PyTorch 2.10.0+cpu and NumPy 2.3.5. The full 2,048-pair receipt has SHA-256

\`2930a006382adb552accd501fe6bf730d6681d0937c998ac061054392f126844\`.

The repository's compact frozen summary preserves exact Mann–Whitney positive-negative pair counts. At held-out, the task-weighted feature wins **26,561 of 40,700** positive-negative comparisons (zero exact ties), giving AUC 0.652604. The rank deficit wins 15,041 and ties 9,086 comparisons, giving AUC 0.481179. [task_auc_audit.py](task_auc_audit.py) independently reconstructs all six empirical AUC values on both splits using the standard library only.

To reproduce locally from repo root:

\`\`\`bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.task_gramian_reference
python -m research.shaft_compactification.task_auc_audit
python -m research.shaft_compactification.experiments.task_weighted_observability \
  --per-goal 16 --permutations 1000 --bootstraps 1000 \
  --out /tmp/comp14_full.json
\`\`\`

The trained experiment is deliberately opt-in; lightweight GitHub CI checks Python syntax, frozen-statistic agreement, exact finite mathematics and the standalone Lean build. Complete per-pair data and an independent second full-run receipt are available in the conversation's research archive. The compact frozen summary and source are committed in this repo.

## 5. Consequences for the Q5 bridge

COMP-13 established that a **generic effective observability rank** was not useful on two unseen GRU seeds (ROC-AUC 0.504). COMP-14 supplies a better-motivated directional score in this controlled task family, and records a positive held-out test accompanied by substantial seed variation. The native-unweighted comparison prevents attributing the result to the proposed decision-boundary weight. For a clean next experiment, freeze a **2×2 factorial comparison**: learned versus native readout, each with rank versus directional sensitivity, using new training seeds and task-family holdouts.

The original LOOPSCOPE Q5 metric \(F_1/F_8\) is a *parameter-gradient fragility ratio*. It remains defined separately. Its 12-arm weights are still absent from the accessible Git branch; COMP-14 makes no Q5 checkpoint performance claim.

This experiment studies local detectability of differences in synthetic recurrent states. Nothing here establishes unknown objectives in frontier agents, resurrection after model deletion, or autonomous distributed persistence.
