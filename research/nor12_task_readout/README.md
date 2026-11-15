# NOR-12: task-aware residual readout conditioned on recurrent basins

**Protocol, November 15, 2026.** A frozen RRC-25 1,000-parameter GRU has two independently certified, locally attracting zero-input fixed points. NOR-11 trained a 13-parameter head that predicted held-out basin membership with AUC 0.983 on 61 unique source-visible tensors, yet an attractor-steering intervention decreased four-check future-correct template predictions from 42/72 to 40/72.

NOR-12 keeps the original hidden vector unchanged. A small **task-supervised residual readout** is fitted on top of the frozen original 16-way candidate scorer, with the previously trained basin-probability output available as one additional covariate.

## Model and supervision

Frozen original checkpoint: SHA-256 `54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b`. The thirteen parameters of the basin head come from the committed NOR-11 auxiliary-head JSON and are also frozen.

Each eligible source case has one public header and nine rows of eight observed features. Its twelve-dimensional hidden state \(h\) yields original candidate scores \(b_j(h)\) for the sixteen possible suffix-operation pairs. We fit a residual matrix \(W\in\mathbb R^{16\times14}\), initially zero:

\[
s_j(h)=b_j(h)+\lambda\left\langle W_j,
\left[
\frac{h_1-\mu_1}{\sigma_1},\dots,\frac{h_{12}-\mu_{12}}{\sigma_{12}},
2\,p_{\rm basin}(h)-1,\;1
\right]\right\rangle.
\]

The mean and scale \(\mu,\sigma\) come exclusively from the earlier basin-head training cohort. The original source model and candidate readout remain frozen.

On training examples, the algorithm performs all sixteen **exact public two-demo checks through degree 192**. Let \(A\subseteq\{1,\dots,16\}\) be all accepted public candidates. It minimizes

\[
\mathcal L=-\log\left(
\sum_{j\in A}
\frac{e^{s_j}}{\sum_k e^{s_k}}
\right)+\frac{\eta}{2}\|W\|_F^2.
\]

This preserves ambiguity when multiple candidate templates agree with the observations. The gradient uses the unconditional softmax probability minus the softmax conditional on accepted public candidates. The model never uses the synthetic generating program, its future query coefficients through degree 256, or the withheld future-correct score during head fitting.

The exhaustive 16-branch public oracle is **an additional training-side cost**. Inference ranking must still respect the original four-paid-template-check budget and does not run the exhaustive public scan.

## Predeclared arms and control design

All task correction heads have 224 allocated parameter slots. The principal head uses the trained NOR-11 basin probability. A matched **shuffled-basin** head uses a probe trained on permuted basin labels over the same original NOR-11 training cohort. A **random-feature** head receives a deterministic scalar constructed from the public-feature SHA-256, with otherwise identical width and optimizer. The **hidden-only** head sets the additional feature to zero; 208 of its 224 allocated coefficients are active. The original frozen ranker is the zero-residual control.

Training: 96 source cases, 32 per signed, sparse and product family. Development: 36 cases from those families. Held-out test: 72 cases, 12 per each of six source families. This is the identical roster and source-locked manifest used by NOR-11. Train/development public feature tensors have no overlaps, and test tensors are disjoint from both. Twelve distinct late-source synthetic cases alias to one input tensor, giving 61 unique held-out model inputs. Raw case-level task outcomes and per-family results remain in the receipt.

All heads train for 200 full-batch Adam updates with learning rate 0.025 and L2 regularization 0.065. For every trained head, the development set chooses a residual amplitude from \(\lambda\in\{0.25,0.5,1,2\}\) under the original exact future scorer. The global development policy also compares those corrections against the original \(\lambda=0\) ranker and freezes the best option before testing. The development utility is future-correct count minus half the false-accept count minus 0.015 times the number of paid checks; ties prefer the original ranker and smaller correction amplitude.

The held-out evaluation reports original baseline, each residual-head type at its **own development-selected amplitude**, and the global development-selected policy. Every arm is run on exactly the same case sequence, with a maximum of four paid public-template checks and no model-weight changes. The source-native withheld future scorer is used only after candidate execution.

## Interpreting the result

Improvement requires higher **held-out future-correct recoveries** at comparable false-accept rates and exact paid check budgets. Better prediction of the zero-input attractor by the frozen auxiliary head is already established in NOR-11 and does not count as a proof-ranking improvement.

If the basin-conditioned residual improves task recovery more than the matched shuffled-basin and public-random-feature residuals, that supports the utility of a basin covariate for this particular synthetic task. Any gain also must be considered relative to the **hidden-only residual** of smaller effective dimension and the more powerful, separately charged sixteen-preview exact ranker from RRC-25.

If all task corrections degrade held-out recovery, this is a negative result about the transfer from asymptotic zero-input geometry to finite-prefix theorem-template reasoning. It does not refute the mathematical two-attractor and basin-capture certificates.

## Reproduce

Run from repository root with Python 3.12 standard library:

    python -m unittest research.nor12_task_readout.test_readout -v
    python research/nor12_task_readout/readout.py --check --train 32 --dev 12 --test 12 --out /tmp/nor12_task_readout.json

The [NOR-12 GitHub Actions workflow](../../.github/workflows/nor12-basin-task-readout.yml) runs source-lock tests and archives the full readout matrices, development-selection grid, paired held-out task comparisons, case-level exact verification traces and source manifest hashes. Train-time full public oracle calls are recorded separately from inference-time paid checks.

## First executed held-out experiment

[Executed CI #43708298739](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43708298739) passed the public-only training and native four-check evaluation. The full 224,473-byte JSON receipt is archived in the Actions artifact under SHA-256 a1995f628038ab6bd447c7d4438ee861ca3c10ef90ca5a87c4063bf99811db95. The 96/36/72 model-input manifests reproduce the original NOR-11 cohorts.

| Readout, dev-selected scale | Future correct /72 | False accepts | Inference paid checks | Paired new successes / losses |
| --- | ---: | ---: | ---: | ---: |
| Original frozen GRU | 42 | 6 | 196 | 0 / 0 |
| Basin coordinate, \(\lambda=.25\) | **44** | 5 | 191 | 2 / 0 |
| Public-hash random coordinate, \(\lambda=.5\) | 43 | 6 | 188 | 2 / 1 |
| Shuffled-basin coordinate, \(\lambda=1\) | 42 | 5 | 187 | 4 / 4 |
| Hidden-only correction, \(\lambda=1\) | 42 | 5 | 187 | 4 / 4 |
| Overall dev-selected: random coordinate, \(\lambda=.5\) | 43 | 6 | 188 | 2 / 1 |

The global development selector chose the random-feature control. This matters when interpreting the better basin-arm held-out score, because all readout types were inspected after the run.

**Both basin-arm gains came from two late-source test cases with the same exact public-feature SHA-256 prefix dc3ab9200911.** The other five source families had the same future-correct case counts as the original GRU. Distinct synthetic public-case identities can be associated with the *same* 9×8 input tensor and 12-dimensional hidden vector while possessing different unseen program continuations. Those two gains are therefore strongly correlated at the model-input level.

The result provides a numerical task-ranking improvement for the chosen first split, with inconclusive independent generalization. The native exact public verifier and withheld degree-256 grading remain unchanged.

## NOR-12B: independent-seed confirmation and post-hoc basin-feature ablation

The precommitted [NOR-12B frozen design](results/nor12_initial_receipt.json) stores the exact original readout weight hashes, four dev-selected amplitudes and SHA-256 of the first experiment result. The [confirmation runner](independent_confirmation.py) reconstructs those weights solely from the same original public training cases, asserts every weight hash matches, and then samples four **new**, seed-separated 72-case held-out cohorts. Other than the intrinsically aliased late-source family, model-visible feature tensors cannot repeat across training, development, the first test or the four confirmation tests.

Each new cohort reports the original ranker and all four fixed readout heads. Training choices and dev-selected amplitudes remain unchanged. The result provides paired native future-success and paid-check counts, a per-feature-cluster aggregation and a separate comparison on nonlate inputs. The confirmation analysis is explicitly **exploratory** because the design followed inspection of the first held-out outcomes.

A separate **retrospective ablation** tests the already fitted basin head on the first 72 cases with its basin coordinate set to zero, replaced by the shuffled-label probe coordinate, or permuted as a block across distinct feature hashes. These tests hold all fitted readout weights and \(\lambda=.25\) fixed. Their effect estimates are diagnostic rather than prospective.

The [NOR-12B CI workflow](../../.github/workflows/nor12b-confirmation.yml) checks frozen training hashes and runs the four new cohorts, preserving their data and the retrospective ablation in a second archived artifact.
