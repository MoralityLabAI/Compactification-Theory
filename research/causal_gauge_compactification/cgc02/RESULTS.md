# CGC-02: registered trained protected causal rank fails unseen-depth portability

**2026-11-18 first fresh-root exploratory receipt.** The [registered hosted run #44239969004](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44239969004) verified the frozen [LOCK.json](LOCK.json) against Git blobs before fitting any fresh teacher. It passed all **12 source/geometry tests**, trained the four predeclared independent roots \`2601,2607,2617,2621\`, and archived [full raw four-root JSON](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44239969004/artifacts/14398059500) (extracted JSON SHA-256 \`bc309cef503a30d778f992f2e7e4e764b1336dd19636e344d36c5ceb6817f070\`). PyTorch 2.5.1+cu124 CPU, one thread; all train/eval root IDs are present in the raw receipt. The separate excluded-engineering [native BPD run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44239667587) passed 12 tests, actual nine-invocation full-mask parity (max 4.17e-7) and full-component donor swap parity (max 8.94e-8).

The original clipped-donor edge case and NumPy-only CGC-01 test discovery conflict were fixed **before fresh data**, with their failed runs and pre-data amendments preserved in [ENGINEERING_AMENDMENT_01.md](ENGINEERING_AMENDMENT_01.md) and [ENGINEERING_AMENDMENT_02.md](ENGINEERING_AMENDMENT_02.md). The CGC-01 math tests have their own successful, unchanged 18-test regression.

## Registered gates

| Gate | Result | Registered criterion |
|---|---:|---|
| Engineering (teacher, physical BPD factors, gauge, full protected readout, signed-Jacobian controls) | **PASS 4/4 roots** | 4/4 |
| Exploratory unseen-depth task-subspace portability | **FAIL 1/4 roots** | ≥3/4 |
| Incremental mechanism advantage over privileged full-task Jacobian matched controls | **FAIL 0/4 roots** | ≥3/4 |
| Independent root-calibrated 95/95 effect inference | **Not claimed** | Requires separate prospective ≥59 null roots and ≥32 confirmation |

Both independent physical BPD \`V@U\` refits per root achieved full shared-weight matrix MSE ≤3.46e-8. Complete factor-gauge transformation and physical weight-SVD projector comparisons were stable within numerical tolerance. The **entire three-class normalized current protected readout was invariant to every declared intervention** (observed maximum immediate probability difference 0.0), because the edit occupies an exact affine protected subkernel. Signed two-output task-Jacobian matching residuals were ≤1e-15.

## Same teacher, different future horizon

The teacher was supervised at post-write depths 2/4/6; depth 8 was held out from training *and* task-subspace calibration. The rank-two task block was selected using 256 fresh training-family calibration examples per root, from the eigenspace of the mean task-observability Gramian \`0.75 G_2+G_4+0.5 G_6\`. The output-weight BPD SVD block was selected from the **independently fitted physical matrix** without task gradients. The test source family has anti-correlated input scalars but the same source alphabet and independent exact target oracle.

| Fresh teacher root | Eligible source-donor pairs / 512 | Clean teacher depth-8 task MAE | Task-conditioned rank-2 donor recovery at depth 4 | Same block at unseen depth 8 | BPD physical weight-SVD rank-2 at depth 8 | Privileged two-task signed-Jacobian sham mean at depth 8 |
|---:|---:|---:|---:|---:|---:|---:|
| 2601 | 442 | 0.01934 | 100% | **5.88%** | 75.11% | 6.08% |
| 2607 | 455 | 0.04206 | 100% | **9.23%** | 22.42% | 9.23% |
| 2617 | 432 | 0.03276 | 100% | **25.23%** | 91.67% | 24.90% |
| 2621 | 389 | 0.03374 | 100% | **19.28%** | 5.14% | 19.28% |
| **Root mean** | 1,718 eligible / 2,048 total generated | | **100%** | **14.91%** | **48.58%** | approximately **14.90%** |

All eligibility counts arose from the **preregistered clean depth-4 target-correct, donor-target-different and current-full-softmax-close** filters. Every root exceeded the minimum 200 eligible cases. The task-conditioned block's mean eligible donor MSE at depth 8 was **0.01353**, compared with **0.01088** for physical BPD weight-SVD rank two. The full reconstructed physical BPD write recovered 100% of eligible donor model outputs at *both* depths 4 and 8, as did the full true teacher write and projected exact donor state. The full physical match is a **weight-reconstruction control** and does not establish sparse BPD mechanism identification.

At supervised depth 4 the task-selected rank-two block achieved mean eligible MSE **7.48e-5**, while the rank-two BPD weight-SVD baseline averaged **0.01054**. Thus the superior first-order task subspace around supervised depths can fail drastically outside its selection horizon. At depth 8 the physical SVD baseline performs much better on three roots and much worse on root 2621, illustrating important model-to-model variation rather than a stable rank-two winner.

## Why local rank and exact protected readout do not certify a recurrent mechanism

The task-conditioned Gramian \`G=E[J^T J]\` optimizes a declared **first-order task response** objective at its selected depth/source distribution. It does not enforce invariance of the chosen subspace under the learned physical shared recurrence. Our independent *public-anchored zero-fill one-step closure* diagnostic for the selected task block gives nonzero mean projected-state defects **0.00938, 0.00949, 0.01151, 0.01378**, respectively. These numerical defects are descriptive; no certified global Lipschitz/Hessian bound or autonomous reduced dynamics is established.

The privileged null control selects random protected directions to match the **entire signed 2D task Jacobian response and physical L2 dose** of the frozen task subspace. It closely reproduces the selected edit's depth-8 donor recovery across all four roots, with differences of approximately -0.20, 0.00, +0.33 and 0.00 **percentage points**. The registered stronger incremental gate therefore fails **0/4**, independently of the weight-SVD comparison. Reconstructing the same future task derivative explains essentially all measured rank-two task-subspace success in this gym.

The first protected softmax is preserved **exactly** at the injection step because the selected physical subkernel zeroes its two variable logits. Future normalized public readouts can change under recurrence, and the experiment reports that drift separately. The limited three-class scalar toy does not supply transcript privacy, multi-token normalized-logit closure or proof of genuine learned secret-keeping.

## Mathematical implication for Compactification-Theory

**CGC-01's local Eckart–Young rank theorem remains valid.** CGC-02 identifies its key additional obligation for compact recurrent reasoning: the task-relevant block must also provide an approximately **closed dynamical quotient** over the intended horizon and source families. One natural next quantity is the **temporal principal-angle drift** between the development-selected protected task projector and a later-horizon protected task response eigenspace. The triangle-inequality bound
\[
\|J_k(I-\Pi_{\rm dev})\|_{\rm op}
\le \sigma_{r+1}(J_k)+
\|J_k\|_{\rm op}\|\Pi_{k,r}-\Pi_{\rm dev}\|_{\rm op}
\]
shows that even a small local singular tail can coexist with a large mismatch for a **frozen** earlier projector if task directions rotate with recurrent depth. This is a deduction from linear algebra, not a measured or certified angle in CGC-02.

A prospective follow-up should freeze a time-transport/closure penalty for physical BPD/SAE blocks, measure protected task spectra across independently trained model roots, and score unseen horizons with the same privileged signed-Jacobian controls. It should compare task-ranked blocks, physical weight-SVD, same-dose controls, independently refitted factor gauges and a separately trained *autonomous* reduced transition with an explicit description/state/precision budget. The 59-root independent tolerance and source-family-disjoint 32-root confirmation are future protocols, not available conclusions.

**Scope:** four fresh but same-source-family synthetic recurrent model roots, a learned physical BPD matrix with all factors active, a simple three-class softmax observer, and no newly trained real TRM/Nanbeige. Existing NOR/SMC/LSV, BPD SF03 0/32 and SF04 3/8 negative receipts and the sealed 168-case ARG-03-N dataset are unchanged.
