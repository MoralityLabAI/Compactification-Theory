# SCM-16 protocol (revision 1, 2026-11-14)

## Ownership and scientific scope

SCM-16 extends the already executed SCM-15 research baseline with a **new causal factorization** of the residual adapter. The original SCM-15 code, tests and historical receipts remain untouched. EvalMath maintains the independent exact source verifier and its public 192-case roster; SCM-16 merely imports it from a pinned external checkout. This is exploratory model research with small synthetic weights trained in the CPU runtime.

## Defined before the pinned EvalMath long-run readout

* Three architecture seeds: `160, 1157, 2154`; one 32-hidden, four-head, four-visit tied core per seed, `readout_dim=8`, SAE width 64, TopK=6, VPD rank 4.
* Same 384 generated training and 96 discovery cases; train counts 120/60/180/60 optimizer steps with batch 24 and learning rate 0.002.
* Standard inference has 12 exact candidate checks per case. Every exact discovery landscape search charges 64 additional checks. A static first-12 and exhaustive 64-query symbolic baseline must be recorded separately. Do not compare the exhaustive baseline's accuracy to a neural top-12 without reporting its larger verification budget.
* Four per-visit VPD sectors: untuned tied control, trained joint, visible-only, fiber-only. Include shuffled gates and zero SAE codes as inference-time ablations on the same trained weights. Training-time FLOPs are not matched to the no-adapter baseline.
* The source readout `W2 W1` is linear and deliberately rank-deficient. Lock the `ker(W1)` projector immediately after base training. The same readout bottleneck is shared by every arm. Verify the base parameters' SHA-256 after all adaptation.
* Select at most 4 exact Hamming-one first-fault maxima from the first 24 discovery tasks. Prefer first-fault maxima to accepted-prefix aliases. Generate 4 replay-training examples per selected source cell and 4 **additional** challenge examples per cell from separate disjoint seed ranges; seal new gold independently.
* Load the preexisting EvalMath 192-case roster **after** freezing all model weights and candidate-selection choices. Retain source-generator and paired-prefix-collision labels. Report overall and per-family exact future success, false-future acceptance, verifier costs, and 3-seed variability. A 2.6 pp average synthetic improvement alone has no claim of statistical significance.
* Matched-norm discovery patches into the final readout kernel should leave final-visit logits unchanged (up to numerical rounding). Earlier patches may change final logits; report this transport separately from skill gains.

## Acceptance gates

The engineering gate requires passed deterministic unit tests, GitHub CI, frozen source hashes, readout projector orthogonality, replay generation without overlap, and no private challenge gold in GitHub. The mathematical gate requires source-checked finite program local maxima with explicit first-fault or prefix-alias labels.

A *reasoning improvement* would require a consistent positive joint-versus-tied effect across held-out independent generator families and seeds; it must also exceed the shuffled-visit control and survive testing with comparable exact verifier budgets. Causal fiber contribution requires a matched intervention or ablation change beyond baseline on held-out math endpoints. A dimensional-topology claim requires separate continuum/representation certificates from SCM-05–14.

## Known limits and pilot-informed amendment

The earliest SCM-16 toy pilot selected accepted-prefix aliases because the first prioritization sorted every maximum by prefix score. After observing that synthetic pilot, the discovery selector was amended to prioritize witnessed first-fault maxima; the toy pilot and the amended run remain separate receipts. This amendment precedes the **new SCM-16 EvalMath CI experiment**, but the EvalMath roster itself was previously public and remains an exploratory readout.

SCM-16 is not an existing Goodfire VPD fit. The low-rank W1 head is deliberately imposed and is not evidence that a pretrained model has the same fiber dimensions. Model/SAE training uses source-program supervision from generated examples. No hidden EvalMath gold labels are copied into evaluation inputs or into newly generated public challenge files.
