# SIML-01N: measured recurrent learning, 2026-11-02

**Exploratory five-seed CPU result.** This new-seed cohort followed earlier observations with seeds 0–2 and 10–12, so the experiment is a controlled comparison of training initializations, with a post-hoc developed curriculum policy. Training uses PyTorch 2.10.0+cpu, width 48, seed list `20,21,22,23,24`, 320 AdamW updates per arm and 32 sampled examples per update, with two tied GRU transitions per example. Diagnostics are collected every 16 updates on development tasks, at a matched inspection schedule. Full raw outcomes and allocation traces: [`results/neural_curriculum_5seed_v02.json.gz`](results/neural_curriculum_5seed_v02.json.gz).

| Arm | Weighted T0/T1/T4 accuracy, mean of 5 seeds | T1 unseen two-word accuracy | T4 held-out pairs | Secondary unseen three-step T2 |
|---|---:|---:|---:|---:|
| Fixed | 0.5048 | 0.5625 | 0.0056 | 0.8750 |
| Random | **0.5446** | 0.6500 | 0.0069 | 0.9167 |
| Behavioral greedy | 0.4425 | 0.5000 | 0.0583 | 0.5000 |
| Introspective greedy | 0.3519 | 0.4875 | 0.0875 | 0.4917 |
| Guarded introspective | 0.5325 | 0.6250 | 0.0042 | 0.9417 |
| Guarded shuffled diagnostic | 0.5381 | 0.6375 | 0.0042 | **0.9500** |

The full dataset is small and task families reuse the same symbolic grammar. T0 is an eight-state sanity check; its worlds overlap train/dev/test. T1 holds out two compositions of known action primitives but shares the eight possible microstates. T4's 144 test state-pairs are disjoint from 678 training and 139 development pairs. The 24 T2 three-action cases are unseen sequence lengths, using known primitives. Their non-independence and small population limit how broadly this outcome can generalize.

**Observed behavior.** Both greedy strategies concentrate training on T4's difficult 31-way arithmetic, reducing aggregate performance. The guarded policy enforces positive family coverage and achieves markedly higher aggregate accuracy than either greedy arm. Random selection remains strongest on the primary weighted score. The shuffled diagnostic arm slightly exceeds the measured introspective arm; there is no observed incremental advantage from the closure residual in this study. The guarded policy's high T2 transfer accuracy is interesting but the shuffled control is higher still.

**Resource accounting.** Each arm receives 320 updates, 10,240 sampled training examples and 20,480 recurrent step-example operations per seed, with the same initial checkpoint, width, GRU and diagnostic rounds. Task-specific classification/probe operations and exact numbers of privileged labels differ with each curriculum's task allocation. Elapsed time is reported per run. A FLOP profiler and fully matched auxiliary-target acquisition cost will be necessary for strict equal-total-compute claims. No proposer LLM inference or paid compute was used.

**Prior diagnostic follow-up.** In a separate exploratory 1,600-update run focusing mostly on T4 (seed 14), the difficult shear task reached 0.382 held-out accuracy (55/144 correct), illustrating that the short budget does not support a claim that the arithmetic is unlearnable. That diagnostic follows the earlier pilot and is outside the frozen five-seed comparison.

**Evidence checks.** The standard-library `verify_neural_receipt.py` validates source hashes, all 30 run identities, task allocation, aggregate arithmetic and recurrent-work counts. The optional `--retrain` mode regenerated all 30 trained checkpoint fingerprints and final score vectors on the same CPU environment. The full unit suite passes in normal and optimized Python. Exact source hash and fixture records remain separate from the earlier tabular reference receipt.

## Next scientific steps

A compelling introspection claim requires matched computational acquisition of task features and larger independent semantic task generators. An extension can test the prediction that successor-consistency supervision improves neural composition transfer when only the candidate representation's dynamics are deficient, using independently frozen model families and no privileged post-hoc tuning. A new semantic-family exam should be generated after the next design freeze, with T4 held-out input combinations fully unavailable to curriculum development. The local 27B proposer can submit typed experiment proposals to an external broker, with proposer inference included in total cost.
