# SIML-01N: recurrent neural student and task-graph curriculum pilot

**Exploratory version 0.2, 2026-11-02.** This implements a genuinely trainable, tied-weight PyTorch GRU and a read-only self-diagnostic based on the discrepancy between predicted and canonical next hidden states. The prior table learner remains a separate exact finite reference. FAW and FRC are intentionally deferred.

## Empirical question

Can an introspection-guided task scheduler produce more held-out capability per recurrent update than random task allocation, a fixed graph curriculum or a behavioral learning-progress scheduler? Does internal closure information add anything beyond the same scheduler using shuffled closure measurements?

A naive greedy policy is included as a failure control. The guarded policy mixes a fixed family distribution with its diagnostic priority. Task-family coverage stays positive; a difficult and initially unlearnable task cannot capture every update.

## Fixed model and learning interface

Model: width-48 initial-state encoder, tied GRUCell reused exactly twice per training example, learned action embeddings and shared 31-way classifier, plus a 3-bit auxiliary state probe. Total parameter count is computed from the actual model. PyTorch 2.10.0+cpu, float32, AdamW 0.003 with zero decay, no scheduler, no teacher model, no GPUs. Training batch size 32.

T0 and T1 receive a three-bit microscopic initial state, using the exact operations and task definitions from SIML's finite reference. T4 receives **engineered** one-hot codes of `(s,a)` with `s=a+b mod 31` and phase coordinates `cos/sin(2πs/31),cos/sin(2πa/31)`. This inductive bias makes the joint shear-sufficient information accessible at input and is charged as known preprocessing. The model still learns its own recurrent transition and output classifier; there is no exact modular-addition decoder in its execution. Results do not test discovery of those engineered coordinates.

Every example runs the same two recurrent calls. A T0 sample uses `nop,nop`, T1 uses a genuine two-action word, and T4 uses `shear,nop`. The task-specific cross-entropy uses binary output classes on bit tasks and all 31 classes on shear. An auxiliary BCE state probe uses privileged supervised one-step and two-step microstates on T1 training examples, with coefficient 0.35. Every arm has the same architecture, coefficient, optimizer and initial checkpoint per seed; different allocation schedules encounter different amounts of privileged T1 supervision and the task-conditioned auxiliary work. **Matched budgets mean equal optimizer updates, batch sizes, GRU executions and diagnostic rounds, not exact FLOPs or identical per-task data exposure.** Record task allocations, measured elapsed time and raw results.

## Task splits and exposure

| Task | Train | Development | Final evaluation |
|---|---|---|---|
| T0 | All 8 states | Same 8 states | Same 8 states; sanity check, no unseen-state claim |
| T1 | 18 two-action word types × 8 states | 5 disjoint two-action words × 8 | `flip_b -> reveal_b` and `flip_y -> reveal_b`, 16 cases |
| T4 | 678 `(a,b)` states | 139 distinct pairs | 144 separate pairs |
| T2, secondary | No length-3 training words | None | Three untouched length-3 action sequences × 8 states, measured after freeze |

T4 train/dev/test buckets use a fixed SHA-256 mapping of `(a,b)`, independent of labels, run seed or model. T1 final words were already seen in prior mathematical/code-development pilots: their word-level separation is genuine within each training run, but the broad study is exploratory. The T2 length-three sequence family was added after preliminary pilot inspection and frozen for the next seed cohort, as a secondary novel-horizon check. Its primitive actions remain the same known operations. This is a modest generalization test, with no claim about arbitrary new task rules or unseen hidden states.

## Arms and information access

- `fixed`: 5:9:6 repeating T0/T1/T4 allocation. Primary fixed curriculum baseline.
- `random`: task-family weighted random allocation with a fixed policy seed.
- `learning_progress`: greedy behavioral-dev accuracy gap with observed recent progress; contains no hidden-state diagnostic.
- `introspective_naive`: greedy weighted dev gap plus measured latent closure residual; explicitly tests starvation.
- `introspective`: uses the same latent closure residual, but mixes its derived preference 35% with the frozen family distribution 65%.
- `introspective_shuffled`: runs the same guarded selector using a fixed permutation of the residual values across task families. Its call pattern and diagnostic costs match the introspective arm.

All arms perform the same development diagnostic rounds, computing all-family development accuracy and mean one-step hidden-state closure error. Only the eligible policy reads those metrics to choose tasks. The test split is constructed and scored after training has ended and the model checkpoint has been hashed. Every test read is therefore a report, rather than a training input. All scientific claims remain conditional on the declared task construction and holdout conventions.

## Measured closure residual

For an observed development initial state `x` and permitted first action `a`, let `enc` be the learned encoder and `GRU` the learned single step. Measure

\[
\rho_a(x)=\big\|\mathrm{GRU}(\mathrm{enc}(x),a)-\mathrm{enc}(F_a(x))\big\|_2^2/d.
\]

The diagnostic uses privileged **development** successor states and is sensitive to latent coordinate scale. A low residual does not prove future functional sufficiency or intervention closure. The shuffled control tests the practical contribution of the measured task assignment. An independently trained probe and near-fiber intervention experiments remain needed for a mechanistic interpretation.

## Frozen new-seed cohort, following pilot development

Use seeds `20,21,22,23,24`, 320 updates per arm, 32 examples/update, width 48, diagnostics every 16 steps, all six arms. This is a **new-initialization exploratory comparison**, because earlier pilot results with other seeds informed the architecture, engineered features and guarded policy. The frozen primary measure is the family-weighted accuracy on T0/T1/T4, with weights `0.25,0.45,0.30`. The secondary measure is exact length-three T2 accuracy. Retain arm-wise seed values and signed pairwise differences; avoid inferential claims from a five-seed mean. The critical diagnostic ablation is guarded introspection minus shuffled closure. Compare all arms under their stated physical costs.

The 27B local proposer adapter in `local_proposer.py` is a strict JSON schema and a loopback-only HTTP client for an opt-in OpenAI-compatible server. It returns a validated proposed operator and predicted update cost. It neither launches a 27B model nor authorizes a train/promotion action. A connected 27B experiment needs a known model checkpoint, proper log isolation, matched proposer-token accounting and fresh graph families before a capability claim.

## Reproduce

```sh
python -m unittest discover -s research/self_introspective_meta_learning_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/self_introspective_meta_learning_v1 -p 'test_*.py' -v
python research/self_introspective_meta_learning_v1/neural_curriculum.py --seeds 20,21,22,23,24 --steps 320 --batch 32 --width 48 --interval 16 --output /tmp/siml01n.json
python research/self_introspective_meta_learning_v1/verify_neural_receipt.py --receipt /tmp/siml01n.json --retrain
```

CPU PyTorch is an optional dependency for the neural pilot; pure standard-library finite tests remain independent. Source-hashed receipts allow reproducibility checks; exact reproducibility across heterogeneous torch kernels and devices requires additional controls.

## Next study

Run the same interface on independent task generators, including unseen intervention primitives, with neural student checkpoint files, budget-constrained model-proposer calls and a physically isolated broker. Compare learnable curriculum policies at the same total training/inference cost. FAW/FRC can be integrated after the basic pipeline has demonstrated repeatable neural transfer.
