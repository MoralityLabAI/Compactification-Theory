# SCM-20 preregistration: causal consistency with preserved clean accuracy

Registered client date: 2026-11-16, Asia/Taipei. Based on SCM-19 commit
`f54ac7b01eb1903fc7c4d0f1f1d156714a89764b`. Seal source/configuration, pinned
oracle and three warm-start checkpoint hashes in `results/registration.json`,
then commit before any SCM-20 training or development scoring. Unit/interface
fixtures may run before sealing. Earlier results motivate the study but are
excluded from new training, selection, calibration and final case roles.

## Question and fixed architecture

Can explicitly training native-write interchange improve first-operator causal
mediation while retaining SCM-19 clean top-one accuracy? Reuse each seed's
frozen SCM-19 coefficient-encoding, four-visit, 128-hidden model and shared SAE.
Keep input projections, positions, tied core, head, SAE and head-fiber projector
byte-identical. Only the same rank-four adapter parameters can change. Warm
starts are `frozen_s191.pt`, `frozen_s1191.pt`, `frozen_s2191.pt` from SCM-19.
No new architecture, preprocessing, rank or per-token coordinate budget is tuned.

A clean prediction remains one neural argmax over 64 programs, with no oracle,
candidate execution, verifier or search in prediction. The unchanged independent
EvalMath ARG-01 integer oracle grades the chosen prediction afterward.
Oracle SHA-256: `c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b`.
The official EvalMath roster is forbidden in every process.

## Fresh data and labels

Use new deterministic seed ranges and signature deduplication against all SCM-18
and SCM-19 prior-role/final signatures and across all SCM-20 roles. Public
signatures exclude case IDs, source graphs and hidden futures. Same source
families as SCM-19: signed/product/sparse for ID training; rational/squared/
lacunary/twisted_product remain OOD and are never used for updates.

Training: 4,096 generatively valid first-operator pairs, starting seed 20100000.
Pairs hold input series, topology and operators two/three fixed and cycle
operator one I→N→S→R→I. Reject identical futures, identical public/encoded
observations and signature collisions, without consulting a trained model.
Use the inherited deterministic topology/family cycles and stride-17 program
schedule. Data features are public coefficient features only. Positive program
sets use SCM-19's training-only finite-probe equivalence proposals and exact
verification on each training case's full demonstrations and full query future.
Finite probes are not an algebraic equivalence certificate.

Development clean: 512 ID at 20300000, 512 OOD at 20400000. Development pairs:
256 ID at 20500000, 512 OOD at 20600000. Calibration: 64 ID clean cases at
20700000 and 32 ID pairs at 20750000. Calibration data cannot affect losses,
checkpoint choice or training. New training pair recipes/gold stay private.

Final data must remain uncreated until all weights, checkpoints and directions
freeze: 512 clean ID at 20800000, 512 clean OOD at 20900000, **512 ID pairs**
at 21000000 and **1,024 OOD pairs** at 21100000. The OOD pair pool is eight
times SCM-19's 128 pairs. No correctness-driven extension of final pools.

## Causal-consistency objective and matched control

Each update samples four pairs (each forward batch <=8). Run clean recipient,
clean donor and patched recipient with the current adapter. At visit two
(zero-based 1), replace only the recipient CLS native adapter write with the
donor's native write: add donor_write - recipient_write. All other token states
remain recipient states. Gradients flow through donor and recipient writes;
there is no detached donor target for the native intervention.

Let setNLL(z,A) = logsumexp(all logits) - logsumexp(correct-program logits).
The clean loss is the mean recipient/donor setNLL. The clean KL is the mean
KL(frozen warm-start prediction || current clean prediction). Causal setNLL
targets the donor's verified program set under the patched recipient prediction.
Apply causal loss only to training pairs that both frozen warm-start predictions
solve exactly; this mask is fixed before training for each seed. Every sampled
pair contributes to clean loss and distillation. If a batch has no eligible
pairs, its causal numerator is zero and denominator clamps at one.

L = clean_setNLL + 2 * clean_KL + lambda * causal_setNLL
    + 0.001 * mean(clean recipient/donor native-write squared energy).

Register lambda in {0,1,4}; lambda zero is the parameter- and update-matched
math/distillation control. It executes all three forward graphs and the same
combined backward graph, with causal coefficient zero. Common initial weights,
sampler sequence and optimizer hyperparameters within each seed. AdamW LR .003,
weight decay .01, global gradient clipping 2, constant LR, FP32. No base/SAE
updates. Report all four loss terms separately and count all work, including
unused pilot updates and development evaluation. Selection evaluation costs
differ from the nonselected control; do not claim end-to-end FLOP equality.

## Development-only promotion rule

For seed 191, train all three lambda arms for 4,000 updates. Save immutable
checkpoints at 1,000, 2,000 and 4,000. Evaluate causal arms lambda 1/4 at these
three fixed checkpoints. A candidate is admissible only if both development
clean accuracies drop by at most .02 from that seed's original SCM-19 adapter.
Measure development native-write and per-pair norm-matched random-fiber
fidelities after visit two on the first 64 pairs jointly clean-correct under
warm start and candidate, in each fixed development stratum.

Among admissible candidates with nonempty tests in both strata, maximize the
minimum of ID/OOD (native fidelity - random fidelity), then its two-stratum
mean, lower lambda, earlier checkpoint. No final data participate. If no
candidate is admissible, promote no causal model and retain warm starts as the
explicit guarded fallback; the registered positive hypothesis cannot pass.
Otherwise train seeds 1191/2191 for the chosen common update budget, each with
the selected lambda and lambda-zero matched control. Do not tune these seeds
or extend training. The seed-191 control uses its checkpoint at the selected
budget, while all 4,000 pilot updates remain in compute accounting.

## Freeze and final common-eligibility assay

For each seed, compare warm start, matched lambda-zero control and selected
causal checkpoint. Fit only descriptive directions using the fixed calibration
data: PCA-4 of CLS, difference-PCA-4, four most-used SAE decoder directions and
native rank-four write span. Fix a random rank-four head-fiber basis per seed,
shared across all three compared models. Freeze/hash every checkpoint, SAE,
head projector, direction, implementation and selection before final generation.

On the fixed final pair pool, report each model's own clean eligibility. The
primary comparison uses the **intersection** where recipient and donor are
exactly clean-correct under all three compared models. Assay the first 128 such
pairs per stratum, preserving common IDs across models and methods. This avoids
comparing favorable, different eligibility subsets after training.

CLS-only methods after each of four visits: sham, full CLS, head-readable,
head-fiber, PCA-4, difference-PCA-4, SAE-group-4, native write-span-4, actual
native VPD write swap, fixed random fiber-4. Random patches match the actual
native-write delta norm separately for each pair/model/visit. A rank-four patch
has at most four global CLS coordinates, not four per token. Full CLS after
visit four is the donor-readout positive control; sham and terminal head-fiber
patches are null controls. Record exact selected predictions, post-hoc outcomes
and norms, including zero denominators as undefined.

## Registered positive criterion

The primary endpoint is actual native-write swap after visit two. A robust
positive requires promotion and, in **all three seeds and both strata**:

- At least 64 common-eligible final pairs in the fixed pool.
- At least .90 causal fidelity on the assayed common pairs.
- At least .15 fidelity advantage over the norm-matched random control.
- At least .15 fidelity advantage over each warm and lambda-zero control's
  native-write endpoint on those identical assayed pair IDs.
- Fresh clean top-one accuracy within .02 of the warm start, separately ID/OOD.

Report improvements below this threshold as limited/exploratory rather than a
positive confirmation. Secondary visits/directions cannot replace the primary.
Do not infer a unique reasoning variable, continuum/topology certificate or
general mathematical reasoning from this finite-DSL assay. The selection
seed's development results are not independent confirmation.

## Local resource and durability constraints

Reuse installed Python 3.12/PyTorch 2.9.1+cu126; no package installation, cloud
compute, hosted experiments or unrelated process termination. RTX 3050 4GB,
FP32, batches <=8, four visits, 3,000 MiB allocation cap, at least 500 MiB
sampled free headroom. Stream CPU CLS summaries; release CUDA between runs.
Stop on nonfinite loss or insufficient headroom/disk space.

Append/flush every forward, selected prediction, oracle check/grade, update and
cell receipt. Save optimizer/RNG state every 500 updates, and atomic current
model-work/oracle/cache snapshots at stage/checkpoint boundaries and on caught
exceptions. Final receipt records cumulative model forwards, token/example
visits, readouts, updates, exact evaluator and pure-source/cache counts, VRAM
and journal hash. If interrupted beyond durable accounting, disclose missing
counts; never invent them. Keep raw gold, weights and optimizer state outside
Git under local outputs. Commit/push a separate branch and draft PR, no merge,
with `[skip ci]` on all commits to prevent hosted experiment execution.
