# SCM-21 preregistration: stronger distillation and bounded causal updates

Client date 2026-11-17, Asia/Taipei. Based on SCM-20 commit
`c8b07c8ce1f025da661334a63dd75bbd03fce957`. Seal source/configuration,
unchanged oracle and warm-start hashes in `results/registration.json`, commit
and push before any SCM-21 training or development scoring. Small CPU interface
fixtures and reconstruction of the already-known historical encoding catalog
may run before sealing. Historical results motivate this study; they are not
new selection or confirmation data. Null results remain valid outcomes.

## Question and frozen components

Test whether stronger clean distillation and smaller causal updates can retain
clean top-one mathematics while improving native VPD-write interchange.
Restart from the three original SCM-19 `frozen_s191/1191/2191.pt` checkpoints,
not the accuracy-degraded SCM-20 causal checkpoints. Keep the same coefficient
encoder, four visits, 128 hidden units, shared SAE, linear head and head-fiber
projector. Only 1,552 existing rank-four adapter parameters can change. Verify
all protected weight/dictionary/projector hashes against the warm starts.

Prediction is one neural argmax over 64 programs, with zero oracle/candidate
execution/verifier/search calls to select it. The independent exact integer
EvalMath ARG-01 oracle grades afterward. Pinned SHA-256:
`c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b`.
The official EvalMath roster is forbidden in every process.

## Loss, optimizer and genuinely bounded parameter increments

For each four-pair batch, run clean recipient, clean donor, and patched
recipient. At zero-based visit 1 (after the second visit), replace only the
recipient CLS native adapter write with the donor's native write. Gradients
flow through both writes. Use the inherited setNLL over case-verified correct
programs, accepting equivalent programs. Positive labels use SCM-19's fixed
training-only finite-probe proposals, each verified on the actual training
case's full demonstrations/query future. No algebraic certificate is claimed.

L = clean_setNLL + 10*KL(warm || current clean)
    + lambda*causal_setNLL + .001*mean clean native-write squared energy.

Clean loss/KL average recipient and donor. Causal setNLL targets the donor's
verified correct-program set and applies only to training pairs both solved
exactly by that seed's frozen warm start; the eligibility mask and teacher
logits freeze before updates. All pairs receive clean loss and distillation.
A batch with zero eligible pairs contributes zero causal loss, not a claim of
perfect causal prediction.

Three pilot arms: lambda in {0,.25,.5}. KL weight 10 is five times SCM-20's 2;
causal coefficients are two/four times smaller than its selected 1. AdamW
constant LR .001 (one third of .003), weight decay .01, gradient-norm clipping
1 (previously 2). FP32 only. The zero-causal-loss arm uses the identical three
forward/backward graphs, warm weights, pair sampler, optimizer and step bound.

After each AdamW proposal, project the entire adapter increment onto an L2 ball
of radius .01 about its pre-step parameter vector. If capped, scale the
increment to .9999*.01, leaving FP32 roundoff margin. AdamW moments retain the
unscaled proposal history. This is a projected-step optimizer, not ordinary
AdamW. Log raw increment L2, actual increment L2/L-infinity, cap status and
gradient norm. Enforce actual L2 <= .01*(1+1e-4) numerical tolerance. This bounds
each optimizer increment; it is not a bound on total displacement or clean
accuracy. Do not infer measured SCM-20 increment sizes from learning rates.

## Freshness and full family/topology coverage

Deduplicate public signatures against recorded SCM-18/19/20 roles and final
cases. Also reject exact coefficient-encoder input repeats, hashing only the
first 28 float32 features actually consumed by the model (two demonstrations,
topology/degree/demo indicators; query is not encoded). The initial encoded
catalog covers SCM-19/20 training caches, clean development/calibration,
development/calibration pairs and prior public final cases. SCM-19 calibration
pairs are reconstructed from its fixed recipe and checked against frame hashes.
Keep hashes of catalog source files. Prior model final cases never train this
study, even when queries differ but encoded demonstrations coincide.

Maintain one signature/encoding registry across every SCM-21 role. Reject
within-pair identical encodings/public observations, identical exact futures,
or any prior-role public/encoding collision. Final cases also remain mutually
distinct by actual encoder input. Generation never consults model correctness.

Generate accepted quotas per family/topology cell, differing by at most one
when count is not divisible by the cell count. Within a cell, attempt a case
with seed = role_start + cell_index*100000 + attempt and program index
(17*attempt)%64. Cell order is family-major, topology-minor. Stop/reject as
specified, then deterministically shuffle the accepted role pool with RNG seed
role_start+99999. Cap attempts at max(1000,quota*200) per cell; fail rather than
silently relax freshness. Pair recipients/donors hold input series, topology
and operators two/three fixed, cycling operator one I→N→S→R→I.

ID families: signed/product/sparse, across all four topologies. OOD families:
rational/squared/lacunary/twisted_product, across **all 16** family/topology
combinations, correcting SCM-20's coupling. OOD cases never receive updates.
This new balanced/novel-input distribution prevents a direct cross-study
attribution from raw SCM-20 scores; warm and matched controls use identical new
cases within this study.

Fixed role sizes and starts:

| Role | Count | Start |
|:---|---:|---:|
| Training ID pairs | 4096 | 210000000 |
| Development clean ID / OOD | 1024 / 1024 | 220000000 / 230000000 |
| Development pairs ID / OOD | 512 / 1024 | 240000000 / 250000000 |
| Calibration clean / pairs (ID) | 64 / 32 | 260000000 / 270000000 |
| Fresh clean ID / OOD | 1024 / 1024 | 280000000 / 290000000 |
| Fresh pairs ID / OOD | 512 / 1024 | 300000000 / 310000000 |

## Fixed development choice and clean-accuracy buffer

Seed 191 trains all three arms to 4000 updates; save immutable checkpoints at
1000/2000/4000. Evaluate the two nonzero causal arms at those fixed checkpoints.
A candidate must keep both development clean accuracies within **.01** of warm
and have at least 64 jointly clean-correct warm/candidate pairs in each fixed
development stratum. Assay the first 64 eligible pairs per stratum for native
and norm-matched random fiber writes after visit two.

Among admissible candidates, maximize the worse ID/OOD native-minus-random
fidelity, then its mean, lower lambda, earlier step. If no candidate is
admissible, promote none and retain the warm starts as an explicit failed-study
fallback. Otherwise repeat the selected lambda and common step budget in seeds
1191/2191, each with its matched lambda-zero control. No tuning or extension of
these seeds. Seed 191 uses its control checkpoint at the selected step. Count
all unused pilot updates and development evaluations; do not claim identical
end-to-end selection compute between arms. Repeated checkpoints are not new
independent configurations or confirmation cases.

## Freeze, common-eligibility assay and registered confirmation

Calibrate the same PCA-4, difference-PCA-4, SAE group-4 and native write-span-4
directions on the fixed calibration cases only. Freeze/hash every compared
checkpoint, SAE, projector, direction, source and selection before creating any
fresh clean/pair cases. A fixed random rank-four fiber basis is shared across
warm/control/causal models within each seed. No final feedback changes training,
thresholds, direction choice, pool size or which checkpoint is reported.

Report each model's own pair eligibility. Primary comparisons use the
intersection where both recipient and donor are exactly clean-correct under
all three models. Assay the first 128 such pairs per stratum, with identical
IDs across models/methods. CLS-only methods after each of four visits: sham,
full CLS, head-readable, head-fiber, PCA-4, difference-PCA-4, SAE group-4,
native span-4, actual native VPD-write swap and fixed random fiber-4. Match random
patch norms to native deltas per pair/model/visit. Four coordinates are global
CLS intervention coordinates, not four at every token. Full CLS at visit four
is the positive control; sham/terminal fiber are null controls. Undefined
fidelity at zero tests remains undefined.

A robust positive requires promotion and **all three seeds / both strata**:
at least 64 common-eligible pairs; native-write fidelity >=.90 after visit two;
>=.15 advantage over each random, warm-native and control-native endpoint on
the same assayed pairs; and fresh clean accuracy loss <=.02 versus warm in each
split. The .01 development buffer does not guarantee this .02 final criterion.
Report smaller gains and clean failures explicitly. Secondary visits/directions
cannot rescue the primary endpoint. No unique latent variable, natural-language
reasoning, algebraic-equivalence, continuum or topology certificate is claimed.

## Local resources, accounting and artifacts

RTX 3050 4GB only; existing Python 3.12/PyTorch 2.9.1+cu126, no installations or
cloud compute. FP32, full-model batches <=8, four visits, 3000 MiB allocator cap,
at least 500 MiB sampled free. CPU-cache features/CLS summaries, sequential GPU
jobs, release memory between models. Stop on nonfinite loss/increment, exceeded
step or VRAM bounds, or disk exhaustion. Do not kill unrelated processes or
change device power limits.

Append/flush forward/prediction/oracle/update/cell events; save optimizer/RNG
state every 500 steps and atomic resource/oracle/cache snapshots at checkpoints
and on caught exceptions. Count worker and independent CPU audit separately.
Verify frozen hashes, exact outcomes, shared eligibility, norms, public and
encoder novelty, factorial quotas, actual parameter-step bounds and complete
accounting. Disclose any interruption or missing counters. Keep raw gold,
weights, caches and journals local outside Git. Produce report/figures/receipts,
a local results file URI and a draft PR on a separate branch. All commits must
use `[skip ci]`; no hosted experiment or merge is authorized.
