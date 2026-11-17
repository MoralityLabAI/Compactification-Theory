# SCM-19 preregistration — clean mathematics before mediation

Registered 2026-11-15 on a new branch based on SCM-18 commit
`714c5ba87bb17a16f1ce4814c02abde2d1093e9d`. Configuration and implementation
hashes are sealed and committed before any follow-up training or development
readout. Small unit/interface fixtures may run before sealing. SCM-18's final
results motivated this follow-up; its cases and scores are not follow-up
selection data. This experiment makes no claim of a new blinded competition.

## Primary question and clean prediction

First acquire independently correct top-one mathematical program predictions,
then test a named first-operator intervention on fresh held-out pairs. A clean
prediction is one neural forward and one argmax over 64 programs, with **zero
oracle/verifier calls, candidate executions or search before choosing it**.
An evaluator subsequently checks the chosen program's observed demonstrations
and withheld query future with the unchanged EvalMath integer oracle. Selecting
a program through 12 verifier attempts is not a clean top-one result.

Pin EvalMath to commit `88bb6cf8d2fe172e37d25964dfb2c285d9033841`, source SHA-256
`c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b`.
The official EvalMath roster is forbidden in every SCM-19 process. All new data
come from new seeds, with public-signature deduplication against SCM-18 and
across follow-up roles. Original RMP gates, checkpoints and sealed data remain
untouched. No package installation or cloud compute is required.

## Fixed changes and controls

Two input encodings feed a weight-tied, four-visit, 128-hidden transformer with
a rank-64 linear readout. Both represent the two demonstrations coefficient by
coefficient (26 tokens plus a CLS token), without SCM-18's clipping of large
log magnitudes. The same program is inferred from both demonstrations; public
query coefficients are unnecessary for program selection and are not encoded.

1. **Coefficient:** signed log, bounded signed magnitude, zero indicator and
   modular residues of public input/output coefficients and their sum/difference.
2. **Polynomial Gram:** additionally represent candidate-independent input
   monomials indexed by integer partitions through degree eight (at most 22
   monomials), normalized jointly with the observed output coefficient. Their
   outer products give generic quadratic relationships. No candidate program,
   oracle output, hidden source graph or future is used to compute these features.

Degree-specific learned input projections share weights across demonstrations.
Their mean initializes CLS, followed by the same recurrent core. The polynomial
arm deliberately supplies stronger mathematical preprocessing; parameter count,
feature arithmetic, token-visits and measured work must be reported. A gain
cannot be attributed to recurrence or raw-token reasoning. Frozen zero-visit
CLS readout is an additional diagnostic, not a selectable architecture.

Training uses a set-valued correctness loss: negative log probability mass on
all verified correct programs in a discovered equivalence group. Fixed signed
series probes at full horizon 32 group programs separately by topology. These
groups are **finite-probe candidates, not certified algebraic identities**.
Every positive training label is independently rechecked on that training
case's full two demonstrations and full query future. This removes penalties
for verified equivalent outputs without accessing any held-out labels.

All generated pools enumerate topologies cyclically and program indices with
the fixed stride-17 permutation modulo 64. This spreads the four first-operator
values across small development/calibration pools. Public-signature rejection
can perturb the resulting balance; report actual case/family counts.

The historical three SCM-18 tied models are scored on the same new development
and clean held-out examples as a reference. Architecture, input encoding,
training data and compute differ; this comparison does not isolate their causes.

## Acquisition and selection

`config.json` fixes all numbers and coefficients. Generate 8,192 training tasks
from signed/product/sparse families, balanced over topologies and source programs.
Generate 256 ID and 128 OOD development cases. Both encodings train seed 191
for 4,000 updates, batch eight, AdamW learning rate .001, weight decay .01.
Report curves every 500 steps; only the step-4,000 endpoint selects the encoding.
Restrict to arms passing both clean gates when any exists; otherwise consider
both. Choose highest ID clean accuracy, then OOD, then the simpler coefficient arm.

The clean gate is >=50% ID and >=25% OOD development accuracy. If neither arm
passes both gates, extend only the selected arm to exactly 8,000 updates. This
extension rule is fixed before results. Train seeds 1191 and 2191 with the chosen
encoding and same final step budget. No additional architecture search or
extension beyond the registered budget is allowed. If the gate remains unmet,
report an acquisition failure and perform only the qualified held-out eligibility
audit; never fabricate a mediation denominator or claim a mechanism.

After base training, freeze its parameters and readout frame. Fit a shared
TopK SAE on CPU-cached training activations for 400 updates. If the first-seed
clean gate passed, fit a rank-four fiber-only VPD adapter for 400 updates with
correct-program loss + .5 KL to frozen-base predictions + .01 correction energy.
Fit each seed independently with identical examples/update seeds and budgets.
Report the three losses separately. The base and dictionary remain frozen.
Evaluate base and adapter separately even if the adapter hurts development
accuracy; no final-driven adapter rejection or retuning is allowed.

## Split and freeze boundary

Fresh seed ranges start at: training 19100000; generic grouping probes 19200000;
development ID 19300000; development OOD 19400000; calibration cases 19500000;
calibration pairs 19550000; clean final ID 19600000; clean final OOD 19700000;
mediation ID 19800000; mediation OOD 19900000. Train/ID families are signed,
product and sparse. OOD families are rational, squared, lacunary and
twisted_product. OOD development means these families are not blind source
families; they remain absent from base/adapter training.

Use 64 fixed calibration examples for PCA and SAE feature usage, and 32 separate
calibration pairs for first-operator difference-PCA directions. Freeze all base
weights, dictionaries, adapters, projections, data choices and code hashes before
creating/scoring the fresh 512 ID + 256 OOD clean cases or 256 ID + 128 OOD
mediation pairs. New gold stays in a private artifact outside Git. Hold-out
case creation is inaccessible to the acquisition controller before freeze.

## Semantic mediation

Pairs use the same source input series, topology and other two operators, with
the first operator cycled `I→N→S→R→I`. Reject only generative degeneracies:
identical oracle futures, identical public demonstrations/encoded features,
or public-signature overlap with earlier roles. No model correctness or patch
success may influence pair generation. Select eligible pairs only after the
fixed pool is frozen: both clean recipient and donor must have correct exact
top-one futures under the policy being tested. Cap the assay at the first 64
eligible ID and first 64 eligible OOD pairs; report every attempted/eligible count.

Patch **CLS only** after each of four visits. Thus k hidden directions mean k
global intervention coordinates, not k per every input token. Registered methods
are sham, full CLS, head-readable, head-fiber, calibration PCA-4, semantic
difference-PCA-4, SAE feature-group-4, native adapter write-span-4, actual
SAE-conditioned fiber-write swap, and a fixed random fiber rank-four control.
The random patch is norm-matched per pair to the actual VPD write swap. Full CLS
at the last visit is the donor-readout positive control; last-visit fiber is the
linear-head null control. Preserve patch norms and exact outcomes.

Primary endpoint: actual VPD fiber-write swap after the **second** visit
(zero-based visit 1). Require at least 64 eligible ID and 32 eligible OOD pairs
per seed, >=.90 causal fidelity, >=.15 advantage over the matched random control,
and <=.02 loss of fresh clean accuracy versus the frozen base. A robust positive
requires all three seeds; otherwise report exploratory signals or a null.
Other visits/methods are prespecified secondary diagnostics, not substitutes for
the primary endpoint. Fidelity is correct counterfactual outcomes / valid tests;
a zero denominator is undefined. No topology certificate is sought here.

## Hardware, durability and accounting

RTX 3050 4GB, local CPU/GPU only, FP32, batch <=8, 3000 MiB allocator cap and
at least 500 MiB observed free memory. Stop on insufficient headroom, nonfinite
loss or disk exhaustion; never kill unrelated processes or alter power limits.
Cache inputs, SAE targets and activation summaries on CPU. Release CUDA memory
between models. No mixed precision.

Use a hidden local worker, append-only forward/update/oracle journals, periodic
optimizer checkpoints and immutable completed-stage receipts. Record model
calls, examples, recurrent token/example-visits, head calls, optimizer steps,
post-prediction oracle checks/grades, source-execution cache hits/misses, wall
time, and peak allocated/reserved/free VRAM. Flush journals continuously and
record interrupted attempts with their last durable event. Final methods share
case pools, batch size and one neural prediction per patched outcome; report
extra preprocessing/projection costs rather than claiming exact FLOP equality.
