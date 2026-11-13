# SCM-02: RMP cyclic-mechanism pilot, exploratory registration

**Written 2026-11-13 before any trained-model SCM-02 activation traces were run.**
The file registers instrumentation, selection discipline and falsifiers.
No RMP checkpoint or local dictionary from the user's Windows machine is
mounted in this execution environment, so no learned-model outcome is
registered or inferred here.

## Fixed observational construction

- Input: one deterministic frozen RMP `AnalysisModel` checkpoint from
  `MoralityLabAI/LoopedTransformers` branch `rmp-v0-phase2` with its source
  `research_gym/interp/phase2.py` cell registry.
- Initial models: `LOOP-T s1` and `FF-U s3`, both checkpoints with recorded
  G0 status and their **existing** `D-shared` dictionaries. Later controls:
  remaining tied seeds, their random-initialized `C0` counterparts, and the
  fixed random-chart control on the *same input and perturbation directions*.
- Intervention: at a single fixed token position, after visit 0, pulse
  `delta(theta) = epsilon*rms(resid[1]_source) * (cos(theta)*u + sin(theta)*v)`;
  `u,v` are the two preselected normalized decoder rows of the existing
  dictionary. The input tokens and model stay frozen throughout the cycle.
- Readout: two different downstream named token positions. They **exclude**
  the directly injected position, preventing the identity component of the
  experimental injection from masquerading as a propagated reasoning circuit.
  At visits 1..8 take two scalar linear projections from the same frozen chart
  for both token states. For `FF-U`, the existing input re-injection remains
  active through `AnalysisModel.run`.
- Initial resolution: 32 angular segments plus the repeated endpoint;
  Gate B repeats the same experiment at 64 segments, holding all other choices
  fixed. Explicitly distinguish a polygonal invariant from a certified smooth
  path invariant. Default clearance: 1e-7 in measured chart coordinates.
- Chart selection: initial SAE decoder-row chart and deterministic
  random-orthogonal readout control. **The perturbation axes remain unchanged
  when only the readout chart changes.** Record every chart/axis SHA-256.
- Initial amplitude grid: `epsilon in {0.02, 0.05, 0.1, 0.2}` in source-token
  residual RMS units, submitted as separate immutable receipts. Chart feature
  indices and token positions must be frozen and recorded *before interpreting
  any topology results*. The command line makes them explicit; this document
  does not claim they have already been chosen for the user's checkpoints.

## Sampling and selection

The initial pilot takes **64 group-unique examples**, 16 per core family,
from the hashed *selection half* of `monitor_calibration`, with deterministic
seed 20261113. No previous seal is modified. Never perform chart selection on
the final test or reporting half. Each output JSONL row records model name,
checkpoint/dictionary/chart/axis hashes, group and task hashes, token hash,
intervention site, amplitude, angular points, all visit-wise projected paths,
final output predictions, analysis validity, and exact rejection reasons.
Run separate file IDs for each condition. Never overwrite a receipt.

## Gates and decisions

**Gate B (repeatability):** report valid-extraction fraction per model,
family, visit, chart, amplitude, and the exact fraction of valid cases agreeing
under angular doubling on pairwise winding, Alexander polynomial and
boundary component count. A failed sample remains a failed case. A robust
nonzero pair winding constitutes a *projected* discriminant candidate;
continuous-path certification requires the independent derivative-bound
contract in `SCM02_MATH.md`.

**Gate C (causal usefulness, future protocol):** freeze at least 128 family-
disjoint tasks after chart/feature/radius selection. Record named intermediate
variables, task correctness under matched causal feature ablations or patching,
and topology descriptors with no label leakage. Compare against a matched
state-only PCA/SAE baseline, shuffled topology labels, random charts,
random-initialized models, and linearized Jacobian responses. Match forward
calls, model parameters, compressed bit budgets and clean-task accuracy when
claiming a compactification advantage. Report confidence intervals by
bootstrapping groups; use a held-out prediction metric selected in advance.

**Gate D (4D surfaces):** requires a mathematically valid embedded relative-
boundary surface construction and an isotopy-sensitive surface invariant;
SCM-02 does not attempt this.

## Failure interpretation

- Topology changes under 32-to-64 doubling: unresolved sampling or projected
  collision proximity; reserve it as an explicit invalid or unstable case.
- Topology appears only at the directly injected position: identity artifact;
  that case is excluded by construction.
- Stable topology appears equally in random charts and trained/untied
  controls: chart-level effect without evidence for recurrence specificity.
- Topology predicts no incremental, out-of-family causal change after matching
  baselines: the compactification hypothesis fails for the tested setting.

## Existing program invariants

Leave unchanged: RMP G0 accuracy threshold 0.98, visits 1..8, task split
definitions, existing registered feature dictionaries, all sealed metrics,
checkpoints and baseline reports. This code creates a separate experimental
namespace and never modifies RMP training or checkpoint loading functions.
