# SCM-21 audit protocol, post-hoc version 1

This is **retrospective**. SCM-20's results were inspected on 2026-11-17 before the diagnostic endpoint was defined. Do not describe this as a newly preregistered model experiment or an independent held-out test. The sole already-executed neural data originate in SCM-20.

## Frozen source

Repository: `MoralityLabAI/Compactification-Theory`, SCM-20 PR #180, public path `research/dimensional_atlas/gym/scm20_causal_consistency/results/summary.json`. Git blob SHA-1: `dd0fa7005aa67fd4e0d5b3b9faf513ff32744713`.

Extract only the 60 method-level cells with model=`causal`, visit=`1`, seeds `[191,1191,2191]`, strata `[id,ood]`, and ten frozen method names. Six denominators are `[128,120,128,128,128,116]`. Revalidate against the full 720-cell secondary assay and registered primary outcome when the entire source JSON is available. SCM-20's independent audit reports source-bound pair identity, freeze chronology and model hashes; SCM-21 cannot independently replay those private artifacts.

## Exact comparison

For any fixed set of n common-eligible recipient/donor pairs, two methods A and B with respective successes a and b, let x count successful A / unsuccessful B. Without knowing individual pair identities, the **sharp** possible range is

`max(0,a-b) <= x <= min(a,n-b)`.

Proof: the intersection of the a and b successful-pair sets ranges from `max(0,a+b-n)` through `min(a,b)`. Therefore `x=a-intersection`. Enumerate every pair of subsets for n up to 6 in tests. Sum the lower bounds separately over six seed-stratum groups; this counts repeated tasks across seeds as separate evaluations, not independent trials. Per-group method rankings can be established from exact counts alone.

Compare `semantic_diff4` (a rank-four projector built using calibration-pair donor-recipient differences), `pca4` (a calibration-state PCA projector), `native_write4` (the projector onto the learned rank-four native write span, applied to the full state difference), `native_vpd_write` (the actual donor-recipient native adapter-write delta), `random_fiber4`, `full_cls`, and the other existing controls. All use zero-based visit=1, same model, same matched pair IDs within each seed and split. `full_cls` has 128 dimensions and `head_readable` has a larger subspace; neither is a rank-four comparator.

## Confounding and scope

- The mean delta norms for `semantic_diff4` exceed the corresponding `native_write4` mean norms by 2.5%–6.2%. SCM-20 did not apply a per-case norm matching operation between those two projections, so the outcome contrast cannot isolate angular alignment.
- Aggregates do not supply pair-level cross-method overlap; assert sharp bounds, not signed-rank tests, per-pair wins, or independent confidence intervals.
- Means across three seeds share the same final cases. The six seed/stratum cells are paired assay blocks, not 748 independently sampled observations.
- Projection capacity and exact downstream interchange function are distinct. A high projected norm fraction alone cannot establish behavioral sufficiency; an observed exact donor-replacement accuracy is stronger but still task-specific.
- The SCM-20 selected causal checkpoint lost clean accuracy in five of six seed/split comparisons. Any subsequent training requires preserving this failure as a fixed counterexample.

## Separate constructed CPU control

Build an R128 linear readout of rank four and two orthogonal rank-four subspaces. Fit one projector using 80 calibration donor differences; compare it with the other projector on 256 fresh synthetic pairs with *exactly matched patch norms per pair*. This construction illustrates why identical rank and norm need not imply equal causal capability. It does **not** simulate trained SCM-20 or create a new EvalMath result.
