# SCM-12: rank-one response funnels and matched causal topology audits

**Scope, November 14, 2026.** Exact finite-dimensional mathematics and an executable, synthetic-only instrument. No trained `LOOP-T` or `FF-U` weights or D-shared dictionaries were available to this run. The experiment measures oriented braids in a frozen *planar readout chart*, rather than intrinsic neural-state topology or embedded four-dimensional Seifert surfaces.

## 1. Rank-one response obstruction

Let `n >= 2` labeled projected points depend continuously on the cyclic intervention parameter `theta`:

`z_i(theta) = a_i + b_i f(theta) + e_i(theta)`,

where `a_i,b_i` are fixed vectors in `R^2`, `f:S^1 -> [l,u]` is continuous, `e_i` is continuous and `||e_i(theta)|| <= eta_i` for **every** angle. Define the *affine relative minimum clearance*

`d_ij = min_{t in [l,u]} || (a_i-a_j)+(b_i-b_j)t ||`.

**Theorem SCM-12.A.** If `d_ij > eta_i+eta_j` for every pair, the loop `theta -> (z_1(theta),...,z_n(theta))` is null-homotopic in the ordered planar configuration space `Conf_n(R^2)`. Its pure braid element is the identity, and the closure is the labeled unlink; in particular all pairwise winding numbers and defined Milnor invariants vanish.

**Proof.** For `s in [0,1]`, set `H_s(theta)_i = a_i+b_i f(theta)+s e_i(theta)`. The pairwise distance satisfies

`||H_s(theta)_i-H_s(theta)_j|| >= d_ij-(eta_i+eta_j) > 0`.

Hence `H_s` is a free homotopy through collision-free configurations. Its endpoint factors through the contractible real interval `[l,u]` and therefore contracts through `Conf_n(R^2)`. If the initial configuration is held fixed, one can base the contraction there. QED.

**Exact checker.** `scm12_rank_one.py` treats all inputs as `Fraction`, finds the minimum distance from the origin to each rational relative segment, and rounds the square root **downward** on a stated rational decimal grid. It issues `conditional_null_braid` only under a strictly positive margin. Its supplied uniform `eta_i` remain unverified assumptions until a source verifier proves them. This checker makes no assertion about trained hidden states by itself.

**Dimensional implication.** A nontrivial braid cannot arise from a collision-free continuous factorization through a one-dimensional interval. This is a precise necessary condition on any claimed topological feature of a compressed recurrent representation. Effective low rank alone, judged from finite activations, is insufficient for invoking the theorem; its uniform approximation bound must be independently established.

## 2. Exact gauge specialization

A pre-normalized RMP block satisfies `B(X+U)=B(X)+alpha^2 U` for tokenwise channel-constant shifts `U`, as proved in SCM-11. A gauge perturbation to the injected token alone therefore remains at that token after subsequent blocks and leaves every other token's states unchanged. Fixed embedding reinjection commutes with the difference calculation. If one circular intervention axis is an *exact* channel-constant vector and the second is arbitrary, then for observed tokens excluding the injected position, all projected outputs depend only on the second axis coefficient `sin(theta)`.

The observed loop thus factors through `[-1,1]` even when its scalar response is nonlinear. Under full-path collision freedom it is contractible. This is an exact-real null control for the specified architecture; floating-point execution and sampled apparent crossings remain distinct verification problems.

## 3. Paired causal experiment contract

An observational topology metric is meaningful for mechanistic reasoning only after source, intervention, and task matching. SCM-12 freezes the two original RMP cells (`LOOP-T s1`, `FF-U s3`), original `C3` checkpoints, D-shared dictionary selection, two feature IDs per cell, selection-half monitor calibration data, intervention radius `0.05`, token `4`, output slots `16,32,48`, and `32/64` angular sampling. The four arms are:

1. LOOP-T with frozen D-shared chart.
2. LOOP-T with random orthogonal **readout only**.
3. FF-U with frozen D-shared chart.
4. FF-U with random orthogonal readout only.

Crucially the intervention axes are unchanged within each model's two arms. Therefore their output logits and predictions should agree for each task and angle; any disagreement causes a rejection. This separates dependence of the braid on the measurement chart from effects of the intervention on the model's answers. Source checkpoint, dictionary, task ID, group ID, target, token sequence, feature IDs and intervention axes are checked for exact identity. Tasks must be unique by calibration group, 16 per family at the 64-task target.

The offline `scm12_compare.py` reports per-visit exact polygonal Artin-action validity, any detected nontrivial braid, unresolved witness counts, mean prediction flips, and an **exploratory** exact two-sided sign test of SAE-only versus random-chart-only nontrivialities. This sign test is descriptive because feature/chart selection and multiple testing are not corrected here. Nontrivial observed braids remain polygonal until continuous-path and numerical execution certificates are established.

## 4. Source and security boundaries

The `scm12_launch.py` preflight refuses absent RMP source files, chosen checkpoints and their integrity sidecars, matching training results, selected D-shared dictionaries, the original calibration bundle, dictionary-selection metrics, selection-provenance record, or frozen feature IDs. The dictionary is inspected using PyTorch's `weights_only=True` loader and its dimensions/selection metadata must match the RMP metrics. A source/dictionary mutation after manifest lock prevents execution. The launcher creates new output paths and invokes existing `rmp_capture.py` without changing the RMP repository, gate G0, or held-out split.

The prior source-only `scm11_audit.py` remains a separate numerical proof stage after capture. The current workflow does **not** certify CPU/GPU float activations, trained-model topological classes, compactified neural capabilities, or four-dimensional smooth surface isotopy.
