# SCM-01: Seifert-surface compactification and recurrent mechinterp

**Status (2026-11-13):** mathematical proposal plus executed *synthetic, exact-algebra* CPU tests. No neural activation data, no surface isotopy certificate, no GPU experiments and no claims of hidden knots in a model. The code implements the **closed-braid boundary layer and the combinatorics of its canonical Seifert-algorithm surface**; it does not certify isotopy of surfaces in four dimensions.

## Why Miller's theorem matters

Hayden, Kim, Miller, Park and Sundberg, *Seifert surfaces in the 4-ball*, JEMS 2025, https://doi.org/10.4171/jems/1703, construct pairs of same-genus Seifert surfaces for the same knot whose push-ins to B^4 remain non-isotopic; Khovanov-homology cobordism maps distinguish examples. Kim, Miller and Yoo, *Non-split alternating links bound unique Seifert surfaces in the 4-ball*, Compositio Mathematica 2025, https://www.cambridge.org/core/journals/compositio-mathematica/article/abs/nonsplit-alternating-links-bound-unique-seifert-surfaces-in-the-4ball/04B4F087F03E85E4409B19C10E1BB28F, identify a complementary uniqueness regime for same-genus, boundary-parallel surfaces of non-split alternating links. The distinction suggests **boundary-only information may miss a history-dependent obstruction**. It is a mathematical analogy until the neural construction below satisfies its embedding hypotheses.

An ordinary answer string is neither an embedded knot nor a surface. We must first define an observable knot/link using a reproducible physical experiment.

## Layer 1: construct a closed braid from a cyclic intervention

Fix a trained recurrent model and a single activation site, a deterministic task, `n >= 2` tracked token positions, and a frozen chart `P:R^d -> R^2` (prefer interpretable, registered SAE feature axes; report PCA and random charts as controls). At a fixed earlier site, apply the two-dimensional *cyclic* perturbation

    delta(theta) = eps * (cos(theta) u + sin(theta) v),  0 <= theta <= 2 pi.

The same model input and injection recur at theta=0 and 2pi. Record downstream hidden states `h_i^(r)(theta)` for token/slot `i` at recurrent visit `r`, and form chart points

    z_i^(r)(theta) = P(h_i^(r)(theta)) in R^2.

Only accept a closed braid when the full point configuration is collision-free, the endpoints coincide up to a permutation, and a sufficiently dense sample resolves all crossings. The ordered configuration space has

    pi_1(Conf_n(R^2)) = P_n,

its pure braid group. Forgetting strand labels gives a braid in `B_n`; closing it in the solid torus yields a link `L_r` in `S^3`. If each named slot returns to its initial point, the braid is pure and its closure has exactly n components. This is expected. The input-to-output task may be held fixed across the cycle; an answer is not by itself a knot.

Given the Artin word `beta`, compute its reduced Burau representation `rho(beta)` and the classical one-variable Alexander polynomial

    Delta_closure(beta)(t) ~ (1-t)/(1-t^n) * det(I_(n-1) - rho(beta)),

where `~` means equal up to multiplication by a unit `±t^k`. For a chosen Seifert surface with matrix `V`, also `Delta_L(t) ~ det(V - t V^T)`. These are invariants of the **boundary link**, and in particular *cannot* distinguish two Miller-type spanning surfaces with the same link. The first stage must establish the measurement's reliability before a surface-level experiment is sensible.

For the **canonical Seifert-algorithm surface** of a braid presentation with `n` initial Seifert disks and `c` crossing bands, `chi = n-c`. If the crossing-band graph has `C` connected surface components and the braid closure has `mu` boundary components, its total genus is

    g = (2*C - mu - n + c)/2.

The code computes this genus exactly from a braid word, without requiring an explicit triangulated surface. For example, `sigma_1` and `sigma_1 sigma_1 sigma_1^-1` both close to the unknot, with Alexander polynomial 1, yet their *canonical constructed surfaces* have respective genera 0 and 1. The second presentation carries a canceling crossing pair whose Seifert-algorithm surface adds a handle. This is presentation-dependent and is *not* a knot invariant. It is an elementary same-boundary/different-surface control; Miller's same-boundary **and same-genus** examples demand subtler invariants.

**Interpretation bound:** this construction measures a link in the *fixed 2D measurement chart plus the intervention angle*. It is not a knot invariant of the original high-dimensional hidden-state manifold. Linear mixing or arbitrary projection changes can change the link type. Evaluate a frozen ensemble of charts, report their disagreement, and reject near-collision / unresolved over-under cases. With finite angle sampling, even a collision-free sampled piecewise-linear braid is not a certificate for an unknown continuous neural path without separate derivative/Lipschitz bounds.

## Layer 2: same-boundary explanation surfaces in four dimensions

For a fixed closed link `L`, a *genuine* relative-boundary comparison requires two properly, smoothly embedded orientable surfaces `F_0,F_1` in `B^4` with `partial F_0 = partial F_1 = L`. Different causal intervention fillings provide **candidate maps**, which must be checked for embeddedness and boundary regularity. They may contain intersections and fail to define such surfaces. If embeddings exist, record the genus and crossing/saddle/birth/death movie, then ask whether an accessible 4D invariant distinguishes them. Miller's construction uses Khovanov cobordism maps. Merely obtaining different internal activation histories or different Alexander polynomials is not an isotopy obstruction for two spanning surfaces.

Proposed fillings: (A) continuously shrink intervention radius to zero, (B) progressively ablate/replace a causally selected feature while keeping the same outer-loop boundary. The shrinking map can have singular caps, especially for linked components. Check the surface/immersion contract before any claim of genus or isotopy; for a link with nonzero pairwise linking numbers, disjoint embedded disk fillings in B^4 are obstructed.

## Compactification statement (conditional, exact)

Let `C` denote a *verified* discrete class of a set of finitely many histories, computed from their closed braid/link or from a valid surface invariant. If a deterministic compressed code `E(H)` admits an error-free decoder of `C` for `K` realized classes, `E` needs at least `K` distinct codewords. Any fixed-length binary encoding needs at least `ceil(log2 K)` bits. Under a probability distribution, zero-error variable-length coding has expected length bounded by the appropriate source-coding entropy. This follows by pigeonhole/counting and does not prove that the model itself uses or needs a knot representation. Test if preserving causally meaningful distinctions is more efficient with topology-aware compressed features than PCA or matched-size SAE baselines.

**Falsifier:** if inferred knot classes change arbitrarily under stable chart choices, endpoint selection, infinitesimal perturbation, or matched-feature ablations, they are projection artifacts for this protocol. If they are stable but do not predict causal mechanism differences beyond standard representation baselines, topology adds no measured interpretive benefit.

## SCM-01 executable reference

- `knot.py`: exact reduced Burau matrices, normalized Alexander polynomial over `Z[t,t^-1]`, canonical Seifert-surface Euler/genus statistics, conservative piecewise-linear crossing extraction, synthetic braid fixtures. Requires Python 3.11+ and `sympy`.
- `tests/test_knot.py`: trefoil, figure-eight, Hopf link, unlink, exact surface genus, canceling-band genus counterexample, braid relations, Markov stabilization, conjugation, trace extraction, affine-chart robustness, malformed open boundary and unresolved crossing.
- Run: `python -m unittest discover -s research/seifert_activation/tests -v` from repository root.
- Synthetic fixtures validate code for *known mathematical braids*. They are not mechanistic evidence.

Trace input to `extract_closed_braid` is `frames[angle_sample][strand_id][x_or_y]`. It checks frame/strand shape, finite values, sampled collisions, sampled x ties, endpoint set equality, crossing over/under, adjacency, triple x crossings and simultaneous events. The outcome is an Artin word with sample times. It is an exact invariant of the *accepted piecewise-linear interpolation* modulo its explicitly stated measurement convention, not of an arbitrarily sampled continuous path.

## SCM-02: RMP integration proposal, not executed

**Source:** `MoralityLabAI/LoopedTransformers`, `docs/rmp_handoff.md`, existing `D-shared` and `D-visit` dictionaries, visits `r=1..8`, `LOOP-T` tied model and `FF-U` untied control. Preserve Phase 2's G0 threshold (0.98), original splits and all sealed registrations. Do not train new dictionaries on the final evaluation examples or change the baseline RMP experiments. Use a separate, clearly exploratory experiment namespace and immutable run IDs.

1. Lock checkpoint hash, task hashes, `n` token identities, SAE decoder/encoder hash, perturbation axes, and an affine-whitened chart. Use identical chart across visits for a given model; reject ambiguous token/feature assignments.
2. On a 64-example exploratory split, use 32 angular samples, 4 *preselected* charts and a registered radius grid. Record the model's output and all 8 visits, so each intervention gives visit-specific observations. Do not select charts for topological richness on the final split.
3. Analyze `validity_rate`, link-component count, Alexander polynomial, optional pairwise linking number, feature-label association and repeatability under doubled angular sampling. Run identical charts on random-init, FF-U, shuffled task content, noise-matched perturbations, linearized Jacobian response, and PCA/state truncation with matched reconstruction error.
4. For surviving conditions, freeze a 128-example held-out set, 64 angles and the chart/radius choices. Compare whether topology predicts a causally validated mechanism (actual feature ablation/patch change in task behavior), conditional on output, activation norm, task family, accuracy, and visit index. Use task-level bootstrap, permutation testing and family-disjoint validation. Report the chart selection cost and failed/unusable traces.
5. Compare compressed students (a) state-only SAE/PCA, (b) state plus a *measured* topology descriptor and (c) topology-label-shuffled. Equalize trainable parameters, bits, clean-task performance and measured forward calls. Assess out-of-family preservation of the counterfactual behavior. Interpretation requires causal effects; a high AUROC for a projected link alone is insufficient.

Expected interface: a standalone extractor/adaptor exporting JSONL manifests (`task_id`, `checkpoint_sha`, `chart_sha`, `token_ids`, `visit`, `angles`, `points`, `output_hash`, `perturbation_hash`, `seed`), with heavy activation arrays stored outside git. A model run must additionally provide evidence that angles parameterize a genuinely cyclic intervention and quantify remaining discretization uncertainty.

## Scientific gates

- **Gate A:** synthetic exact algebra and braid-relator / Markov tests pass. A failed check means the topology instrument is invalid.
- **Gate B:** same checkpoint, fixed chart and cycle produce trace words that agree under 2x angular sampling except in explicitly flagged near-collision cases. Report exclusions by task and feature type.
- **Gate C:** out-of-family causal prediction exceeds a matched non-topological baseline, with uncertainty. State failure plainly if it doesn't.
- **Gate D:** true surface-embeddedness and a surface-specific distinguishing invariant are established before invoking Miller-style non-isotopy. **No SCM-01 result reaches Gate D.**

This track belongs in Compactification-Theory as a mathematical instrument; RMP should import its immutable evaluator only after Gate A. An exploratory PR can link it without changing the RMP benchmark.

## Sources

- Hayden, Kim, Miller, Park, Sundberg, *Seifert surfaces in the 4-ball*, J. Eur. Math. Soc. 2025, https://doi.org/10.4171/jems/1703.
- Kim, Miller, Yoo, *Non-split alternating links bound unique Seifert surfaces in the 4-ball*, Compositio Mathematica 2025, https://www.cambridge.org/core/journals/compositio-mathematica/article/abs/nonsplit-alternating-links-bound-unique-seifert-surfaces-in-the-4ball/04B4F087F03E85E4409B19C10E1BB28F.
- Reduced Burau/Alexander formula, https://www.mit.edu/~anser/files/burau.pdf (Def. 3.3 / Thm. 3.4), and https://academic.oup.com/imrn/article/2024/10/8618/7525791.