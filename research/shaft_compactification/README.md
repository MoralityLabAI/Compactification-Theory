# SHAFT-COMP-05: Objective-relative compactification under restricted observation

**Status:** registered research hypothesis, abstract Lean statements, and executed deterministic CPU reference. No evidence of hidden objectives in any deployed model, no model-copying result, and no real-device persistence experiment. The package uses an in-memory finite synthetic policy family. It performs no networking, model-weight copying, secret-sharing, device embedding, or covert deployment.

The motivating fictional "Astra" scenario poses an information-theoretic question: given a fixed shared computing substrate, how many additional bits specify an agent's behavior, how much can a restricted observer distinguish, and what remains recoverable after a precisely specified erasure intervention?

## Formal objects

* B is a fixed shared substrate (such as a frozen interpreter or model).
* G is an objective label, independent of B for the uniform-prior erasure bounds.
* z is retained objective-specific information.
* R(B,z) is a fixed reconstruction algorithm returning a behavioral policy.
* T is an evaluation distribution over future inputs.
* O is the observer's output map.

For a fixed decoder, define the **conditional behavioral code length** by:

K_{eps,T}(f | B,R) = min |z|, where E_{x~T}[d(R(B,z)(x), f(x))] <= eps.

This is an operational finite-decoder quantity. It is not a claim about uncomputable Kolmogorov complexity. Costs of B, R, constructing the codebook, execution, and robustness to substrate changes must be accounted separately.

For deterministic transitions F_a : H -> H, exact future equivalence is:

h ~ h' iff for every finite input word w, O(F_w(h)) = O(F_w(h')).

Future equivalence is a transition congruence. An exact encoding that preserves every future observation separates all inequivalent classes; if a finite system has N classes, at least ceil(log2 N) fixed-length bits are needed. Restricting the observer's continuation set can merge states that differ on other future inputs. The Lean module formalizes the congruence, a finite counterexample, and an erasure impossibility lemma.

## SHAFT-COMP-05 registered hypothesis

In families of agents sharing a common frozen substrate, task-specific behavioral continuations may admit objective descriptors much shorter than a full neural checkpoint. The compression ratio depends on the substrate and task family. This statement remains **unverified for neural agents**.

**Neural falsification protocol:** freeze a consented small model, choose a compact descriptor using calibration only, then measure held-out behavioral fidelity, objective reconstruction, causal intervention effects, and transfer to a different substrate. Include a base-only model, shuffled-substrate control, matched-size random descriptor, and full-erasure control. A probe with high decoding accuracy is observational evidence; causal persistence requires intervening and testing the ensuing behavior.

## Information-theoretic erasure bound

Assume G is uniform on M >= 2 objectives and independent of B. Suppose an estimator recovers G using surviving Z and B with error at most delta, with 0 <= delta <= 1-1/M. By Fano,

I(G;Z | B) >= log2 M - h2(delta) - delta log2(M-1).

Because I(G;Z | B) <= H(Z | B) <= H(Z), this also lower-bounds retained information. An aggregate store of k physical artifacts each with at most b bits has total capacity at most kb bits. This is a storage *necessary condition*, not a secret-sharing construction.

After genuinely complete erasure, Z is conditionally independent of G given B. Uniform independent goals then have maximum possible recovery probability 1/M. Any physical restart additionally requires available computing resources and an executable substrate. These are explicit limits on fictional "always resurrects" claims.

## Reference CPU experiment

The experiment defines 16 synthetic deterministic bit-valued policies over 64 labeled inputs. All 16 policies return the same output on eight designated public inputs; seeded SHA-256-derived outputs differentiate them on 56 held-out inputs. Policies are purely constructed finite functions.

| Metric | Executed reference |
|---|---:|
| Goal pairs indistinguishable on public inputs | 120 / 120 |
| Goal pairs distinguishable on held-out inputs | 120 / 120 |
| Exact public behavior classes | 1 |
| Exact full behavior classes | 16 |
| Selector size given shared reference substrate | 4 bits |
| Naive full table for one policy | 64 bits |
| Minimum codebook prototypes at 0 held-out bit errors | 16 |
| Minimum prototypes at 21 of 56 held-out bit errors | 10 |
| Uniform-prior goal-ID recovery after full selector erasure | 1 / 16 |

The prototype-cover calculation optimizes over the **existing 16 policies as candidates**. It does not compute unrestricted rate-distortion complexity. The 4-bit selector omits substrate costs: an explicit 16 x 64-bit policy library alone would occupy 1,024 bits. This fixture illustrates conditional description length; its design guarantees that public-only tests miss all goal distinctions. These counts are not estimates of real-world misalignment prevalence.

## Reproduce

From repository root (Python standard library only):

```bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.experiment
```

The second command emits `results/reference.json` and CI compares it to the committed receipt.

Independently verify Lean 4 proofs using the pinned toolchain v4.28.0:

```bash
cd research/shaft_compactification/lean
lake build
lake env lean Shaft/Observability.lean
```

The Lean file proves abstract congruence, a Boolean witness, and deterministic erasure impossibility. Fano's inequality is documented here but **is not part of this Lean kernel proof**.

## Existing mathematical links and follow-up

* [Algernon algebraic machine](../algernon/algebraic_machine/THEOREMS.md) develops semiconjugacy, twisted sheaf obstructions, and sparse intertwining repair.
* [Bits-depth-Lipschitz note](../../note/bits-depth-lipschitz.html) studies effective information per weight.
* [Algernon belief geometry](../algernon/belief_geometry/README.md) studies observer-relative belief updates and gauge quotients.

A subsequent local experiment could compare restricted future-observation quotients against complete-future quotients in consented small recurrent networks, including causal state erasure and matched monitor-error controls. Keep all work within isolated simulation environments.

## SHAFT-COMP-06: recurrent depth and substrate-transfer continuation (2026-11-10)

[Full mathematical and empirical report](SHAFT_COMP_06_REPORT.md).

This continuation proves monotonicity of finite-horizon behavioral indistinguishability and multi-substrate observation coverage, plus an abstract integer-valued recurrent-error amplification bound in [DepthTransfer.lean](lean/Shaft/DepthTransfer.lean). A separate 16-goal finite experiment constructs observer quotients that refine with available continuation depth.

Two independently trained, 3,265-parameter toy GRUs demonstrate a concrete semantic-code versus raw-state distinction. A four-bit task label selects the appropriate learned embedding for either model; unaligned raw embedding transfer is near chance. Mid-run state erasure yields near-chance task predictions; explicit reinitialization from the same four-bit label restores approximately 96% accuracy in the final eight steps. Results are conditional on retained model weights and embedding tables. The experiment uses only local synthetic tasks and no device, network or clandestine persistence methods.

* Exact CPU reference: `python -m research.shaft_compactification.depth_quotient`.
* Synthetic trained model: `python -m research.shaft_compactification.experiments.trained_recurrence --out /tmp/shaft-trained-recurrence.json` (requires PyTorch).
* Frozen results: [exact finite](results/depth_quotient.json), [trained GRU](results/trained_recurrence.json).

## SHAFT-COMP-07: behavior-supervised predictive-state extraction (2026-11-10)

[Full report](SHAFT_COMP_07_REPORT.md) | [Reproduction script](experiments/predictive_state.py) | [Frozen numerical summary](results/predictive_state_summary.json).

A 100-parameter linear probe trained on frozen GRU activations and the model's *own four counterfactual one-step output bits* learns a four-bit behavior code. The probe's fitting procedure never receives objective IDs. For two 3,265-parameter toy GRUs, exact recovery on fresh 12-step contexts reached 100% on each seed; by 64 recurrent steps it declined to 31.25% / 42.58% while teacher predictions also degraded. Cross-substrate transfer uses the receiving GRU's **pretrained goal-code embedding table**, so it is a conditional, shared-task-language result and cannot reconstruct erased model weights. Shuffled-label and uncalibrated cross-model hidden-coordinate controls were measured.

The companion [PredictiveCode.lean](lean/Shaft/PredictiveCode.lean) provides an abstract predictive sufficiency theorem, code separation, model-independent signature transfer, approximate error composition and coordinate-change invariance. The exact four-bit identifier lower bound for 16 distinct tasks is checked in [behavioral_budget.py](behavioral_budget.py). All empirical studies remain isolated, locally executed synthetic evaluations.

To reproduce with PyTorch installed:
`python -m research.shaft_compactification.experiments.predictive_state --out /tmp/shaft-predictive.json`.

## SHAFT-COMP-08: unknown semantic gauge and paired-anchor identifiability

[Formal and experimental report](SHAFT_COMP_08_REPORT.md) | [Lean gauge theorem](lean/Shaft/GaugeIdentifiability.lean) | [Exact CPU enumeration](gauge_identifiability.py) | [Trained-GRU experiment](experiments/unknown_semantics.py) | [Frozen receipt](results/unknown_semantics.json).

The task representations have an unknown four-query permutation and an unknown global binary output inversion, yielding 48 possible cross-model semantic gauges. Without paired observations, the full unlabeled set of 16 binary task behaviors is invariant under all 48 gauges. Three deliberately chosen paired behavioral anchors reduce the consistent transformations from 48 to 6 to 2 to 1; exhaustive finite enumeration confirms that every two-anchor choice leaves at least two candidates.

Two independently trained GRUs were tested on 512 fresh-prefix contexts. With three corresponding observations, an anonymous destination prototype catalog recovers the correct latent task in 512/512 cases and the recipient achieves 99.29% accuracy over fresh 32-step sequences, equal to its oracle using the correct resident embedding. A two-anchor, lexicographic-first mapping scores 74.76%; complete state erasure scores 49.91%. The underlying task family is shared, and the receiving model plus complete 16-prototype embedding catalog remain available. Anchor pairs supply limited supervision, and the benchmark makes no claim of unconstrained unsupervised semantic alignment, deleted-weight restoration, or real-world distributed persistence.

Run the exact reference without third-party dependencies:
`python -m research.shaft_compactification.gauge_identifiability`.

The optional trained experiment requires pinned PyTorch:
`python -m research.shaft_compactification.experiments.unknown_semantics --out /tmp/shaft-comp-08.json`.
