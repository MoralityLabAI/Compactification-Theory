# SHAFT-COMP-11: Observational codes versus causal state sufficiency

**Execution date:** 2026-11-10. **Evidence:** Lean 4 theorems over deterministic state machines; exhaustive eight-state counterexample; two locally trained 3,265-parameter synthetic CPU GRUs, with 512 test states per model/depth. The project studies defensive monitoring and scientific representation adequacy in isolated memory. It contains no unauthorized device behavior, covert persistence mechanism, or erased-weight reconstruction.

## Research question

A probe may accurately compress a neural state into the behavioral outputs of four counterfactual queries. Does equality of those four bits imply equality of the model's outputs after arbitrary future computations?

The premise resembles the question of whether a compact predictive state is a **dynamical factor** of the full recurrent system. The earlier SHAFT-COMP-07 code was obtained by distilling four one-step output bits from a frozen GRU. Here we test whether it is causally sufficient.

Let a transition system have state H, inputs a in Σ, transitions F_a:H→H, code π:H→C and observable O:H→Y. A sufficient exact quotient requires a transition congruence:

  π(h)=π(h')  ==>  π(F_a(h))=π(F_a(h')) for every relevant a,h,h'.

Outputs must factor through the code: O(h)=o(π(h)). If a restoration map R:C→H is a section on reachable codes, π(R(π(h)))=π(h), then for every finite input sequence w,

  **O(F_w(R(π(h)))) = O(F_w(h)).**

The theorem is a consequence of induction on input words. It is formalized in [CausalSufficiency.lean](lean/Shaft/CausalSufficiency.lean), alongside an impossibility lemma: if two states share a code but have different observations after a continuation, a single state reconstruction from that code cannot reproduce both.

The theorem concerns the *same specified transition system*. Replacing the underlying model weights demands a separate semantic intertwiner between the old and new substrates.

## Exact counterexample

Use an eight-state, three-bit delay line H={0,1}³. Its observation code π(b₀,b₁,b₂)=b₀. An activated step shifts bits left, filling the final position with the previous b₂:

  F(b₀,b₁,b₂)=(b₁,b₂,b₂).

States (0,0,0) and (0,0,1) have equal present codes and equal outputs after one step. They differ after two steps. The standard-library [causal_finite.py](causal_finite.py) tests every state and all 64 deterministic one-bit-code-to-state decoders; none reproduces all two-step outputs. A full three-bit state code trivially suffices on the same finite model.

The Lean counterexample proves the two-step distinction by exact Boolean computation. The enumeration of all 64 short decoders is separately checked in Python.

## Experiment A: orthogonal neural-state interventions

The two frozen GRUs reuse the SHAFT-COMP-06 architecture and training protocol: 24-dimensional hidden state, synthetic 16-way four-bit lookup task, and independent seeds 101/202. Each linear probe has four outputs (24×4 weights plus bias), fitted on the network's **own** four counterfactual response bits from recurrent depths 4, 8 and 12. The probe fit does not receive explicit goal labels; the base network was originally trained with those labels.

Take the 4×24 probe matrix W and an SVD of its right singular vectors. In each test condition, apply fixed-L2 perturbations in three arms:

1. **Nullspace:** project Gaussian vectors into ker(W), normalize to radius r.
2. **Rowspace:** project into row(W), normalize to the same radius r.
3. **Random direction:** an unprojected Gaussian vector normalized to radius r.

All arms use the same frozen hidden states, random-vector source and future input sequences. The code is the thresholded output of the linear probe. Since Wδ=0 in the nullspace arm, its **continuous probe logits** should be unchanged to numerical rounding, even though the hidden state changes.

The model itself supplies the ground-truth *counterfactual four-query behavior*, measured after the same input continuation of length k∈{0,1,4,12}. States originate from depths 12 or 40; radii are 0.75, 1.5 and 3.0. Each depth uses 512 fresh states per trained model.

### Nullspace arm, radius 1.5

| Model | Starting depth | Probe code retained | Immediate future response changed | Response changed after 12 steps |
|---|---:|---:|---:|---:|
| 101 | 12 | 512/512 | 31/512 (6.05%) | 56/512 (10.94%) |
| 202 | 12 | 512/512 | 17/512 (3.32%) | 44/512 (8.59%) |
| 101 | 40 | 512/512 | 180/512 (35.16%) | 115/512 (22.46%) |
| 202 | 40 | 512/512 | 132/512 (25.78%) | 90/512 (17.58%) |

**Response changed** means at least one of four binary counterfactual answers differs after the indicated number of common input steps. It is *not* the task-accuracy difference, nor a claim that the difference first appeared at that time.

The four-bit probe output stayed identical on every nullspace example, while the largest change in any real-valued probe logit at radius 1.5 across these conditions was 5.72×10⁻⁶. The model's recurrent transition need not preserve the nullspace of an independently fitted linear probe, so the later trajectories can diverge.

At radius 0.75, depth-12 nullspace interventions already changed the four-query signature after 12 steps in 25/512 (model 101) and 12/512 (model 202) cases. At radius 3.0, the corresponding counts were 155/512 and 121/512. These interventions may create states outside the naturally reachable hidden-state manifold. This makes the next control decisive.

## Experiment B: naturally reached state pairs

Construct two recurrent hidden states for **the same** synthetic task using independent random input prefixes. Pair states by task and feed each the *same* fresh future token sequence. There is **no artificial hidden-state perturbation**. The probe, base weights and task are held fixed.

At starting depth 12, all 512 pairs in each model have identical four-bit probe codes and identical immediate four-query response signatures. Later:

| Future continuation | Model 101: different signatures | Model 202: different signatures |
|---|---:|---:|
| 0 | 0/512 | 0/512 |
| 12 | 13/512 | 11/512 |
| 32 | 39/512 | 37/512 |

This directly falsifies an exact *all-future* sufficiency claim for the four-bit code within the synthetic reachable-state sample. The two states can be behaviorally equal on the immediate four-query test and differ after a shared later continuation.

At depth 40, equal predicted codes occur in 376/512 pairs (seed 101) and 410/512 (seed 202). Even among those matched-code pairs, immediate signature disagreement occurs in 53/376 and 81/410 respectively, reflecting the probe's deterioration beyond its training horizon.

The finite observed disagreements suffice to disprove universal sufficiency for these fitted codes on these models, provided the states were generated correctly. They do not determine a minimal adequate code size, prove any hidden malicious objective, or show that a particular behavior persists after model deletion.

## Precision and interpretation

The code's information content is conditional on retaining the trained probe and model substrate. A four-bit *output alphabet* does not imply that the representation learner itself consists of four stored bits: the probe contains 100 learned parameters and the frozen GRU retains 3,265 parameters.

Nullspace interventions are causal interventions, yet some lie off-manifold. The natural-pair test addresses that concern by using independent reachable states. Results are finite-sample proportions on one fixed task family and two seeds, with no confidence claim about frontier models.

The mechanism is a failure of **forward invariance of the probe kernel**. In a local linearization with recurrent Jacobian J, an initially unobserved displacement δ satisfying Wδ=0 can become observable after k steps when WJᵏδ≠0. Its finite-horizon unobservable subspace is

  N_L = ⋂_{k=0}^L ker(WJᵏ).

For a linear time-invariant recurrent approximation, N_L is also the nullspace of the observability Gramian

  Q_L = Σ_{k=0}^L (Jᵏ)ᵀ WᵀWJᵏ.

This is classical system-observability mathematics. The neural GRU experiment tests the nonlinear behavior directly, rather than claiming to estimate a global J or certify this local approximation.

## Artifacts and reproduction

- [causal_probe.py](experiments/causal_probe.py): deterministic fixed-norm nullspace/rowspace/random intervention arms, exact probe-output preservation and delayed four-query agreement.
- [natural_alias.py](experiments/natural_alias.py): paired independent reachable states from each synthetic task.
- [causal_finite.py](causal_finite.py): exhaustive eight-state delayed-bit example.
- [CausalSufficiency.lean](lean/Shaft/CausalSufficiency.lean): transition congruence, sufficient restoration theorem and impossibility of universal decoding after a same-code observational split.
- [Frozen compact summary](results/comp11_summary.json): the 12 nullspace settings, six natural-state conditions, and SHA-256 hashes of full local receipts.
- The complete full-resolution JSON outputs are preserved as a conversation artifact; lightweight CI checks the compact summary and executes the standard-library finite test.

Execute from the repository root with torch 2.10.0+cpu:

```bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.causal_finite
python -m research.shaft_compactification.experiments.causal_probe \
  --out /tmp/shaft-comp11-nullspace.json
python -m research.shaft_compactification.experiments.natural_alias \
  --out /tmp/shaft-comp11-natural.json
cd research/shaft_compactification/lean && lake build
```

The neural runs are optional and intentionally excluded from the rapid CI job. Separate validation would be needed for other hardware/BLAS versions and more model seeds.

## Mathematical continuation

The strongest next research direction is to estimate the *finite-horizon causal observability dimension* of recurrent states and compare it with the four-bit contemporaneous behavioral dimension. Any candidate compression theorem must specify the input family, observation horizon, intervention class and how closely its quotient respects recurrent transitions.
