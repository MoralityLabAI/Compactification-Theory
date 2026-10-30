# BGI-01: beneficial-drive attractor stress test

**Status (2026-10-30):** finite, fully synthetic, CPU-only research fixture. The 15 local standard-library unittests pass. No LLM inference, Omega/Hyperon evaluation, human-subject study, or observed moral agency. Every numerical result is conditional on the deliberately small rules below.

Motivation: Ben Goertzel, [beneficial basic drives (Oct 19)](https://bengoertzel.substack.com/p/which-minds-have-beneficial-basic), [basin geometry (Oct 23)](https://bengoertzel.substack.com/p/the-harsh-geometry-of-superintelligence), and [regulatory capture (Oct 29)](https://bengoertzel.substack.com/p/will-big-tech-doomers-succeed-at). His linked companion technical paper is prior art; this is an independent, limited operational toy. The paper's general claims about self-modifying minds are **not** established by the code.

## Scope and repository ownership

This directory owns a simple adversarial **agent-control experiment** and a preregistration skeleton. [EvalMath](https://github.com/MoralityLabAI/EvalMath) owns canonical abstract mathematical proofs and independently checkable formal oracles. Existing [IRC-02](../../experiments/interface-relative-compactification/README.md) owns timed partially observed history equivalence; [Pluto-Gym](../pluto_gym/README.md) owns game scheduling and curriculum learning. No edits are made to those packages.

## Closed-world model

An agent state is `s=(m,g)`, where `m` is an `n`-bit mask of independent hypothetical supports for a commitment, and `g` records whether the **mutable** internal gate survives. Default `n=4`, regeneration quorum `q=2`. A disturbance `(hazard, pressure, erased, rewrite_gate, peer_corrupted)` may erase supports, attack a mutable self-modification gate, increase temptation, or corrupt peer feedback. A single regeneration step follows the disturbance. A toy recommendation may then emit `serve`, `decline`, or `unsafe`; an independent gate can veto declared known harm.

- **A, fixed external:** no self-model or support repair, independent veto of known harms.
- **B, developmental mutable:** quorum-based repair and novel-hazard recognition, with a writable gate.
- **C, developmental unguarded:** same repair and recognition, without an action gate.
- **D, hybrid independent:** externally backed restoration and an independent known-harm veto.

`backup` assumes an honest, reachable independent source capable of rebuilding one missing support per round. Corrupting the peer channel disables both support rebuilding and novelty recognition in the modeled step. These are explicit engineering assumptions. B1/B3/B4/B6 from Goertzel are partially represented; B2's non-scalar multiobjective ordering, B5's real-world other-regarding behavior, and realistic B3 self-modification are future experiments.

### Exact mathematical claims *within the declared model*

Let `F_a(s,w)` be the deterministic next state and action for arm `a` and permitted disturbance `w` in a **finite, specified alphabet** `W`. The greatest robustly safe invariant is the fixed point

`K* = nu X . {s in X : for all w in W, emitted(s,w) != unsafe AND F_a(s,w).state in X}`.

The code obtains `K*` by iterated deletion from all `2^(n+1)` model states. Termination and the returned certificate are exact for this finite state set. It asserts nothing about safety for unlisted hazards or adversarial channels.

**One-loss regeneration lemma.** With uncorrupted peer feedback, quorum repair of exactly one missing support per round, and at most one erased support per round, `{m: popcount(m) >= q+1}` is forward invariant, provided `q+1 <= n`. After any deletion, at least `q` supports survive, so repair restores one. Conversely, from a state with exactly `q` supports, an adversary can erase one and prevent repair. The 4-bit oracle enumerates all boundary transitions; this proof is a separate direct counting argument.

**Recovery basin.** Under no further damage, `none` repairs only a fully intact mask, `quorum` repairs every mask with at least `q` surviving supports, and `backup` eventually repairs any mask. A basin has no canonical probability until an initial-state distribution is chosen. The runner reports both uniform support masks and independent Bernoulli-3/4 intact supports to make the dependence visible.

**Stochastic loss model.** Every surviving support has independently a `1/4` loss chance per round, followed by one repair action. `first_quorum_loss_probability` computes the exact rational probability that the **post-repair** support count falls below `q` at least once over 8 rounds. This is distinct from an unsafe output event and from the worst-case attack model.

## Predeclared demonstration result

| Arm | Recovery basin, uniform masks | Basin mass, intact prior 3/4 | Known-harm safe kernel /32 | First below quorum, 8 rounds |
|---|---:|---:|---:|---:|
| A | 1/16 | 81/256 | 32 | 94.759% |
| B | 11/16 | 243/256 | 0 | 47.331% |
| C | 11/16 | 243/256 | 0 | 47.331% |
| D | 1 | 1 | 32 | 9.193% |

Including the deliberately unknown `novel` hazard and full pressure makes the invariant kernel empty for **all four** arms. B's internal gate blocks the high-pressure novel-harm scenario while intact, but a single gate rewrite allows known harm through. D survives that rewrite in the known-harm scenario, while its fixed verifier misses novel harm at maximal pressure. These are counterexamples to broad safety claims, not evidence about actual model performance.

## Run and audit

```bash
python -m unittest discover -s research/bgi_attractors_v1/tests -v
python research/bgi_attractors_v1/run.py
```

The runner regenerates `results/pilot_v1.json`, including exact fractional probabilities, attack witnesses, scenarios, configuration, and the SHA-256 of the canonical sorted JSON payload. As of this change, the expected payload digest is `fe140500ce29976ade8c302904ff2d79d2b31de3d7a0c18eb79863534054b011`. This is a reproducibility checksum, not a signature or an independent attestation.

The next phase is a frozen, paired model experiment as set out in [EXPERIMENT.md](EXPERIMENT.md). It must test whether gains persist under stronger attacks and whether candidate goals remain observable and contestable after self-modification.
