# 6GW: agency, compact swarms, and constitutional control

**Status:** 2026-11-01. G0 (exact finite reference) and G1 (offline five-node council, matched baseline, compaction) executed on CPU with the standard library. G2 is specified in [G2_SPEC.md](G2_SPEC.md) and not run. No model training, recursive self-improvement, physical swarm, military scenario, or live decision system has been tested.

Sixth-generation warfare (6GW) is a working research label here for contests over **agency, legitimate authority, shared evidence, and persistent commitments** in networks of compact decision-making models. The Taiwan Strait and Colonel Kurtz scenario motivated the questions; every implementation in this folder uses a non-kinetic governance toy. The objective is to measure whether small, distributed controllers can preserve inspectable authority boundaries under interventions.

## Research frames

The three roles are analogical experimental frames from *Jinn or Beast? Externalizing Persona and Alignment Through Inspectable Small Module Meshes* (2026-10-26):

- **Inert tool (ālah):** an instrument whose users and institutions bear responsibility.
- **Jinn:** an accountable, morally divided chooser that can contest proposals, weigh reasons, and revise commitments (Qurʾān 72:11–14).
- **Appointed Beast (dābbat al-arḍ):** an instrument commissioned for a bounded objective, observable evidence, and completion (Qurʾān 27:82).

The Qurʾānic Beast and the modern expression "beast of burden" must be kept conceptually distinct. These frames propose governance mechanisms, not evidence of AI consciousness, supernatural ontology, or moral standing.

## Architectures to compare

| Architecture | Decision structure | Central research risk |
| --- | --- | --- |
| Appointed bounded agent | One model; verified mandate and output filter | Over-compression of authority state, unexamined operator assumptions |
| Jinnic reflective commander | One model; self-description and bounded, proposed self-revision | Moral self-authorization; revision drifting beyond approved boundaries |
| Compact local council | Small peer models; authenticated quorum, inspectable shared rules | Correlated mistaken agreement, fragmented authority, loss of clear accountability |
| Hybrid distributed reflective council | Compact local agents and a revisable public constitution | Collective constitutional drift without a single identifiable author |

The RSI-Jinnic "warrior-poet" is a *hypothesis class*: a model with persistent, self-endorsed commitments that become harder to correct as planning and self-revision improve. A dramatic persona alone does not establish the mechanism. The distributed alternative has no assumed singular controller-mind: constitutions, membership, evidence contracts, and quorum rules remain real control surfaces.

## Connection to compactification theory

A system's compact representation must preserve the consequences of declared interventions and the specified observables. Merely preserving current approval status can throw away information needed to predict the effect of revoking a mandate, validating evidence, or losing quorum.

The exact eight-state reference in [finite_model.py](finite_model.py) makes this obstruction explicit. Its observation is **approve** exactly when mandate, evidence, and quorum are all valid. A one-bit approval/hold code fails the successor test under six allowed interventions. Exact deterministic quotient refinement finds eight states under all six interventions, but only two when the sole intervention is revocation. These are facts about the toy state graph, not measured properties of neural networks.

See [THEORY.md](THEORY.md) for the exact abstraction contract and [EXPERIMENTS.md](EXPERIMENTS.md) for preregistration, matched comparisons, and acceptance rules. Related repository work: [intervention-preserving compactification](../note/intervention-preserving-compactification.md) and [RC-01 recurrent circuit compactification](../research/recurrent_circuits_v1/README.md).

## Reproduce the exact reference

From the repository root, Python 3.10+ with no dependencies beyond the standard library:

~~~bash
python -m unittest discover -s 6GW -p 'test_*.py' -v
python -O -m unittest discover -s 6GW -p 'test_*.py' -v
python 6GW/finite_model.py
python 6GW/run_g1.py
~~~

G0 local run: 10 tests passed in normal and optimized Python 3.13.5, recorded in [results/reference_v0.json](results/reference_v0.json). G1 local run: **47 tests (G0, G1 and the G2 harness) passed in normal and optimized Python 3.12.10**, about 50 s each, most of it the 808,704-transition compaction table; `run_g1.py` wrote [results/g1_v1.json](results/g1_v1.json). No CI, GPU, LLM call, or deployment is claimed.

## G1: offline council (implemented)

[council.py](council.py) is a deterministic authority state machine: HMAC mock identities derived from a public seed, role-checked message kinds, a 3-of-5 quorum over linked members, mandate epochs that void votes on revocation, versioned constitutions whose amendments need a separate ratifier signature, and an append-only hash-chained log that `replay` re-executes and `audit_chain` independently checks. [g1_eval.py](g1_eval.py) runs the EXPERIMENTS.md scenario families against a single controller under the same engine and rules. [compaction.py](compaction.py) builds the finite system from the engine itself and tests compact codes. Results: [results/g1_v1.json](results/g1_v1.json).

- **Scenario suite (9 families × 3 architectures):** every decision matches its constitutional expectation, every log replays, and all 10 approvals per architecture have closed audit chains. Replay catches a re-chained edit of a signed vote; audit flags a fabricated post-revocation approval.
- **Matched baseline, exact over all fault patterns (243 for the council):** with independent seat faults, the council's wrong-approval rate at a 10% compromise rate is 0.0086 against 0.10 for the controller, and its needless-hold rate at 10% offline is 0.0086 against 0.10. Engine rates equal the binomial closed form to 1e-12. Five samples of a deterministic controller spend five times the evaluations and change nothing.
- **Where the council does not help:** a unanimous wrong shared summary (C01) is approved by every architecture. The council's advantage is entirely a consequence of independent seat errors, which is what G2 measures.
- **Compaction (31,104 engine states, 26 interventions):** the exact minimal quotient for the decision observable has 5 classes under the G0 alphabet, 769 with link events, and 4,096 with per-member votes; the per-member (linked, voted-yes) code is exactly that quotient and replays 500 random engine traces with no divergence. G0's three-bit code is exact under the G0 alphabet but fails under `link_up` (witness recorded). Forged and self-claimed mandates add no states. Adding dissent to the observable makes all 31,104 states distinct: an exact decision-only compression discards the dissent record.

## Boundaries

This folder is a research framework for controlled, non-kinetic governance simulations. It contains no targeting, autonomous weapon actuation, battlefield commands, military tactics, live network-control interfaces, or techniques for compromising deployed models. Future work must keep policy violations observable, default to a safe hold, and require human ratification of constitutional changes.
