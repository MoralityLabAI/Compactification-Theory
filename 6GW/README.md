# 6GW: agency, compact swarms, and constitutional control

**Status:** research seed, 2026-11-01. Exact finite-state reference executed on CPU; no model training, recursive self-improvement, physical swarm, military scenario, or live decision system has been tested.

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
~~~

Local run: **10 tests passed in normal and optimized Python**, with Python 3.13.5. The generated result is recorded in [results/reference_v0.json](results/reference_v0.json). No CI, GPU, LLM call, or deployment is claimed.

## Boundaries

This folder is a research framework for controlled, non-kinetic governance simulations. It contains no targeting, autonomous weapon actuation, battlefield commands, military tactics, live network-control interfaces, or techniques for compromising deployed models. Future work must keep policy violations observable, default to a safe hold, and require human ratification of constitutional changes.
