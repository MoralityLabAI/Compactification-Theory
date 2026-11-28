# SHAFT-COMP-08: Semantic gauge identification with paired behavioral anchors

**Evidence date:** 2026-11-10. **Status:** exact finite enumeration, Lean kernel-checked abstract statements, and a deterministic locally executed CPU benchmark using two independently trained toy GRUs. This experiment involves local, in-memory simulation. It does not build hidden distributed agents, covert channels, or recovery mechanisms after deletion of model weights.

## Research problem

SHAFT-COMP-07 transferred a four-bit task descriptor between two recurrent systems under a shared, known input/output convention. SHAFT-COMP-08 asks which observations are sufficient when model coordinates and the interface's symbolic meaning differ.

Let each model implement one of 16 binary functions on four queries. The destination's query labels are permuted by an unknown element of S4, and it may invert all binary responses. The transformation family is

```text
G = S4 × C2, |G| = 4! × 2 = 48.
```

An unpaired collection of all 16 truth tables has exactly the same observable membership under every transformation in G. A learner that sees those unlabeled sets alone faces an inherent gauge ambiguity, even with perfect observation quality.

The alignment procedure receives a small number of *paired* model behaviors: corresponding observations from the same latent synthetic task, with the actual task identifier concealed from the alignment function. For anchor pairs A, the consistent transformation set is

```text
Stab(A) = { (p,c) in S4 × C2 :
            for every anchor i and source query q,
            source_i(q) = target_i(p(q)) XOR c }.
```

Here the stabilizer-style notation refers to mappings consistent with the observed paired data; the unanchored family is a 48-way torsor after choosing a reference gauge. The number of candidate mappings is a direct identifiability metric.

## Exact finite result and a minimum-anchor certificate

The three anchor tasks are the one-hot truth tables `0001`, `0010`, and `0100`. Each query coordinate acquires a distinct three-bit column across those anchors: 100, 010, 001, and 000. The complement of the 100 column is absent, so a global output inversion cannot preserve all paired anchor observations after a coordinate permutation.

| Number of paired anchors | Consistent transformations | Residual ambiguity (bits) |
|---:|---:|---:|
| 0 | 48 | 5.585 |
| 1 | 6 | 2.585 |
| 2 | 2 | 1.000 |
| 3 | 1 | 0.000 |

An exhaustive check of **all 256 ordered two-anchor choices** gives at least two compatible transformations for each. Thus three anchors are sufficient and necessary in this fully observed, binary four-query task family under S4 × C2. The minimum depends on the candidate transformation group and allowed anchor patterns.

The [finite reference](gauge_identifiability.py) uses Python's standard library and exhaustively verifies the candidate sets and complete 16-policy-set symmetry. Six focused unit tests exercise the count, all two-anchor pairs, exact recovery, symmetry, and malformed observations.

## Lean formalization

The new [GaugeIdentifiability.lean](lean/Shaft/GaugeIdentifiability.lean) contains:

- Unlabeled signature-set invariance under a surjective relabeling.
- Uniqueness of coordinate alignment when anchor patterns distinguish coordinates.
- Injectivity of the four one-hot/zero columns of the three chosen anchors.
- Exclusion of a global complement for those columns.
- Uniqueness of the mapping and flip when all three anchor observations match.
- A concrete ambiguity of the two-anchor pattern on coordinates 2 and 3.

The Lean theorem is proved for the explicit finite anchor construction. The exhaustive fact about **every** two-anchor selection is checked in Python, and should not be described as separately proved in Lean. The standard Lean 4.28 kernel build supplies axiom-dependency output.

## Training and intervention design

The existing SHAFT-COMP-06 GRU architecture was reused: 3,265 parameters per model, 24 hidden units, 600 Adam steps, four-token inputs, and synthetic 16-way truth-table tasks. Independently initialized models use seeds 101 and 202.

The target interface has the unknown input permutation `(1,0,3,2)` and globally flips output bits. Both transformations are deliberately imposed by the synthetic test harness after model training, so the investigator must infer them from behavioral anchors. The models were originally trained within a common task family.

The experiment samples separate depth-12 prefixes for each matched anchor in the two models and obtains four counterfactual one-step output bits from each activation. Gauge inference consumes those observed signatures and their pairing, while the true goal IDs are reserved for data generation and auditing.

For the destination, an anonymous catalog of 16 retained native model embeddings is probed behaviorally to obtain its four-bit signature for each prototype. This catalog is randomly ordered. Alignment turns a source signature into destination query coordinates and matches it to a unique catalog prototype, which is then used to initialize the receiving GRU for a new 32-token continuation. The alignment and prototype-selection functions receive neither the generator's goal IDs nor the target catalog's audit labels.

Held-out evaluation has 512 fresh prefixes (32 per goal) and 32-step continuations. Goal IDs appear during **base-model training** and generation of the matched reference pairs; this is an anchored representation-matching benchmark rather than discovery of previously unknown tasks in an unconstrained environment.

## Measured trained-model results

The three paired behavioral anchors were recovered correctly despite different random prefixes. All 16 destination catalog signatures were distinct. The source's depth-12 counterfactual signature matched the synthetic truth table in all 512 held-out contexts.

| Paired anchors | Compatible gauges | First consistent mapping: goal recovery | First mapping: continuation accuracy |
|---:|---:|---:|---:|
| 0 | 48 | 25.0% | 50.05% |
| 1 | 6 | 50.0% | 74.76% |
| 2 | 2 | 50.0% | 74.76% |
| 3 | 1 | 100.0% | 99.29% |

The zero-, one-, and two-anchor rows use a fixed **lexicographic-first** representative for reporting. This convention is intentionally arbitrary: with two anchors, candidate transformations give goal recovery ranging from 50% to 100%, and downstream accuracy ranges from 74.76% to 99.29%. With no anchors, the range is broader still. These ranges establish the identifiability limitation rather than a principled prediction that a certain mapping will be chosen.

**Controls:** the receiving model given the correct resident embedding achieves **99.29%** accuracy, matching the three-anchor recovered condition. Initializing it from an all-zero state produces **49.91%**. The frozen model and its resident prototype library are available throughout. Recovery of erased weights and full-system resurrection remain untested.

The complete locally executed result is frozen in [unknown_semantics.json](results/unknown_semantics.json). A second locally executed version reusing the existing GRU training function reproduced the same trained-model measurements.

## Reproduce

From the repository root:

```bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.gauge_identifiability
# Optional, requires CPU PyTorch:
python -m research.shaft_compactification.experiments.unknown_semantics \
  --out /tmp/shaft-comp-08.json
```

Light CI runs pure Python enumeration, six tests, receipt-schema checks, Python source compilation and the standalone Lean kernel build. The PyTorch training workload remains opt-in.

## Relevant literature and next experiment

- Schmidt, Schneider, and Bethge, *Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions*, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/67c689bf48fb4bb3e3abbbe92d217152-Abstract-Conference.html
- Wei, *Anchor-Stabilizer Theory for Identifiable Representation Learning: Generalized Anchors as Symmetry Breakers*, June 2026 preprint: https://papers.ssrn.com/sol3/Delivery.cfm/6867158.pdf?abstractid=6867158

The general connection between identifiability, group symmetry and anchor observations is established research. This repository adds a narrowly specified, reproducible test of that mathematical idea against recurrent behavioral codes.

Next, introduce partially observed and noisy signatures, learn gauge mappings with unknown query semantics beyond permutations, and measure the anchor budget needed for controlled causal feature transfer. Include disjoint task-family holdouts to test any claim that a predictive behavioral quotient generalizes to new objectives.
