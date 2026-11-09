# RRC-12: direct recurrent proposal of unknown fork/join graphs

**Status:** single exploratory CPU pilot, November 10, 2026. Written during implementation. The RRC-11 dataset and mathematical vocabulary were already exposed. This experiment uses a new fixed seed and a new, graph-class-disjoint training split. There is no external preregistration or replication on an independent research group's data.

## Primary question

Can a small recurrent encoder propose a complete executable algebraic graph from **two input/output demonstrations**, without evaluating a catalogue of candidate graphs during inference? A public integer interpreter is used strictly after the proposal to check whether it matches all observed coefficients. The primary held-out metric is the number of accepted, degree-13–32 correct graph proposals over an unseen combination family. The matched nonrecurrent baseline gets exactly the same inputs and verification policy.

## Formal-series task

All series have integer coefficients and constant 1. The graph is `post(left(F)*right(F))`; each of the three nodes selects `I` (identity), `N` (sign substitution), `S` (degree doubling), or `R` (formal reciprocal). `*` denotes exact truncated convolution. There are 64 graph strings; syntactic aliases mean the graph map is not injective. Eight fixed, independent integer probe series group the 64 strings into 25 **finite-probe classes**. This grouping is a design convention, audited again with additional probes, not a general algebraic completeness theorem.

The direct encoder receives 26 four-element tuples `(signed_log1p(input_n)/10, signed_log1p(output_n)/10, degree/12, demo_index)` from two demonstration pairs, each observed only through degree 12. The output is three four-way token predictions. The query input and output beyond the observed demonstrations are withheld until scoring. The direct model has no candidate-program executor, hidden source label, Euler transform, residual oracle, or preview results in its inference path.

## Split and contamination boundaries

One deterministic seed `2026111012`. Training comprises **2,040** examples over 17 empirical graph classes whose lexicographically first representative does *not* contain both `S` and `R`; development contains 320 fresh examples from the same classes. All four primitives appear during training. The eight held-out empirical classes contain both `S` and `R` in their canonical representatives and admit no equivalence-class overlap under the registered finite-probe grouping.

| Held-out evaluation | Cases | Generation |
|---|---:|---|
| Familiar classes | 68 | fresh signed-integer series, familiar training graphs |
| Unseen graph combinations | 160 | signed-integer series, eight excluded classes |
| Source-family transfer | 64 | periodic-part product generating functions, eight excluded classes |
| Out of grammar | 20 | one observed output coefficient mutated, retained only after full 64-candidate search rejects the demonstrations |
| Late future collisions | 16 | eight pairs of identical demos/query, with one member's unseen degree-15 target mutated |

No training input prefix is duplicated in development or held-out cases. The negative samples are selected by the full exact search, so the negative-control rejection rate is **constructed**, and should not be read as a population OOD detection rate. The source distribution of trained examples shares `graph11.source` with an earlier RRC research track; the new random seed and held-out graph-class condition do not imply full research-sequence blinding.

## Architectures and budgets

- **GRU:** one layer, input 4, hidden 32, three four-way output heads; **4,044** FP32 parameters, 1,024 bare bits of live state. Train for 420 Adam steps, learning rate 0.004, batch 128, single CPU thread, teacher-forced graph-token targets. Freeze one-scan weights; re-run them at 2 and 4 scans solely as a depth ablation.
- **Feedforward:** flattened 104-feature input, tanh hidden 34, the same 12 outputs; **3,990** FP32 parameters, 1.35% fewer than GRU. Identical batch schedule and training budget.
- **Shuffled-target GRU:** 4,044 parameters; identical optimizer budget with a single fixed permutation of training graph labels.
- **Untrained public-preview heuristic:** evaluate all 64 candidates through degree four on both demonstrations and try the first four ranked graphs through degree 12; full execution work and primitive calls charged.
- **Exact exhaustive baseline:** compare up to 64 graphs through degree 12 with early first-demonstration failure stopping. This method is complete for the declared 64-string grammar and observed horizon. The blind first-graph comparator uses one trained-grammar candidate.

All candidate graph proposals undergo the same exact post-proposal verification. Accepted programs are then executed on the withheld degree-32 query; every collision false acceptance is retained. Count FP neural multiply-accumulates and **integer coefficient-loop work** separately, with no invented hardware conversion rate. Training data generation, PyTorch optimization, interpreter implementation, coefficient bit complexity and Python overhead are separately acknowledged resources.

## Verification and claims gates

1. Check exact finite algebra, equivalence-class separation on independent probes, source-family isolation, absence of visible train/test aliases, and strict demo-prefix-only feature formation.
2. Freeze the three checkpoints before evaluating the held-out cases. No test metric is used for gradient updates or checkpoint selection.
3. Assert the portable standard-library GRU and feedforward implementations agree with PyTorch on an independently chosen development case.
4. Record exact accepted and correct-future totals on each family, one/two/four scan ablations, blind/shuffled controls and exact/preview comparisons. Direct-model gains require accepted predictions over the feedforward control and transfer to *unseen combinations*.
5. Mutating only future coefficients must leave every model and search proposal unchanged; the registered eight collision pairs bound deterministic future correctness to eight within the paired set.
6. Regenerate the source- and checkpoint-hashed receipt byte-for-byte. Independently retrain all models and compare checkpoint bytes. CI installs only Python, runs model inference and exact verification, and skips optional torch parity cleanly.

## Commands

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_graph12.py' -v
python research/rrc_compactification/bench12.py --check

# Optional CPU checkpoint reproduction with PyTorch installed:
python research/rrc_compactification/train12.py --out-dir /tmp/rrc12_retrain
cmp /tmp/rrc12_retrain/rrc12_recurrent.json research/rrc_compactification/results/rrc12_recurrent.json
```
