# RRC-09: withheld-primitive composition pilot

**Status:** executed CPU pilot, 2026-11-09. One frozen 900-example training roster; 90 development cases and 216 held-out cases; exact test-time coefficient verification. The direct neural model does not learn any successful `C/D` proposal. Execution-guided selection succeeds because the public interpreter supplies those primitives.

## Measured held-out results

| Method | Verified prefixes /216 | Future-correct /216 | Mean primitive calls | Nominal coefficient work / case |
|---|---:|---:|---:|---:|
| Exhaustive bounded exact search | **192** | **184** | **31.981** | **776.153** |
| Direct recurrent GRU, one scan | 32 | 32 | 1.116 | 57.088 |
| Direct recurrent GRU, two scans | 18 | 18 | 1.144 | 68.875 |
| Direct recurrent GRU, four scans | 18 | 18 | 1.153 | 69.755 |
| Direct feedforward | 36 | 36 | 1.310 | 96.671 |
| Shuffled-label GRU | 5 | 5 | 1.000 | 19.000 |
| Learned guided ranker, one full check | 178 | 171 | 56.718 | 1,275.713 |
| Learned guided ranker, four full checks | **192** | **184** | 57.380 | 1,376.111 |
| Untrained preview heuristic, one full check | 178 | 171 | 56.718 | 1,275.616 |
| Untrained preview heuristic, four full checks | **192** | **184** | 57.380 | 1,384.176 |

The 24 abstentions of the exact and four-check guided methods are all from selected out-of-grammar cases. Eight of the 192 accepted prefixes are deliberately planted late mutants and lead to incorrect future predictions. This is the unavoidable paired-prefix ambiguity on the registered roster. The exact-search and guided methods correctly predict all 48 `C`, 48 `D`, and 32 `CD` source programs through degree 48. Both direct neural models verify **0/128** programs involving the missing primitives. The recurrent model verifies 32/48 on the familiar held-out `N/S/R` graphs, versus 36/48 for the feedforward control.

The direct models have close parameter counts: 1,692 for recurrent, 1,695 for feedforward. Each receives the same 38 transformed coefficient values. The recurrent model is trained at one complete scan; applying two or four frozen-weight scans reduces its exact verified-program success. Training-program label accuracy on the 90 development cases is recorded separately in the frozen checkpoints and is higher for feedforward. The ranker's six trained FP64 weights implement a generic prefix-match scoring rule. Its matching held-out verification rate with the **untrained** preview heuristic gives no demonstrated learning advantage in tool-guided search.

## What the negative and positive findings establish

The categorical neural decoder has no supervised labels for `C` or `D`, and its direct predictions fail to recover any unseen-primitive program. Additional recurrent scans do not repair the deficit. This is an intentionally difficult zero-shot semantics condition and should not be generalized to richer models, pretrained mathematical knowledge, in-context primitive descriptions, or larger training budgets.

The guided learner and the untrained heuristic use the test-time public implementations of `C/D`, including convolution and Euler inversion. Their success is **tool-assisted bounded program synthesis**. They inspect 31 candidates at degree six before each case's exact prefix verification. At budget four they reach the exact solver's coverage but require roughly 77% more nominal coefficient-loop operations and nearly twice the counted primitive applications. This workload does not establish a recurrent or neural compression advantage; the explicit interpreter and the target output prefix provide most of the information.

Every held-out `C/D` example has a valid executable program of length at most two in the supplied DSL. The finite grammar uses 31 candidate syntax programs, while syntax aliases are accepted if they execute to the same observed series. The 24 negative cases were selected after ruling out all 31 programs on the observed prefix. The exact solver is complete only under this chosen finite program class. Observed I/O pairs are disjoint from training, while *source-only* input series sometimes recur with different output operators. The measured aliases are 20 development; 10 seen-graph; 11 C; 13 D; 10 CD; eight out-of-grammar; and two late-collision cases. The main 128 unseen-operation cases therefore include 34 reused input source prefixes, with new target operators. The receipt records both paired-input and source-only aliases. The experiment makes an operator-transfer claim, with no claim of unseen source-generator families or sequence-wide blind independence from older RRC datasets.

## Costs and reproducibility

Per-case interpreter-call counts and nominal coefficient-loop work include six-degree previews and full-prefix candidate verification. The work estimate omits coefficient integer bit complexity, Python overhead, target generation, model parameter evaluation, and benchmark scoring; it is a declared operation proxy rather than measured elapsed time. The direct decoder's recurrent hidden state stores 20 FP32 numbers (640 bare bits). Fixed-length two-token DSL syntax occupies six bits, excluding the public interpreter, pretrained model, target data, and verifier. Checkpoint bytes and live memory are separate costs.

`bench09.py --check` regenerates the entire source- and checkpoint-hashed JSON receipt with standard-library Python. Seventeen independent tests cover formal-series mathematics, brute-force partitions, model/interpreter separation, family leakage, late collisions, checkpoint locks and optional PyTorch-vs-portable inference parity. Exact artifact reproduction from the CPU trainer has been checked separately. The tests and full CI are an evidence gate, not a proof of any infinite new partition identity.

## Next hypothesis

**RRC-10:** supply *semantics of unseen primitives as small input-output demonstrations* to a recurrent program inducer, while hiding their source code and direct application at test time. Compare explicit compositional generalization against matched nonrecurrent few-shot models, bounded synthesis that uses the same demonstrations, and tool-augmented interpreters. Reserve an unseen signature family and repeat the source split under new seeds before interpreting any learned advantage.
