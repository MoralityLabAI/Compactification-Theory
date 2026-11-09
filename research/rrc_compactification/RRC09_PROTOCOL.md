# RRC-09: withheld-primitive program-composition protocol

**Execution date:** 2026-11-09. This is one exploratory CPU run with a frozen deterministic roster and source-bound replay. The document and tests were finalized during the implementation. It is not an externally preregistered experiment or a novel formal theorem.

## Question and observable data

The task is **I/O program identification**. The model observes 19 input coefficients \(f_0\ldots f_{18}\) and 19 corresponding output coefficients \(g_0\ldots g_{18}\) from an unknown formal-series operator program. It must propose a program with at most two operations, executed by a public interpreter and checked for exact agreement. The observed output examples are extra supervision compared with RRC-08's unary conversion of a raw source series into its Euler exponents. They are never hidden from the exact search.

The operation set is `N`: sign twist, `S`: substitute \(q\mapsto q^2\), `R`: formal reciprocal, `C`: convolution square, and `D`: Euler exponent/divisor inversion. The direct neural classifier is trained only on `N/S/R` programs, with `C/D` unseen in all train and development labels and inputs. At test time, the exact solver and guided ranker receive executable `C/D` definitions and may evaluate them. This asymmetry is the test's substantive distinction: **direct zero-shot program recovery** versus **tool-augmented rule selection**.

## Fixed rosters

Random source series are periodic-product partition generating functions with multiplicity cap \(\in\{\infty,1,2\}\), moduli sampled from \(\{3,4,5,6,7,8,9,10,11,12,13,17\}\), and independently sampled nonempty residue masks. Sources are nonconstant in early degrees. All source-series coefficients are generated exactly through degree 48. Only degree 0–18 input/output pairs enter student inference and finite validation; degrees 19–48 are held back for future prediction grading.

| Split | Cases | Target operator program |
|---|---:|---|
| Training | 900 | length ≤2 over `N/S/R` |
| Development | 90 | length ≤2 over `N/S/R`, unique I/O prefixes |
| Held-out seen graph | 48 | `N/S/R`, new source inputs |
| Held-out `C` | 48 | one or two operators involving `C`, excluding `D` |
| Held-out `D` | 48 | one or two operators involving `D`, excluding `C` |
| Held-out `CD` | 32 | `C∘D` or `D∘C` |
| Out-of-grammar | 24 | observed-degree mutations selected after exhaustive rejection of all 31 DSL programs |
| Late collisions | 16 | eight exact-prefix base/mutant pairs, one future change per pair |

The 216 held-out items are input/output-prefix distinct from the 900 training examples; each late collision pair deliberately duplicates its own observed input. Some source-only prefixes recur despite distinct output targets: 20/90 development, 10/48 seen-graph, 11/48 C, 13/48 D, 10/32 CD, 8/24 out-of-grammar and 2/16 collision cases. These are case-level alias counts and include intentional collision replication. The source *series* generator is shared across training and testing: these are held-out **operator families**, not held-out underlying partition-family generators. No test labels are used during gradient training or model selection. The benchmark and its single seed are exposed for subsequent research, so future models must use a fresh test roster.

## Models, budgets and verification

- Direct GRU: 20 hidden units, 1,692 FP32 parameters, flattened two-token classifier head. One trained evidence scan of 19 positions; extra frozen-weight scans of 2 and 4 are an untrained depth ablation. Live hidden state: 20 floats, or 640 bare bits.
- Direct feedforward: full 38-element observed pair, 33 hidden tanh units, 1,695 FP32 parameters. Its 0.18% parameter-count difference is explicit. It receives the same 38 real features.
- Shuffled-label GRU: 1,692 parameters and the same training sequence, with a fixed permutation of program targets.
- All direct models: 450 Adam steps, minibatch 96, learning rate 0.005, matching minibatch sequence, one PyTorch CPU thread and frozen checkpoints exported as JSON. Each coefficient is encoded as signed `log1p` divided by 10, clipped at 30 before scaling. Program labels use exact canonical prefix search in the **training-only** grammar.
- Execution-guided linear scorer: six FP64 learned parameters, 360 Adam steps on program-preview features, trained only from `N/S/R` candidate executions. At inference it consumes all 31 six-degree candidate previews, including the public `C/D` interpreter. Top \(k\) candidates are checked through degree 18.
- Nonlearned controls: complete bounded exact search, and an untrained first-fault / residual preview sorter with the same 31 candidate previews and verifier budgets.

All direct proposals must pass the *same* finite integer coefficient verifier before any future prediction receives credit. Programs and logits are not amended using withheld degree 19–48 values. The exact program search provides the class-conditional upper control. For any method, falsely accepted late mutants appear as future prediction errors. Absence of an exact-prefix program is recorded as abstention; a forced wrong proposal that the checker rejects is also recorded as abstention, distinguished from source-program recovery.

## Reproduce and acceptance

Python 3.10+ standard library for all tests and replay. From repository root:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_synth09.py' -v
python research/rrc_compactification/bench09.py --check
```

Independent *optional* checkpoint reproduction requires PyTorch CPU:

```bash
python research/rrc_compactification/train09.py --out-dir /tmp/rrc09_retrain
# Compare /tmp/rrc09_retrain/rrc09_*.json against the frozen results files.
```

CI performs no model training or network downloads. For evidence integrity, the replay requires exact agreement of every score and the stored receipt plus source SHA-256 and checkpoint SHA-256. A source-code or checkpoint change demands an explicit regenerated receipt and fresh review. The local optional PyTorch-portable inference parity test skips cleanly on a standard-library-only runner.

The primary success criterion is **complete exact program recovery on held-out `C/D` compositions from the direct student**, with independent grammar verification and cost-matched recurrence controls. The tool-assisted ranker is separately assessed for *total* search effort, since it can call operations whose semantics were withheld from supervised examples. Neutral or negative results remain reported.
