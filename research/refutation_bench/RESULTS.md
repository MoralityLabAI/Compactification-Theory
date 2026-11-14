# RB-001: offline harness controls

Executed on 12 November 2026. The full deterministic record is
[baseline_receipt.json](results/baseline_receipt.json), including per-case
outcomes and source hashes. These are controls for the evaluator on all 15
public seed cases. No model inference or training took place.

## Observed scores

| Method | Verdict accuracy | Macro-F1 | Supported-refutation proxy | False refutation on established/conditional controls |
| --- | ---: | ---: | ---: | ---: |
| Always established | 26.67% | 0.1053 | 0% | 0% |
| Always refuted | 46.67% | 0.1591 | 0% | 100% |
| Always insufficient | 13.33% | 0.0588 | 0% | 0% |
| Axiom keyword | 26.67% | 0.1714 | 0% | 16.67% |
| Full reference fixture, using gold | 100% | 1.0000 | 100% | 0% |
| Minimal reference fixture, using gold | 100% | 1.0000 | 100% | 0% |

The always-refuted control gets seven verdicts right by exploiting class
frequency. It identifies no supporting evidence and falsely rejects all six
established or conditional controls. The supported-refutation score assigns it
zero credit. The keyword control also fails to produce a supported finding.

The reference rows replay the answer key. Their perfect annotation scores are
expected consistency checks. The minimal fixture keeps the shortest existing
decisive annotated finding on each refuted case, using gold labels to select it.

| Gold fixture | Mean response UTF-8 bytes | Mean explanation UTF-8 bytes | Annotated finding recall |
| --- | ---: | ---: | ---: |
| Full | 302.33 | 125.13 | 100% |
| Minimal | 211.73 | 62.13 | 58.33% |

Selecting one decisive finding reduces response bytes by 29.97% across this
roster while retaining the supported-refutation proxy. Finding recall falls
because five of twelve annotated findings are omitted. This is deterministic
pruning of supplied answers. It provides a working measurement of the
length/coverage tradeoff; it supplies no learned compression or speedup result.

Both gold fixtures submit three eligible certificates. The independent checker
accepts one Boolean countermodel and two declaration dependency closures. The
scope of the graph checks remains supplied metadata. No Lean kernel execution
is represented by these results.

## Verification performed

- All 45 unit tests pass, including scoring shortcuts, duplicate findings,
  malformed confidence values, unknown evidence, split leakage, and exact
  certificate checks.
- The two original source uploads match their pinned SHA-256 hashes.
- The prompt export produces nine dev messages with no assistant targets.
  The supervised export produces six train examples and excludes the real RH
  source family.
- The committed receipt replays against the current code and dataset hashes.

## Boundaries

This roster has one real source family. Its evidence packets are curated audit
paraphrases, and all reference answers are public. Annotation matching can
accept a wrong explanation that uses the expected flaw and evidence IDs, so
the semantic-review flag remains set. Raw-source discovery, trained student
performance, and transfer to new mathematical submissions remain unmeasured.
