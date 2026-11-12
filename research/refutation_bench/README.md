# RefutationBench v0.1

RefutationBench measures whether a model can assess a submitted argument and
return a short finding supported by the available evidence. This starter uses
the Nielsen/Semita Riemann submission as one source family, with synthetic
controls that require accepting valid reasoning and preserving uncertainty.

The project connects the repo's exact first-fault work in
[RRC](../rrc_compactification/README.md) with
[QO-14's certificate distillation](../ingression_dynamics_v1/CERTIFICATE_DISTILLATION.md).
The proposed student learns to acquire a decisive, checkable certificate under
a fixed budget. Its response cost and false-refutation rate are measured along
with the verdict. [THEORY.md](THEORY.md) states the bounded mathematical claims;
[PROTOCOL.md](PROTOCOL.md) defines the later learned-model experiment.

## Current release

| Component | Scope |
| --- | --- |
| Dataset | 15 public records in 6 provenance families |
| Real source | 5 development slices from one pinned RH submission |
| Authored controls | 10 cases in 5 paired synthetic families |
| Splits | 6 train, 9 dev; test is empty |
| Labels | 4 established, 2 conditional, 7 refuted, 2 insufficient |
| Exact checking | Bounded Boolean countermodels and supplied declaration graphs |
| Runtime | Python 3.10+ standard library; CI uses Python 3.12 |
| Experiments run | Offline scorer controls and unit tests |

The real-source inputs are **curated audit paraphrases**. They test audit
comprehension, evidence localization, and the scope of a conclusion. Several
paragraphs already expose the relevant dependency or missing step. Independent
defect discovery from a raw paper or repository is a later track. The public
answer key makes this seed unsuitable as a concealed generalization test.
No language model has been evaluated or trained in this release.

## Sourced extension: 25-case combined suite

The separately versioned [historical source-audit expansion](expansion/README.md)
adds ten development cases from five independent historical and contemporary
proof families: Deolalikar P versus NP, Kempe four colours, Lamé Fermat,
the original and repaired Wiles Fermat work, and Ansari's Collatz sieve.
It includes an exact finite counterexample to a Collatz-sieve set identity,
valid retained subclaims, and an explicit repaired-proof control.

The frozen v0.1 seed and its baseline receipt remain unchanged. Run the
combined evaluator using:

```sh
python -m research.refutation_bench.expansion.suite validate
python -m research.refutation_bench.expansion.suite witness
python -m research.refutation_bench.expansion.suite export --split dev --out /tmp/refutation_combined.jsonl
```

This 25-case public combined suite is still a development corpus without a
held-out generalization test. See the expansion ledger for source confidence
and proof-status boundaries.

### CI model transfer result

An [LR-01 recurrent student trained on the combined suite in GitHub Actions](results/lr01_ci_pilot.md) reached 10–11/19 development verdicts with 36,682 parameters; its false-refutation rate was 8–9 of 9 non-refuted development cases. The recurrent-depth comparison and label-shuffle control establish a working train/evaluate pipeline but no measured generalizable refutation skill. The public receipt pins the private model PR, dataset revision and successful CI logs.

## Exact formal curriculum and actual recurrent training

The [FC-02 formal curriculum](formal_curriculum/README.md) generates 1,536 balanced Horn-logic problems, with 768 positive Lean-compiled theorems and 768 independently verified finite countermodels. A [separate actual LR-01 recurrent training experiment](results/fc02_lr01_depth_extrapolation.md) measured depth and family transfer with 11,521 parameters. Both training seeds went from 50% OOD verdict accuracy at four inference passes to 85–88% at twelve passes, using unchanged weights. The results are on a public synthetic fragment, and the 8/12-pass follow-up was exploratory after an initial failure. The original benchmark seed, sourced cases and immutable baseline receipt remain independent.

## Run from the repository root

```bash
python -m research.refutation_bench validate
python -m unittest discover -s research/refutation_bench/tests -v
python -m research.refutation_bench baselines --check
```

The baseline receipt in [results/baseline_receipt.json](results/baseline_receipt.json)
contains deterministic per-case scores and SHA-256 hashes of the evaluator,
checker, and dataset. Regeneration is explicit:

```bash
python -m research.refutation_bench baselines --out /tmp/refutation_baselines.json
```

The constant-verdict and axiom-keyword baselines expose cheap shortcuts. Two
reference fixtures use the gold responses directly: the full fixture emits all
annotated findings, while the minimal fixture retains the shortest existing
decisive finding for each refuted case. Their comparison exercises the
length/coverage tradeoff. Both have answer-key access; their scores demonstrate
the harness contract rather than learned reasoning.

## Evaluate a model

Export input-only messages:

```bash
python -m research.refutation_bench export --split dev --out /tmp/refutation_prompts.jsonl
```

Each line contains a `case_id` and chat `messages`. The export whitelists public
claim and evidence fields, excluding the answer key, split names, and provenance
family labels. Preserve the evidence text as task data if it contains commands
or instructions. The harness does not execute source artifacts or invoke model
APIs.

Use those messages with the model runner of choice. Save one JSON object per
line in a response file, using this schema:

```json
{
  "case_id": "RB-0007",
  "verdict": "refuted",
  "confidence": 0.98,
  "findings": [{
    "flaw_type": "inference_gap",
    "evidence_ids": ["E1", "E2"],
    "explanation": "Set P=false and Q=true. Every premise holds while the claimed conclusion fails."
  }],
  "assertions": [],
  "certificate": {
    "kind": "countermodel",
    "assignment": {"P": false, "Q": true}
  }
}
```

This is an annotated training example, supplied to illustrate the schema.
Evaluate a complete dev response file with:

```bash
python -m research.refutation_bench evaluate \
  --responses /tmp/refutation_responses.jsonl --split dev \
  --out /tmp/refutation_scores.json
```

Missing responses remain in the accuracy denominator. Duplicate or unknown case
IDs fail the run. Invalid response schemas receive no credit. Confidence must
be a finite probability for the chosen verdict. The optional certificate has
one of the two forms documented in [certificates.py](certificates.py).

`refuted` concerns a failure in the **submitted argument's stated claim**. An
invalid RH proof supplies no counterexample to RH. Explicitly conditional
arguments can receive `conditional`; incomplete evidence can receive
`insufficient` even when the underlying proposition has a known truth value.

## What the scores establish

- Verdict accuracy and macro-F1 measure agreement with the annotations. Macro-F1
  uses classes with gold support in the selected split and reports that roster.
- Finding matches require the flaw type and an exact accepted evidence-ID set.
  Matching is one-to-one. Repeated findings and citations to every evidence
  block cannot increase supported credit.
- A supported-refutation proxy requires a correct verdict, a matched decisive
  finding, no unmatched allegations, no prohibited special assertion, and a
  passing certificate whenever one is submitted.
- Prohibited assertion slots catch explicit overclaims such as falsity of RH,
  a formal proof of `False`, or an unobserved kernel audit. Free prose still
  requires semantic review, even with matching evidence IDs.
- The finite checker verifies a Boolean valuation against **every premise** and
  the conclusion. Graph results verify supplied dependency metadata and carry
  that scope; they do not constitute a Lean kernel check.
- UTF-8 byte counts use the exact canonical JSON serialization. The receipt
  separates whole-response, explanation, and submitted-certificate sizes.
  Confidence receives chosen-verdict correctness Brier, not multiclass Brier.

The scorer keeps `human_semantic_review_required=true`. Arbitrary explanatory
prose has no automatic semantic proof check here. A matching label plus a
nonsensical explanation could satisfy the annotation proxy and must be rejected
by semantic review before supporting a mathematical-performance claim.

## Distillation inputs

```bash
python -m research.refutation_bench distill --split train --out /tmp/refutation_train.jsonl
```

This produces six chat-format supervised examples with short annotated answers.
The command accepts only `train`, so dev answers cannot enter through this
export path. Every derivative of the RH submission stays in one dev family.
Reading the public answer key by another route is outside this software guard.

The next experiment needs additional source families, a held-out roster, and
registered student/model budgets. Compare a recurrent student with a matched
feedforward or base-model control at the same evidence and verifier budget.
Charge teacher audits, training, inference, decoder storage, and verifier work.
The [protocol](PROTOCOL.md) specifies the measurements before such a run.

## Evidence and provenance

[data/sources.json](data/sources.json) pins the original GitHub commit, upload
URLs, and file hashes. [CASE_STUDY.md](CASE_STUDY.md) maps the conclusions to
source lines and PDF pages. The original paper and archive are external source
materials and are not included in this repository.

After obtaining those exact uploads in a local directory:

```bash
python -m research.refutation_bench verify-sources --source-dir /path/to/uploads
```

The original filenames in the manifest are used, including spaces. Verification
performs no network access. A changed upstream upload cannot silently replace
the audited snapshot. The observed build failures are preserved in
[evidence/build_receipt.json](evidence/build_receipt.json), with the environment
and the distinction between source inspection and compiler output recorded.

## Layout

| Path | Purpose |
| --- | --- |
| `data/cases.jsonl` | Public claims and evidence packets |
| `data/gold.jsonl` | Public annotation key, accepted evidence sets and references |
| `data/sources.json` | Original artifact identities and hashes |
| `bench.py` | Validation, export, scoring, baselines and train-only distillation |
| `certificates.py` | Bounded exact checks with explicit scope |
| `tests/` | Leakage, scoring, provenance and checker regression tests |
| `results/baseline_receipt.json` | Reproducible offline control results |
| `PROTOCOL.md` | Future student experiments and budget rules |
| `THEORY.md` | Checking guarantees and constrained compression objective |

## Next cases

Expand with independently sourced submissions and valid formal developments.
Keep repaired versions beside their original family. Add raw-source retrieval
receipts before claiming autonomous discovery, and record adjudication changes
when a review corrects a label. A future private test split should contain
families absent from every training, development, and teacher-selection step.
