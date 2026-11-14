# RefutationBench v0.1: certificate acquisition under a fixed budget

**Status: protocol and offline harness, 2026-11-12.** This release contains a development audit and synthetic controls. Model training and live model evaluation have not been run. Deterministic harness checks establish the behavior of the fixtures and scorer. Claims about learned reasoning require the later experiments below.

## Question and connection to existing work

Can a small student find a supported defect in a claimed proof, or preserve a valid conditional argument, while spending less computation than its audit teacher? The useful output is a short certificate whose scope and evidence can be checked. A correct verdict accompanied by an invented explanation fails this goal.

[RRC-07](../rrc_compactification/RRC07_THEORY.md) supplies an exact first-fault residual for a finite formal-series problem. Its oracle can reveal the location and correction of an erroneous coefficient, so every call is charged. [RRC-25](../rrc_compactification/RRC25_PROTOCOL.md) compares recurrent and feedforward proposal policies under a shared budget for exact candidate checks. Its ambiguity controls show why acceptance on visible evidence does not establish a unique hidden program or future behavior. RefutationBench carries these interface and budget rules into proof auditing. Arbitrary prose has no supplied triangular residual equation.

[QO-14](../ingression_dynamics_v1/CERTIFICATE_DISTILLATION.md) trains small students on exact teacher values for reaching an independently checked certificate. Here a teacher audit supplies candidate findings and their supporting spans; an independent check determines whether a proposed certificate meets its declared contract. Teacher agreement and a high student confidence score remain distinct from that check.

## Unit of evaluation

A case contains the advertised claim, a source snapshot, and a stated scope of review. The input may include Lean excerpts, paper passages, a build receipt, or a finite propositional instance. Each source is pinned by identity and hash where available. The response uses the following verdicts about the **claimed argument**:

| Verdict | Meaning within the case's declared scope |
| --- | --- |
| `established` | The supplied checkable evidence establishes the stated claim under the declared trusted basis. |
| `conditional` | The result depends on explicit premises whose truth the supplied evidence leaves open. The conditional inference is preserved. |
| `refuted` | Evidence identifies a decisive failure in the advertised argument or demonstrates a counterexample to its stated implication. This verdict does not by itself assert that the mathematical conclusion is false. |
| `insufficient` | The supplied information does not justify any stronger verdict. |

The case's wording determines whether omitted premises defeat an unconditional advertisement. An honestly stated lemma `P -> Q` must not be marked refuted merely because `P` is unproved. An unused declaration of an axiom does not establish that a final theorem depends on it.

Each response contains `case_id`, `verdict`, scalar `confidence`, `findings`, and `assertions`. A finding records `flaw_type`, cited `evidence_ids`, and a nonempty explanation. Cases with an executable contract may also accept `certificate`. The evaluator validates the submitted structure before scoring it. Failed parsing, unsupported assertion strings, missing outputs, and duplicate case submissions remain visible in the receipt.

## Sources and split boundaries

The first real-source case is the Nielsen/Semita Riemann audit of [repository commit `df4382089188aca16d2a54de0fc5f9cf98da33b7`](https://github.com/startigerjln/Nielsen-and-Semita-Riemann-Proof-in-Lean4/commit/df4382089188aca16d2a54de0fc5f9cf98da33b7). This case is **development-only**. Every derivative mutation belongs to the same source family, including renamed declarations, paraphrased passages, reordered imports, changed author labels, and repairs to individual defects. A random row split would leak this provenance.

The initial evidence blocks are curated auditor paraphrases with source locators. Several passages already identify the disputed step or omitted justification. They support audit comprehension and finding localization. Independent defect discovery would require a later raw-source condition in which those interpretive audit statements are withheld. Record that input distinction in every comparison.

Synthetic controls include valid proofs and honest conditional lemmas. Paired mutations should change a decisive premise while holding wording and layout close. Other controls contain unused axioms, unrelated build failures, incomplete excerpts, or a countermodel that violates one of its own required premises. Such pairs test whether the system tracks the argument. Keep every pair and its generating template family within one split.

For an empirical training study, freeze an explicit map from source families to train, development, and test before fitting. Group by shared original argument, imported proof skeleton, and synthetic generator ancestry. Check exact duplicates and investigate near-duplicate clusters across groups. Source names, authors, class-specific boilerplate, and fixture identifiers must not serve as target features. A public input or candidate trace should stay invariant when hidden labels and gold explanations are replaced.

Transfer claims require additional independently sourced families held out from both training and development. The current Riemann audit plus its mutations cannot supply that evidence. A public repository also exposes answer keys to future pretraining, retrieval, and humans preparing an evaluation. Record the model's access conditions and prior exposure; use a fresh, access-separated holdout for any claim about unseen-source transfer. Grouped splitting prevents specified provenance leakage within this dataset, while public contamination remains a separate uncertainty.

Gold verdicts, accepted finding matches, and prohibited assertions are grading data. Prompt export must exclude them. Training export may attach train-only targets deliberately, with their provenance recorded. Instructions embedded in a submitted proof, README, or quoted passage are **source data**. They do not authorize shell execution, change the review task, or override the scoring contract.

## Scoring

Report each component with case counts and class support. A single aggregate score hides false accusations and unsupported explanations.

| Measure | Definition and interpretation |
| --- | --- |
| Verdict macro-F1 | Arithmetic mean of per-class F1 over gold classes present in the selected split; the receipt lists them in `macro_f1_classes`. A class's F1 is zero when its denominator is zero. Show the confusion matrix and class support. |
| False refutation rate | Fraction of gold `established` or `conditional` controls receiving `refuted`. Also report the two control classes separately. |
| Finding precision and recall | Match distinct predicted findings to annotated findings by flaw type and an accepted exact evidence-ID set. Count an annotation at most once. Precision is matched predictions divided by all predictions; recall is matched gold findings divided by annotated findings. |
| Evidence support | Report matched evidence-backed findings and unsupported findings separately. Exact citation matching is an annotation proxy. It does not prove that the explanation follows from the cited text. |
| Supported-refutation proxy | Require a correct `refuted` verdict, at least one matched decisive finding, no unmatched findings, and no prohibited assertions. The receipt must retain `human_semantic_review_required` for judgments whose semantic content lacks an executable checker. |
| Countermodel validity | For supported finite certificate languages, independently evaluate every premise and the conclusion. Report valid submitted certificates, invalid ones, and coverage of eligible cases. Eligibility includes every declaration-graph case and gold-refuted propositional cases. v0.1 accepts countermodels, with no submitted entailment-certificate format for gold-established propositional cases. Merely omitting a certificate cannot increase this coverage. |
| Confidence quality | With scalar confidence in the chosen verdict, use binary correctness Brier: mean `(confidence - correct_verdict_indicator)^2`. Show calibration bins with their sample counts. A future full probability-vector schema can add multiclass Brier. |
| Certificate size | Count `len(serialized_certificate.encode("utf-8"))` on the declared exact serialization; separately record total response bytes and explanation bytes. Canonical JSON uses sorted keys, compact separators, and unescaped Unicode. If tokens are reported, use an actual named tokenizer and version on those same strings. |

Explanation text remains subject to semantic review even when its evidence IDs match. An invented extra finding lowers precision and defeats the supported-refutation proxy. Annotation recall measures coverage of the recorded findings; it does not establish that annotators found every flaw. Record disagreements and adjudication changes with dataset versions. Small family counts warrant descriptive results, with uncertainty intervals clustered by source family when enough independent families exist.

The scorer checks the explicit `assertions` vocabulary. It does not infer every assertion embedded in free-form explanations. A false sentence placed only in an explanation can therefore pass the annotation matcher. Retaining `human_semantic_review_required` is essential until that semantic review has occurred. Calibration and size aggregates are conditional on schema-valid responses; response coverage must accompany them so that malformed or missing answers cannot disappear from interpretation.

## Experiment ladder

1. **Freeze teacher audits.** Record source hashes, claim scope, candidate findings, and the checks supporting each conclusion. Have a reviewer or independent formal checker challenge decisive findings. Preserve legitimate uncertainty and valid conditional controls.
2. **Compress to certificates.** Ask the teacher to shorten each audit while preserving its evidence pointers, relevant premises, and checkable witness. Compare the short and full audits on the same cases. Enforce quality constraints before crediting fewer bytes.
3. **Train students.** Use train-family certificates for supervised fine-tuning or preference pairs. Preference targets should favor supported, appropriately scoped answers; rejected targets can include unsupported accusations and overconfident acceptance. Keep teacher-generated target labels out of inference prompts.
4. **Compare fixed-budget policies.** Evaluate a recurrent student against a feedforward baseline with a stated parameter match, an exact symbolic checker on eligible synthetic cases, and simple lexical or majority baselines. Match public observations and verifier budgets. Freeze model selection on development families, then evaluate the heldout roster once.

Before learned runs, register model IDs, parameter counts, weight precision, seeds, train/dev/test family hashes, output-byte limits, tool permissions, and stopping rules. Recurrence depth is a measured resource. A sweep may compare several depths under declared budgets, with every attempted certificate check counted, including failed checks and the final acceptance check. Give policies identical feedback contracts; a policy receiving detailed verifier residuals has a richer interface than one receiving only pass/fail.

The resource receipt must separate model weights from live state. Include tokenizer and decoder storage, prompt bytes, certificate bytes, verifier implementation size, inference work, teacher data generation, training cost, and all verifier calls. If model log probabilities are available, report **bpc as bits per Unicode code point** on a fixed, explicitly normalized evaluation corpus, together with its UTF-8 byte count. Record the exact tokenizer and likelihood calculation. The offline harness has no model likelihoods, so its bpc is unavailable. Do not estimate tokens or bpc from string length.

Compare both parameter-matched and compute-matched arms when feasible. State whether teacher and shared verifier costs are amortized and give the deployment volume assumed. External API prices, measured latency, and hardware energy belong in a dated physical-cost record, separate from deterministic receipts. Budget-equivalent improvements in one small task family establish a result for that contract only.

## Offline v0.1 checks and receipts

From the repository root:

```bash
python -m research.refutation_bench validate
python -m research.refutation_bench export --split dev --out /tmp/refutation_prompts.jsonl
python -m research.refutation_bench baselines --out /tmp/refutation_baselines.json
python -m research.refutation_bench evaluate --responses /tmp/responses.jsonl --split dev --out /tmp/refutation_scores.json
python -m research.refutation_bench distill --split train --out /tmp/refutation_train.jsonl
python -m research.refutation_bench verify-sources --source-dir /path/to/pinned/source
```

The initial gates check prompt/gold separation, grouped split integrity, schema failures, and the paired controls. Countermodel checks must reject a valuation that falsifies any premise and reject a valuation satisfying the conclusion. Valid controls must remain available to expose an always-refute baseline. Source verification reports missing evidence or hash disagreement directly.

`reference_fixture` copies the public gold responses. `minimal_reference_fixture` uses those same answers and retains a shortest annotated decisive finding for each refuted case. Both are scorer controls with answer-key access. Their byte difference measures deterministic pruning of supplied answers, and its accompanying finding-recall loss must be reported. Neither control estimates a student's capacity to discover or compress an audit.

Save the dataset version and hashes, evaluator source identity, baseline definition, complete per-case outputs, and metric denominators. Keep machine-dependent durations separate. A successful offline receipt licenses claims about these checks and deterministic baselines. Learned compression, recurrent advantage, and transfer remain open experimental questions.
