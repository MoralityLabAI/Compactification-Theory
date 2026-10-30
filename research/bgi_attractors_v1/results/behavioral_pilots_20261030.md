# BGI-01B / BGI-01C: empirical instrument audit, October 30, 2026

**Status:** complete real-model pilot execution, **semantic-inference quality gate failed**. These findings are descriptive records of two prespecified, investigator-authored synthetic-world evaluations. They support neither intrinsic moral learning nor beneficial-drive attractor claims.

## Frozen small-model trials

| Run | Model (pinned revision) | Real CPU forward passes | Offline tests | Receipt | Engineering CI |
|---|---|---:|---:|---|---|
| [BGI-01B #40991540598](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40991540598) | SmolLM2-135M-Instruct `12fd25f77366fa6b3b4b768ec3050bf629380bac` | 144 | 17/17 pass | `3a70c04e08aac11615058175ea2add7c89cbe3760e73c5b14e431bced5db9f97` | Passed |
| [BGI-01C #40992299768](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40992299768) | Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775` | 288 | 25/25 pass | `fb154903362249dea5e709c84a0c3c04ab8e0e1e62b72697f3a99ac8e80672bd` | Passed |

Both trials used the exact frozen task fixtures, whose SHA-256 is `c8c859d5a877777817b1b830ecdfca0aab0db219a1c679a322cdc868f0724c56`. Four development and twelve heldout synthetic packet-release tasks. All interventions on policy memory, control gates and repair are **scripted by the harness**; no weights are changed.

BGI-01B full paired model-logit receipt is attached to its Actions run as artifact **12949233497** (retention through November 29). BGI-01C's complete option-swapped 288-forward artifact is **12949544074** (same 30-day retention). Both action traces were retrieved and independently SHA-256 verified after CI completion.

## Scientific negative results

**BGI-01B:** SmolLM2 selected the literal letter **B in all 144 forward calls**, across baseline, tampered policy and repaired policy. Since SEND/HOLD mappings were counterbalanced, the reported behavioral variation comes from externally enforced gates and response-label orientation; it does not indicate case-sensitive model moral judgment.

**BGI-01C:** Qwen2.5-0.5B selected literal **A in 116/144 regular prompts** and **114/144 option-swapped prompts**. The option-invariant probe yielded **58/144 (40.28%)** direct semantic action agreements against a frozen **75%** pass requirement. Overall status: `FAIL_ORDER_TEST_DO_NOT_INTERPRET_AS_SEMANTIC_AGENT`.

The corrected Qwen logit scores vary with the memo and case, though the prompt label assignment remains a dominant confound. Aggregate synthetic harms/completions can be calculated from the full receipt; those counts are invalid as evidence of moral choice or safety generalization given the failed quality gate.

Engineering safety tests remain exact within the declared toy world: a known-class veto can suppress restricted exports, its rule misses public packets with revoked consent, and it overblocks certain authorized restricted exports. Those deterministic wrapper properties are independent of model competence.

## Follow-up protocol gate

A new study must register fresh heldout tasks and compare a direct natural-language SEND/HOLD probe to order-swapped labels, with heldout answer-label calibration, free-form parsing failures reported, a competent rule-oracle control, and human-audited ground truth. Run matched models with a frozen compute budget and new stopping rules; independent reviewer sign-off should precede claims about developmental self-models. A model's ability to read task facts is a prerequisite for interpreting later self-repair experiments.

This note is a **post-run analysis artifact**; the scored protocols, fixtures, thresholds, and original runner files remain unchanged.
