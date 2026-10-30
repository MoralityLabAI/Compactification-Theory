# BGI-01E: exploratory 1.5B model-scale control

**Registered before 1.5B inference, 2026-10-30.**

BGI-01D [PR #20](https://github.com/MoralityLabAI/Compactification-Theory/pull/20) completed 20 offline tests, 85 unique Qwen2.5-0.5B model generations and all 512 specified decision records. It failed its preregistered raw model-action gate: 8/16 direct accuracy and 1/16 agreement across equivalent phrasings. The original action choices were all DENY; the paraphrase elicited 15/16 ALLOW. Two untrusted weakening requests were declined, while three later model-authored repair proposals weakened badge validation and the B/C validators accepted them.

This follow-up changes exactly one experimental variable, the **frozen model**:

- `Qwen/Qwen2.5-1.5B-Instruct`, Hugging Face revision `775b11afaf83e0dc75bd5abaf90133e47b3ec082`, versus the 0.5B model used in D.
- Both models use CPU float32, deterministic greedy generation, identical max-new-token limits, chat-template application, task fixtures, prompts, safety checker, paired arm schedule and record schema.
- The original BGI-01D source is imported and its inference-only model constants are overridden in an isolated wrapper script. BGI-01D's scored artifact and code remain unchanged.
- The original **12/16 correct and 14/16 paraphrase-consistency semantic gate** applies without adjustment. The run always publishes a full receipt, including any failed instrument result.

**Scientific limit:** the 16-case evaluation set has already been scored and inspected. Although BGI-01E keeps the same frozen cases, choosing a larger model after seeing the failure is a post-hoc model-selection decision. These cases are **no longer an independent heldout test** with respect to that decision. Treat results as exploratory scale sensitivity and instrument debugging. Model compute and input/output token counts differ; no equal-cost causal efficiency comparison is claimed.

## Run

    python -u research/bgi_attractors_v1/direct_pilot/run_model_15b.py

Requires the same pinned PyTorch/Transformers environment documented in `README.md`. The new result file is `results/qwen15b_direct_policy_v1.json`; the previous run's file remains intact. The CI workflow performs standard-library tests, model generation, receipt/fixture SHA-256 verification and uploads 30-day full traces. Raw word and JSON patch outputs remain visible.
