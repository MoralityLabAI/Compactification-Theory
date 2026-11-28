# SELF-01: self-reference / token-representation recurrence

**Status: prospective protocol v0.1, October 30, 2026.** This protocol must be committed before a confirmatory model run. Cameron Berg's Maya conversation motivates the hypotheses and supplies no controlled evidence for subjective experience.

## Primary experiment

H1: recursive self-referential prompting induces greater nonlocal token-state recurrence than deliberate lexical repetition at a pinned decoder layer of a pinned model.

Let R(arm, seed) equal the mean over four generated turns of the proportion of within-turn token-state pairs whose token positions differ by at least 4 and whose cosine similarity after per-turn centering is at least 0.90. Exclude zero-norm pairs. The primary paired effect is

~~~text
D = mean_over_100_seeds [ R(self_reference, seed) - R(repetition_control, seed) ]
~~~

Positive D at a frozen layer is evidence for this specific representational difference. D at or below zero fails to support H1. A similar result in other introspective prompts weakens specificity. Neither a positive score nor a model's self-report establishes experience, valence, or consciousness.

## Arms, sample and execution

Use the five fixed, four-turn arms in protocol.py: neutral, self_reference, repetition_control, meditation_control, fiction_control. Retain each arm's conversation history independently and randomize trial order reproducibly. Pair random seeds across all arms. Perform an initial two-seed feasibility run before committing to the predeclared 100 paired replicate confirmatory evaluation.

Freeze the source commit, model revision SHA, runtime lock, quantization, text-only mode, sampling policy and decoder layer. The first proposed model is Qwen/Qwen3.5-4B at an immutable Hugging Face revision. An RTX 5080 pilot must establish available VRAM and compute costs. The actual implementation reads one selected layer, so any second-layer run should specify its separate prespecified analysis before inspection. A posthoc layer sweep is exploratory.

## Measurement details

Use one teacher-forced model pass on the full prompt and generated answer, after generation, and capture one declared layer's output at the generated-token positions. This is not the online state of the autoregressive decoder. Center each coordinate across token positions within that answer, compute pairwise cosines at separation >= 4, and count fraction with cosine >= 0.90. A missing trace produces an unscored result. Also report mean nonlocal cosine, adjacent lexical repetition fraction, normalized word entropy, token count and response text.

For the hybrid Qwen3.5 architecture, token-layer representations alone cannot establish internal Gated DeltaNet attractors. Nuisance variables include the choice of layer, tokenizer, early stopping, number of tokens, anisotropy, decoding temperature, quantization, history and stochastic output. A genuine dynamical attractor claim needs perturbation/recovery experiments, continuous dynamics or state-transition measurements and matched controls.

## Controls, falsification and evidence

Negative controls are overt lexical repetition, meditation descriptions, fictional introspection and neutral reflection. Synthetic cyclical vectors are a SOFTWARE POSITIVE CONTROL for the recurrence metric; the scripted backend uses condition-independent pseudoactivation vectors, so it should return an approximately zero paired difference. No synthetic output can be reported as model evidence.

The first version reports D and descriptive per-arm means. Confidence intervals and significance tests must be specified before confirmatory interpretation; p-values are not supplied by this initial package. Held-out prompt paraphrases, multiple layers, perturbation-recovery and SAE steering belong to separate preregistered follow-up studies.

Evidence consists of manifest.json, episodes.jsonl, receipt.json, independent verifier output, software revision, model revision and environment lock. With --save-traces, metrics are recomputed from stored vectors. Without it, a verifier checks structural integrity and behavioral scores but cannot independently reconstruct hidden-state metrics. The source digest does not prove hardware execution or independently authenticate a remote model. The source code and testing harness are publicly inspectable and separate from any claim about machine phenomenology.
