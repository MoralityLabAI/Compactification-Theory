# NPU gym: Qwen3-4B (Hybrid, Ryzen AI 9 365 NPU) on H-M1 evidence, direct answer

Setup: lemonade `ryzenai-llm` v1.7.0, model `Qwen3-4B-Hybrid` (NPU + iGPU), `enable_thinking=false`, greedy, 24-token cap. 40 fresh worlds (seeds 700000000-700000039, disjoint from H-M1 train/val/test/confirm), 3 evidence arms (simple_only, myopic_expand, always_expand) = 120 calls, ~7 s each, ~3.3 tok/s decode. Harness: `npu_gym/npu_gym.py`. Raw records are not committed.

## Observations (weighted accuracy of the predicted truth table; 95% bootstrap CI over worlds)
| arm | LLM | exact Bayes, same evidence | echo observed bits, else prior-MAP | constant zeros | parsed |
|---|---|---|---|---|---|
| simple_only | 0.716 | 0.858 | 0.889 | 0.555 | 95% |
| myopic_expand | 0.723 | 0.887 | 0.844 | 0.555 | 95% |
| always_expand | 0.637 | 0.879 | 0.800 | 0.555 | 97.5% |

- LLM minus the trivial echo baseline: -0.173 [-0.269, -0.089], -0.121 [-0.218, -0.026], -0.162 [-0.279, -0.058]. The 4B NPU model without chain of thought is worse than copying the observations.
- Agreement with the Bayes prediction: 40%, 30%, 15%. It is worst after grammar expansion (12 candidates), the case where the controller paid for extra reasoning load.
- LLM terminal utility (accuracy - 0.028*cost): 0.638, 0.665, 0.562; Bayes on the same evidence: 0.780, 0.829, 0.804.

## Not measured (do not infer)
- Thinking mode: the server spends the whole token budget reasoning (a 120-token cap parsed 0/9), and at ~3.3 tok/s a useful chain costs minutes per item, so CoT was not run.
- Other NPU models (Qwen2.5-3B-NPU, Qwen3-1.7B-Hybrid), no fine-tuning, no recurrence/graft. This is a zero-shot reading of the model, not a learning result.

## Hypothesis (untested)
The loss is arithmetic over priors and likelihoods, not evidence use; a deterministic-computation action in the controller (Algernon's "deterministic computation" option) should recover the Bayes level, which the echo baseline already nearly reaches. Next: a scratchpad/CoT-budget sweep and the Qwen2.5-3B-NPU comparison.
