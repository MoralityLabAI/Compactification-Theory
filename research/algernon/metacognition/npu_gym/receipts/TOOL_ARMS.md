# NPU gym follow-up: does a deterministic computation fix the 4B NPU model?

Same 40 fresh worlds, 120 calls per arm, Qwen3-4B-Hybrid on the NPU, direct answer (24 tokens, thinking off). Weighted accuracy / agreement with exact Bayes, per evidence arm (simple_only, myopic_expand, always_expand):

| prompt gets | accuracy | agree with Bayes |
|---|---|---|
| evidence only (`direct_...`) | 0.716, 0.723, 0.637 | 40%, 30%, 15% |
| + exact posterior | 0.708, 0.698, 0.726 | 33%, 35%, 23% |
| + exact P(f=1) per input (only thresholding left) | 0.716, 0.742, 0.693 | 30%, 28%, 18% |
| exact Bayes / echo-observations baseline | 0.858-0.887 / 0.80-0.89 | 100% |

## Observations
- Neither tool output raised the model toward Bayes. The earlier hypothesis that the loss is arithmetic over priors and likelihoods is not supported.
- With the marginals supplied, the answer is a threshold at 0.5 per digit, yet whole-table agreement is 17-30% and per-digit agreement 68%. The modal answer is `0000` (42 of 117 parsed).
- Parse rate 95-100%.

## Reading (hypothesis, not tested)
The failure sits in emitting four binary digits from this INT4 hybrid model without a scratchpad (mode collapse toward `0000`), not in the reasoning inputs. Next experiments: per-input yes/no queries (one digit per call), a short scratchpad sweep within the ~3.3 tok/s budget, and Qwen2.5-3B-NPU for comparison.
