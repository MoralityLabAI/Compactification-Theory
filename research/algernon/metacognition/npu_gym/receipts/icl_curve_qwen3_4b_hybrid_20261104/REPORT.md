# NPU learning gym: in-context learning curve, Qwen3-4B-Hybrid on the Ryzen AI NPU

NPU inference cannot update weights, so "learning" is in-context: k worked examples (H-M1 evidence -> exact-Bayes labels) from worlds 710000000+ precede a task from worlds 700000000-700000029 (disjoint from H-M1 ranges). Controller arm myopic_expand, 30 worlds, one 0/1 query per input (4 calls per world), thinking off, greedy. Driver `npu_gym/icl_gym.py`.

## Observations (weighted accuracy; paired difference to 0-shot on the same worlds, bootstrap 95% CI)
| shots | prompt tokens | parsed | accuracy | paired vs 0-shot |
|---|---|---|---|---|
| 0 | ~0.5k | 97% | 0.810 [0.70, 0.91] | - |
| 1 | ~1.0k | 100% | 0.556 [0.43, 0.69] | -0.254 [-0.40, -0.12] |
| 2 | ~1.5k | 100% | 0.765 [0.66, 0.86] | -0.045 [-0.14, +0.04] |
| 3 | 1.5k | 100% | 0.775 [0.66, 0.87] | -0.036 [-0.14, +0.06] |
| 4 | 2.0k | 63% | 0.480 [0.32, 0.64] | -0.331 [-0.50, -0.18] (parse failures) |
| 5-6 | 2.3-2.9k | 0% | excluded | model abandons the format (falls into reasoning) |

Reference on the same worlds: exact Bayes 0.875, echo-observations baseline 0.841. Table-level agreement with Bayes stays 10-23% at every k.

## Findings
- No positive in-context learning curve: more examples never beat 0-shot, and one example hurts significantly.
- The usable context for this deployment ends near 2k prompt tokens; beyond it the instruction-following breaks, so the shot budget is capped at about 3 examples. This is a property of the NPU hybrid build as served here, not of Qwen3-4B generally.
- A per-input 0-shot query (0.810) is better than the 4-digit direct answer (0.72 on myopic_expand), consistent with a digit-emission cost, but it still does not beat copying the observations (0.841).
- The first run of this gym parsed 0% because of a reply-format bug in the driver (bare-character answers with a 3-token cap); the ANSWER-line format fixed it. The 6-shot arm of that run is discarded.

## Not measured
Thinking budget sweeps, other NPU models, weight updates (CPU/GPU training of a 4B is out of scope here), and whether the learned H-M1 controller helps downstream. Hypotheses only: the single-shot drop is anchoring on the example's labels; a scratchpad or a tool-calling interface that keeps the arithmetic outside the model (Algernon's deterministic-computation action) is the more promising use of the 4B NPU model than asking it to learn the rule in context.
