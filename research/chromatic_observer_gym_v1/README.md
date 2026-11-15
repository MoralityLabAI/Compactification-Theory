# COE-01: evaluator-graph chromatic compactification gym

**2026-11-15. Research stage:** exact synthetic finite benchmark, without a trained transformer or SAE result.

Inspired by two distinct mathematical mechanisms from the October OpenAI collection:

- [Family 098](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/docs/098.md): globally bi-Lipschitz faithful embedding of certain compact doubling Hilbert sets into finite-dimensional spaces fails.
- [Family 332](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Metric-Markov-Cotype-Two-of-l1-October-5-2026/README.md): claims arbitrary Hilbert-subset ℓ1-valued Lipschitz functions admit extensions with universal multiplicative Lipschitz loss.

For compact models, the practical question is often which **specific evaluator-relevant distinctions** the latents must preserve. Define a graph with one vertex per synthetic activation state, and an edge for each required distinction. Assign bits to each vertex and require different codes across tested edges.

The direct result proved in [the MathCan theorem note](https://github.com/patrickdugan/MathCan/blob/main/research/compactification/CHROMATIC_OBSERVER_THEOREM.md) is

```
min_ordinary_probe_bits(G) = ceil(log2(chromatic_number(G))).
```

A separation margin `r` in Hamming distance survives any `r-1` erased readout features, or `t` adversarial bit flips when `r>=2*t+1`. Any graph can use a repeated-color code of width `r*ceil(log2 χ)`, while an exhaustive solver can find smaller codes for finite cases. The abstract feature map embeds tested states in a Hilbert star `0,e_1,...,e_n`, and each bit can extend to the entire Hilbert space by a 1-Lipschitz clipped McShane formula. This is a forward-observable result. It does not certify latent-state reconstruction or inverse bi-Lipschitz geometry.

## Reproduction

At repository root (Python 3.10+, standard library only):

```bash
python -m unittest discover -s research/chromatic_observer_gym_v1 -p 'test_*.py' -v
python research/chromatic_observer_gym_v1/observer_codes.py
```

The test suite has five regression methods, including complete enumeration of all 64 undirected simple graphs on four vertices, exact chromatic palette bounds, independent Hamming-erasure/flip checks and 300 deterministic input-pair Lipschitz extension checks.

| Evaluator interface | States | χ | Bare bit width | Margin 3 width |
| --- | ---: | ---: | ---: | ---: |
| Full-distinction clique K6 | 6 | 6 | 3 | 6 |
| Adjacent-step path P8 | 8 | 2 | 1 | 3 |
| Odd cycle C5 | 5 | 3 | 2 | 4 |

Margin-3 values are **exact** for these finite inputs because the solver scans all smaller widths with graph-symmetry-preserving search; check `minimum_observer`, which is exponential and intended only for small graphs. For K6, the width-5 lower bound has a separate six-radius-one-spheres packing proof, and the six-bit witness is also enumerated in Lean.

## How to connect to live compactification experiments

1. Extract a frozen set of latent states or SAE reconstructions from a specific checkpoint. Define evaluator edges from independently tested behavior, splitting train/test states before adjusting the graph.
2. Compare edge accuracy of learned 1-, 2-, ...-bit probes with the exact graph χ bound on a deliberately tiny controlled subset, while varying per-edge Hamming margin. Treat edge sets from held-out tasks separately.
3. Report full pairwise embedding distortion, evaluator-edge retention, erasure tolerance and bit width separately. The graph theorem predicts the best **unconstrained symbolic observer**, not trainability or robustness under nonlinear learned probes.

The present run contains no model training, causal interventions, source-paper proof reproduction, or out-of-sample biological/mechanistic claims. Its purpose is an independently checkable benchmark and a concrete mathematical distinction between global geometry and task-local observation.

## Source and verification ledger

- OpenAI source repository pin: `openai/math@4eaee87548250b2682affda7d05d947c05554bae`.
- Application starting commit: `MoralityLabAI/Compactification-Theory@28d8186d2393f0f80d0d4302e86b0bb82ef61190`.
- Pure math starting commit: `patrickdugan/MathCan@fca46bfb3430ac85ffca16f6a6a83bb0b0231e41`.
- Local test environment: Python 3.13 standard library; `python -m unittest discover` five tests passed on 2026-11-15.
- Lean kernel target: v4.35.0 (repository pinned), build audited in separate MathCan CI; current local container has no Lean compiler, so no claim of local Lean kernel verification before CI.
