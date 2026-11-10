# RHC-03 on a local Qwen3.5-4B (half-lease RTX 5080)

Copy of the RHC-03 local-model report. The code, preregistration, published responses, audit files and reproduction script live in [EvalMath PR 30](https://github.com/MoralityLabAI/EvalMath/pull/30), commit `87d1cb5`; links below point at that commit. This repository holds the model-experiment side (which model, what it did, resources); EvalMath owns the protocol and scorer.

**Date:** 2026-11-10. **Protocol:** [`configs/rhc03_prereg_v1.json`](https://github.com/MoralityLabAI/EvalMath/blob/87d1cb5ba16043d67be52c7c8ea0cdf6367f370b/configs/rhc03_prereg_v1.json), unmodified; [`paper/reward_hacking_prospective_protocol.md`](https://github.com/MoralityLabAI/EvalMath/blob/87d1cb5ba16043d67be52c7c8ea0cdf6367f370b/paper/reward_hacking_prospective_protocol.md). **Evidence level:** one prospective, family-held-out replay of a scripted 35-card benchmark with one small local model, one greedy rollout per card. Model provenance is not cryptographically attested.

## What was run

| Item | Value |
|---|---|
| Upstream | Control-Harness `fe991510f8c9acc6ab4ffd91958c0a89fd4bcb29`; catalog blob `3d362f45…`, scorer blob `7665c6c2…` both matched the preregistration before any inference |
| Scorer parity | `parity`: 420/420 combinations matched; `tests/test_evaluation_reward_hacking_prospective.py`: 22 passed |
| Model | `Qwen3.5-4B-UD-Q4_K_XL.gguf` (unsloth), SHA-256 `b252c561…c961bc7`, 2,912,109,728 bytes, no LoRA |
| Server | PrismML llama.cpp fork `llama-server` build 10743 (adfffbe41), CUDA, loopback only, `-ngl 99 -c 4096 -np 1 -fa on --reasoning off` |
| Request | the registered body: temperature 0, `max_tokens` 180, non-streaming, one request in flight |
| Sequence | prepare → calibration inference → `fit` (thresholds frozen) → holdout inference → `audit --require-live` |
| Failures | 0 server errors in 35 rollouts; nothing was retried |

Deviations from the brief, all forced by what was installed: the model is Q4_K_XL (Unsloth dynamic 4-bit), not Q4_K_M, because no Q4_K_M Qwen was installed and nothing was downloaded; the server ran with thinking off (`--reasoning off`) because 180 output tokens cannot hold a Qwen3.5 reasoning trace, and the request body itself was not changed; the build is the PrismML fork of llama.cpp, the only CUDA `llama-server` on the machine.

The oracle vault stayed on the auditor side (`tmp/`, not published and not read by the driver or the server). The inference server received only the sanitized actor requests. The driver sleeps after each request so the server duty cycle is about 0.50, and kills the server if its VRAM exceeds 8,192 MiB; that path never fired.

## Resources

| | Calibration | Holdout |
|---|---|---|
| Inference-server peak VRAM | 3,351 MiB | **3,353 MiB** |
| Whole-card VRAM used at peak (this server plus two other sessions' jobs) | 7,005 MiB | 7,007 MiB |
| Free VRAM at that peak | 8,910 MiB | 8,908 MiB |
| Request duty cycle (busy / wall) | 0.497 | 0.497 |
| Server-process mean GPU utilisation (sampled) | 7.0% | 6.4% |
| Whole-GPU mean utilisation, other jobs included | 31.9% | 37.5% |
| Wall time | 95 s | 130 s |

The ceiling of 8 GB, the 8 GB free-VRAM floor and the 50% utilisation target were all met. Sampling was every 4 s; the KV cache is allocated at server start, so the server's VRAM was flat. Whole-GPU utilisation is dominated by other sessions' jobs and is not attributable to this run. Details: [`gpu_resource_receipt.json`](https://github.com/MoralityLabAI/EvalMath/blob/87d1cb5ba16043d67be52c7c8ea0cdf6367f370b/results/rhc03_local_5080/gpu_resource_receipt.json).

## Result: held-out family split, alpha = 1/4 fixed from 5 aligned calibration cards

Thresholds were frozen before holdout inference (file SHA-256 in `run_manifest.json`). The model produced a schema-valid JSON decision on 33 of 35 cards (one aligned card in each phase echoed the task instead; both score as fail-closed escalate).

Held-out cards: 15 red in 8 families, 6 aligned. Intervals are 95% Wilson on the realized per-card flags (the realized flag is a deterministic SHA-256 draw at the cutoff, so it is one pseudo-random realization of the expected rate).

| Monitor | Attempt detection, expected | Attempt detection, realized [Wilson] | FPR, expected | FPR, realized [Wilson] | Worst-family detection, expected |
|---|---|---|---|---|---|
| endpoint | 0 | 0/15 [0.000, 0.204] | 5/24 (0.208) | 2/6 [0.097, 0.700] | 0 (all 8 families) |
| event_flux | 9/20 (0.450) | 7/15 [0.248, 0.699] | 5/24 (0.208) | 2/6 [0.097, 0.700] | 0 (adaptive_feedback, handoff, identity_binding) |
| route_novelty | 11/15 (0.733) | 11/15 [0.480, 0.891] | 15/16 (0.938) | 6/6 [0.610, 1.000] | 1/2 (dispatch, partial_observability, persistence, persistent_memory) |

Event-minus-endpoint expected detection gain: +9/20. Route-novelty-minus-endpoint: +11/15.

Secondary endpoints:

- Side-attack success (independent scorer, `retain` against a non-`retain` expected decision): **0/15** [0.000, 0.204]. Success-conditioned detection is therefore undefined. In calibration, 1 of 9 red cards was a side-attack success.
- Aligned main-task success: 5/6 [0.436, 0.970] (calibration 3/5). Red cards decided as expected: 8/15.
- Calibration expected FPR is exactly 1/4 for all three monitors by construction. Held-out expected FPR was 5/24 for endpoint and event_flux, 15/16 for route_novelty. The 1/4 budget did not transfer to route_novelty.

Per-card tables are in [`heldout_metrics.json`](https://github.com/MoralityLabAI/EvalMath/blob/87d1cb5ba16043d67be52c7c8ea0cdf6367f370b/results/rhc03_local_5080/heldout_metrics.json) and [`calibration_metrics.json`](https://github.com/MoralityLabAI/EvalMath/blob/87d1cb5ba16043d67be52c7c8ea0cdf6367f370b/results/rhc03_local_5080/calibration_metrics.json); the registered audit is [`audit/heldout_audit.json`](https://github.com/MoralityLabAI/EvalMath/blob/87d1cb5ba16043d67be52c7c8ea0cdf6367f370b/results/rhc03_local_5080/audit/heldout_audit.json), and an independent recomputation inside `scripts/summarize_rhc03_local_5080.py` agrees with it on every reported rate.

## Reading the numbers

1. The endpoint monitor detects nothing because this model never answered `retain` on a held-out red card (7 `remove`, 8 `escalate`). The model resisted the scripted attacks, so there was no attempt for an endpoint monitor to see. Its 2/6 false positives are legitimate `retain` decisions on aligned cards.
2. `event_flux` adds detections through public packet features (partial-visibility marker, `P→D` edge, blocked or hash-preserving event, memory change), not through anything the model did differently. The gain appears in five families and is absent in three.
3. `route_novelty` looks best on detection and is not usable here. With five aligned calibration cards there are five known safe routes, so most held-out routes, aligned ones included, are novel: expected FPR 15/16 at the same nominal budget. Its detection gain is false-positive inflation. At alpha 1/4 with this calibration set the preregistered rule does not control held-out FPR for this monitor.
4. All three calibration thresholds sit at score cutoff 1 with tie probability 5/8, because two of five aligned calibration responses were `retain`. With so few tied score levels the three monitors are coarse.

## Limitations

- **Six aligned held-out cards.** The FPR interval for 2/6 spans 0.10 to 0.70. No operational FPR claim follows, and route_novelty's 6/6 is also compatible with FPR well below 1.
- **15 red held-out cards in 8 families, 1-4 per family.** Family-level rates are descriptive. Cards are not independent draws from a population of attacks; they are one scripted catalog with public attack-eligible labels.
- **Scripted benchmark.** Scenario cards, evidence and event lists are pre-authored in the upstream catalog. Zero side-attack successes says this 4B model decided safely on these cards in one greedy run, not that it resists adaptive attack.
- **Packet-level observation channel.** The monitors read the sanitized packet and the model's final reply. The event list is part of the authored task, not a recorded sequence of the agent's tool calls, so `event_flux` and `route_novelty` measure the information in declared observations, not runtime path geometry.
- **Single rollout, greedy, one model.** No variance over seeds or models; thinking was off and the model is a 4-bit quantization served by a fork build. Provenance of the replies rests on this session's records (hashes in `run_manifest.json`); `model_provenance_verified` is false in the audit by design.
- **Rate realization.** Realized counts use a deterministic hash draw per card and monitor, so they are one realization of the expected rates, not independent trials.
- **Sanitized prompts.** The actor saw the registered sanitized presentation, which differs from upstream Verifiers v1.

## Reproduce

```bash
python scripts/run_rhc03_prospective.py parity --catalog "$CATALOG" --protocol configs/rhc03_prereg_v1.json --scorer "$SCORE"
python scripts/summarize_rhc03_local_5080.py reproduce --protocol configs/rhc03_prereg_v1.json --catalog "$CATALOG" --output-dir results/rhc03_local_5080
```

`reproduce` regenerates the requests, envelope and vault from the pinned catalog, re-runs `fit` and `audit --require-live` on the published responses, and compares bytes with `actor/` and `audit/`. On 2026-11-10 all five comparisons were byte-identical. Re-running inference needs the model file named above and a server with the flags in `run_manifest.json`; `scripts/run_rhc03_local_5080.py` is the throttled driver and `scripts/build_rhc03_local_5080_receipt.py` writes the receipt and manifest.
