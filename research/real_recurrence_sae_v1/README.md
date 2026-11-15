# SMC-07: Source-locked recurrent SAE and causal intervention audit

**November 15, 2026.** Begins the next experiment after merging [SMC-01–06](https://github.com/MoralityLabAI/Compactification-Theory/pull/145) and [OLM-01–06](https://github.com/patrickdugan/MathCan/pull/62).

## Model and data provenance

This experiment uses the repository's existing third-party **44,001-parameter, TRM-inspired pretrained Sudoku recurrence**, rather than the synthetic K6 matching teacher from SMC-06. The source comes from [LSV-05](../levin_sieve_v1/pretrained_trm/README.md).

- Upstream: NagaSatyaPhanindraVallabhaneni/tiny-recursive-model
- Source commit: f0406195fece4d0db29ced28f3a69d1993d95b1e
- Model checkpoint: models/trm.pt, exactly 181,147 bytes, SHA-256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1
- Architecture: TinyRecursiveModel(d_model=48,n_steps=4), a shared token/channel mixing block iterated four times across 81 Sudoku tokens
- Source-locked 64-puzzle roster SHA-256: eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611
- Checkpoint deserialization uses restricted torch.load(weights_only=True), strict architecture and hash assertions.

The checkpoint was pretrained by its original author. The SAE introduced here trains only on frozen hidden states. This is distinct from the official 7M Tiny Recursive Model and from Qwen.

The split is fixed by puzzle ID modulo 4: 32 training puzzles (IDs 0,1 mod 4), 16 validation (2 mod 4), 16 heldout (3 mod 4). **All 81 tokens of any puzzle remain in its split.** SAE fitting uses stage-one states from training puzzles, validation selects the SAE checkpoint, and heldout states are used only for measurement. Sudoku targets enter neither SAE fitting nor validation.

## SAE and baseline protocol

A 48→96→48 ReLU autoencoder retains the top four positive activations in every encoding. The unsupervised loss is train-standardized reconstruction MSE plus 1e-4 times mean absolute sparse activation. It is trained by AdamW for 300 updates. The best checkpoint uses validation MSE at steps 1,25,50,...,300. Seeds **71,89,113** are frozen.

Only training-state means and standard deviations define the normalization. Heldout reconstruction MSE is measured at all four recurrent depths using the **same depth-one SAE**. Four-component PCA and a seeded four-dimensional random orthogonal projection provide controls. Their per-sample dimension is matched to four active SAE coefficients, but the SAE has a larger dictionary and parameter budget. Do not infer an equal-capacity comparison.

For feature transport, record the same-token top-four activation overlap between depth one and depth two, with a null that swaps puzzle identity at fixed token position.

## Causal interventions

For each of 16 heldout puzzles, select the first blank Sudoku token. Freeze its stage-one recurrent state and continue the teacher through three further iterations.

- Reconstruct that single goal token with the SAE or PCA4, then measure final goal logit MSE, teacher-label agreement and Sudoku target correctness.
- At fixed L2 radii 0.2 and 0.8, perturb the goal token along the largest active SAE decoder feature. Compare matched-radius directions from an inactive SAE atom and seeded isotropic random vector.
- Apply the active SAE direction at a different token, measuring its effect on the original goal token's output. Record changes in final logits, fixed clean top-two margins and prediction flips.
- At the separate **stage-two pre-LayerNorm2 bus**, apply a per-token channel-common shift in FP64, which LayerNorm should erase. Compare against a centered channel perturbation with matched norm.

The gauge control checks an exact symmetry at a specified site. It does not imply recurrence-wide invariance to arbitrary shared shifts.

Every result remains linked to its independent puzzle ID. A positive SAE causal-sensitivity claim would require a paired uncertainty analysis and appropriate controls; the raw difference between mean scores is descriptive.

## Falsification and limits

If SAE reconstruction falls behind PCA4, report it. If cross-depth feature support overlaps the puzzle-shuffled null at comparable rates, report a null result for depth-stable feature labels. If SAE-aligned interventions yield no reproducible difference from matched inactive and isotropic directions, report that finding. FP64 LayerNorm gauge violation fails the implementation control.

The K6 star-cover certificate from the synthetic SMC-06 experiment is **not imposed** on Sudoku activations. The model may not represent anything like a perfect-matching geometry.

## Reproduction

Run from the repository root with CPU-only Python 3.11 and PyTorch 2.7.1:

    python -m unittest discover -s research/real_recurrence_sae_v1 -p 'test_*.py' -v
    python research/real_recurrence_sae_v1/smc07_real_sae.py --fetch --max-cases 16 --steps 75 --seeds 71 --out runs/smc07/pilot.json
    python research/real_recurrence_sae_v1/smc07_real_sae.py --fetch --max-cases 64 --steps 300 --seeds 71,89,113 --out runs/smc07/full.json

Network access is opt-in with --fetch. Alternatively supply --checkpoint /path/to/trm.pt; the exact Git blob SHA-1, SHA-256 and size are still checked. The [SMC-07 workflow](../../.github/workflows/smc07-pretrained-sae.yml) runs both pinned jobs and uploads per-puzzle receipts.

**Evidence status:** This file registers the protocol. Scientific conclusions require completed, inspected, source-locked hosted receipts. The local toy-model unit tests verify harness plumbing only.
