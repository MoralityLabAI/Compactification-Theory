# LSV-18: learned semiconjugacy with a local-state obstruction

**Date:** November 4, 2026. **Evidence:** frozen executable protocol; actual checkpoint results are reported separately from synthetic tests. No exact neural semiconjugacy theorem is claimed.

## Mathematical target

The frozen teacher has a full 81x48 hidden state and a shared recurrent map R. A goal-conditioned 16-dimensional encoder q, latent transition F and decoder d aim to satisfy

    q(R(h)) ≈ F(q(h))
    d(F^k(q(h))) ≈ head(R^k(h))[cell], for k=1,2,3.

Training requires future-output reconstruction, three-step rollout, one-step q/R consistency and a noncollapse variance penalty. Report a scale-normalized closure defect, since scaling q toward zero can artificially reduce raw transition loss.

## Locality trap

The token mixer couples all 81 tokens. If h and h+δ differ only at a remote cell, they have exactly the same selected goal token. A function q of only that goal token must output the same macrostate for both. If future goal readouts differ, no exact deterministic local q/F/d can satisfy both. Our empirical off-site test records a finite-pair witness; a global theorem would require assumptions on the checkpoint weights.

The two observation maps are local goal-token coordinates (48 dimensions), and goal token plus global/row/column averages (192 dimensions). Both compress to 16 dimensions. The context map is still an incomplete summary of the full recurrent state, so any result is goal-conditioned and distribution-specific.

## Frozen experimental design

Use the existing source-locked 44,001-parameter TRM-inspired Sudoku model and 64 generator-fixed puzzles. First missing goal cell per puzzle, split by puzzle i modulo 4: train/train/validation/test. Per puzzle collect one clean state and eight edited states, using independent seeds 71/103 on train puzzles, 211/223 on validation and 307/311 on test, at L2 radii 0.5 and 2.0, sites goal token or remote token (cell+17 mod 81).

Freeze the teacher and compute each edited state's four consecutive full 81x48 states and corresponding goal logits. Four students use identical data and a width-16 macrostate: local linear, local nonlinear, context linear, context nonlinear. Nonlinear models use a 32-wide tanh network. Training uses train-only normalization, AdamW with 0.003 learning rate, 200-epoch ceiling, 35-epoch early-stop patience and deterministic seeds. The rollout starts at the stage-one macrostate and applies F three times with zero teacher refreshes.

Compare against stage-one output persistence and train-mean logits. Keep teacher-label agreement, Sudoku correctness, logit MSE, distribution TV, teacher-forced readout error, latent variance, relative q/R closure defect, parameter counts, and per-puzzle errors. Test puzzles remain unused for model selection. Multiple edited states per puzzle are correlated; the held-out effective sample size is 16 puzzle IDs.

## Interpretation

A successful model should beat meaningful persistence and constant baselines in held-out multi-step readout quality, while maintaining bounded normalized q/R closure error and noncollapsed latents. The total computational cost of computing q must be counted when comparing with the original teacher. The large-field Unique Games theorem motivates the quotient search without implying any neural compression result.

## Reproduction

    python -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p 'test_semiconjugacy_gym.py' -v
    python research/levin_sieve_v1/pretrained_trm/semiconjugacy_gym.py --fetch --max-cases 16 --max-epochs 120 --out runs/lsv18/pilot.json
    python research/levin_sieve_v1/pretrained_trm/semiconjugacy_gym.py --fetch --max-cases 64 --max-epochs 200 --out runs/lsv18/full.json

All runs are CPU-only and source-hash locked. The full receipts identify model and roster fingerprints.
