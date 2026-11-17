# PSS-01 GPU workorder: protected loop surgery on Qwen3.5-9B

Prepared proposal. No GPU allocation has been launched from this repository.

Work in MoralityLabAI/Compactification-Theory, `research/weight_surgery_pss01/`. Read `README.md`, `surgery.py`, and existing `research/algernon/mono01/` before writing additive experiment code. Keep unrelated branches and results intact. Commit the experiment receipt and raw-record index. Record failed and null runs.

## Compute

Use one H100 80GB and Qwen/Qwen3.5-9B in BF16. Proposed hard cap: 6 H100 GPU-hours, including tracing. Reserve at least half the allocation for confirmatory generation. This workorder allocates GPU work when the owner submits it for execution. Stop at the cap and preserve partial artifacts. Small 0.8B reproduction belongs to local work and is outside this GPU allocation.

Pin model/tokenizer revisions and the software environment. Verify the installed Transformers version supports the exact checkpoint; the repository's older root environment is not the deployment specification. Enumerate actual linear receiver modules and their inputs before choosing hooks. Qwen's architecture differs from Monad's; select layer/module locations from this model's measured writes. Reuse compatible existing SAE checkpoints only with matching model revision and hook point.

## Data freeze and pilot

Hash dataset-qualified task-family IDs into design=128, validation=128 and sealed-test=256 prompts. Include mathematical reasoning and instruction-following tasks with deterministic graders. Lock dataset revisions and the split manifest before any fitting. Keep paraphrases and every seed for a task in the same partition. Use `assert_disjoint_ids` plus a dataset-level family check. Store a SHA-256 of every raw prompt and source record.

On design, run the base model with greedy decoding and max_new_tokens=4096. Record exact token IDs, EOS/think termination and repeated spans. A repetition candidate requires the final suffix to contain at least four exact repetitions of a period of 1..64 tokens, with at least 64 repeated tokens total. Manually audit flagged design examples and freeze the detector before validation. Preserve separate labels for truncation, repetition, malformed answer, and wrong answer. A completed repetition required by the task is legitimate behavior and needs a protected label.

If fewer than 16 design examples exhibit pathological loops, stop the repair claim and report low prevalence. Preserve the diagnostic dataset; spend the remaining allocation only under a separately approved question. Do not manufacture pathological prompts or switch to a tiny model to make the edit look effective.

## Trace and construct

Collect receiver inputs and branch writes around pre-loop and in-loop positions. Compare position-, token-length- and task-matched successful examples, with per-prompt weighting. Keep first-token sink statistics separate. Fit directions on design only. Estimate direction stability with a prompt-level bootstrap.

Build S from reproducible sink directions and measured protected task-output features. Build H from a low-rank basis of successful receiver-input activations. Record retained failure energy, singular spectra and the Q^T W H feasibility floor. Include SAE-derived candidate directions only when their checkpoint/hook alignment is verified; compare to the plain mean-difference baseline.

Causal localization occurs on design/validation: intervene at candidate receiver writes and test both behavior and protected utility. High classification AUC alone supplies a detection result. Limit selected layer bundles to at most four receivers. Screen ranks 1,2,4,8 and strengths 0.25,0.5,1.0 with cheap teacher-forced diagnostics, then generate at most four finalists on validation. Freeze the winning configurations and every control before opening test data.

## Arms and implementation controls

Use the same frozen test prompts for: base; unprotected direction edit; protected-output-only edit; two-sided protected edit as unfused factors; the same two-sided edit merged to BF16; and a random protected-subspace control matched in rank, layer locations and realized Frobenius update norm. Freeze three random-direction seeds and report each separately within the compute cap. Controls requiring scaled factors must record the effective scale and actual realized norm.

On a fixed teacher-forced slice, compare the branch hook y <- y + B(Ax) to unfused factors and merged weights. Keep original bias and skip path. Record numerical discrepancies before generation. A projection of the entire residual stream is a separately named intervention. Trace activations for base and the selected repair on a frozen subset to measure whether the proposed failure feature falls while protected coordinates remain stable.

Run greedy as the primary decoding regime. Use temperature=0.7, top_p=0.9 with seeds 11 and 29 for a separately labeled sampling replication. Pilot throughput before unsealing. If the complete sealed matrix exceeds the remaining allocation, retain the sealed test and deliver a validation-only report. Never choose a favorable partial test subset after looking at outcomes.

## Scoring and release gates

Report loop rate and task accuracy with paired, prompt-clustered bootstrap intervals; seed-level samples of one prompt remain in one cluster. Give answer regressions and recoveries, EOS rate, output length, latency and peak VRAM. Plot repair against utility loss. Predeclare a two-percentage-point accuracy noninferiority margin, using the 95% lower confidence bound of paired accuracy difference. Label an underpowered result inconclusive.

Record changed stored entries separately from target tensor size, factor rank and numerical rank of the realized difference. Apply `audit` before and after BF16 merging, then log measured protected-feature drift and divergence of teacher-forced output distributions. Report all required failed constraints, including native precision differences. Algebraic preservation applies at a fixed linear receiver.

Deliver raw generation records, checksummed trace shards and exact configuration in the repository or a persistent artifact store linked from an index. Include a CPU-readable JSON receipt with model revision, split hashes, module keys, patch hashes, GPU-hours and a claim ledger marking each item as proved algebra, synthetic observation or measured model result. End the run with a short table of those three evidence classes and the next unresolved experiment.
