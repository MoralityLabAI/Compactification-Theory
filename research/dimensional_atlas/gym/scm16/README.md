# SCM-16: Head-fiber Atlas-Transport VPD and exact local-maximum math gym

This is the continuation of SCM-15, with the same strictly weight-tied physical transformer block, shared TopK SAE and external exact EvalMath oracle. SCM-16 adds a **deliberately rank-deficient final linear readout**, two parameter-fiber sectors, matched ablations, three-seed training and a discovery-only exact DSL landscape scan. Its source lives in `research/dimensional_atlas/gym/scm16/` so SCM-15's results and interface stay untouched.

## Geometry

After training the tied baseline, freeze the final two-layer **linear** readout `W2 W1`, where `W1` has shape `[r, d]` and `r<d`. Let `Q` be an orthonormal basis of the row space of `W1` and set `P=QQ^T`. Its kernel fiber is `ker(W1)`, containing `(I-P)h`. Using the shared TopK SAE code `a_v` at recurrent visit `v`, define

```
raw_delta_v(h) = (((h @ V) * tanh(R a_v) * tanh(g_v)) @ U) * s
visible_delta_v = tanh(k_v,visible) * P(raw_delta_v)
fiber_delta_v   = tanh(k_v,fiber) * (I-P)(raw_delta_v)
```

The `joint`, `visible_only`, and `fiber_only` arms use these same trained weights. The shuffled-visit and zero-SAE-code arms control for visit identity and sparse routing. The tied baseline disables the VPD adapter. All arms share a **single trained physical block**, and all use the same rank-limited final readout. The adapter changes the effective operation at successive visits, so it is a visit-conditioned residual VPD *analogue*, rather than a decomposition of pretrained weights in the Goodfire implementation.

The readout fiber is an **exact null subspace of the direct final linear readout**. Earlier fiber perturbations can influence logits by passing through a subsequent recurrent block. The package records a matched-norm fiber/visible patch at each visit on a discovery task. These finite-float diagnostics are causal interventions on the toy model, without claims about a pretrained foundation model or intrinsic Seifert surfaces.

## Exact math, local maxima and leakage controls

The training task is to rank one of 64 three-token algebraic programs from two public input/output demonstrations and a query. Exact integer verification is owned by EvalMath, pinned to commit `88bb6cf8d2fe172e37d25964dfb2c285d9033841` and SHA-256 `c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b`. A self-contained *mod-17 toy* backend supports local smoke tests and has distinct algebra.

For discovery case `x` and candidate program `g`, let `S_x(g)` be the earliest disagreement coefficient across two demonstrations, with value `13` when both observed prefixes pass. The neighborhood consists of the **nine Hamming-one edits** of `g`. A selected local maximum satisfies `S_x(g)>=S_x(g')` for every neighbor, with at least one strict inequality. Selection prioritizes wrong-program maxima with an actual first fault; accepted-prefix aliases are labeled separately. The discovery scan charges **64 exact verifier queries per case**, even though the model's own inference uses at most 12.

Each selected discovery cell `(family, topology, source program)` generates disjoint replay-training tasks from fresh seeds and further untrained challenges from a separate seed range. The official already-public EvalMath evaluation roster is first imported after all trainable parameters and selection decisions are frozen. Source programs and future query coefficients for newly generated challenges remain in a local private sidecar. The committed public records contain only problem inputs, generator-provenance labels and hashes. The group of finite-prefix collision pairs in EvalMath is explicitly nonindependent.

## Training and runs

From the root of Compactification-Theory, with PyTorch 2.10 CPU:

```bash
python -m unittest discover -s research/dimensional_atlas/gym/scm16/tests -v
python -m research.dimensional_atlas.gym.scm16.campaign \
  --backend toy --output /tmp/scm16-toy.json --seed-count 3 \
  --base-steps 120 --sae-steps 60 --vpd-steps 180 --replay-steps 60
```

For the independent EvalMath backend, checkout the pinned external repository in a sibling directory and pass the original source file:

```bash
python -m research.dimensional_atlas.gym.scm16.campaign \
  --backend evalmath \
  --evalmath-source ../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py \
  --output /tmp/scm16-evalmath.json --seed-count 3 \
  --base-steps 120 --sae-steps 60 --vpd-steps 180 --replay-steps 60
```

The command refuses overwriting the output or either generated sidecar. EvalMath's held-out future is consumed only during final grading. Each seed trains 420 optimizer steps (120 tied base, 60 shared SAE, 180 joint fiber-VPD, 60 frontier replay). The program tracks exact paid verifier queries and separately charges the 64-program discovery searches. Extra adapter FLOPs and the training-time exact landscape scan make the arms **unequal in total compute**. All model intervention results are exploratory pending independent task-family replication and checkpoint transfer.

## Source and results

- [`PROTOCOL.md`](PROTOCOL.md): frozen SCM-16 evaluation contract, acceptance rules and protocol-change log.
- [`RESULTS.md`](RESULTS.md): local execution receipts and evidence boundaries.
- [`results/scm16_toy_3seed_receipt.json`](results/scm16_toy_3seed_receipt.json): complete public research receipt from the mod-17 backend; contains no new-eval private gold.
- [`results/scm16_toy_generated_public.json`](results/scm16_toy_generated_public.json): source-selected, disjoint new evaluation questions, with gold stored out-of-tree.
- [`results/scm16_evalmath_3seed_receipt.json`](results/scm16_evalmath_3seed_receipt.json): hosted three-seed pinned-EvalMath receipt on the already-public 192-case roster.
- [`results/scm16_evalmath_generated_public.json`](results/scm16_evalmath_generated_public.json): 16 fresh EvalMath-derived public challenges, without their private gold.
- `verify_receipt.py`: read-only source, readout-kernel and public-challenge integrity auditor.
- `fiber.py`: adapter and exact kernel, including stage-specific state patches.
- `landscape.py`: independently executed exact program-neighborhood tests.
- `campaign.py`: learning and EvalMath orchestration with fixed holdout boundary.

**Hosted EvalMath result (2026-11-14):** tied 235/576, joint VPD 237/576 and shuffled visits 235/576; the 16 fresh challenges tied at 31/48 across three seed-task evaluations. The public exact-symbolic 64-program baseline reached 163/192 with a larger verifier budget. See [SCM-16 hosted CI run #6](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43565693173) and `RESULTS.md`. These exploratory numbers do not establish a capability gain.

SCM-05–14's braid and curvature bounds are **not integrated** into this training objective yet. The atlas probe is a finite-difference diagnostic; it makes no certified braid or four-dimensional isotopy claim. Future work can export the trained RMP `C3` checkpoints and transfer this intervention design to real looped models.
