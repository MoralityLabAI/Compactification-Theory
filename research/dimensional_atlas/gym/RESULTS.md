# SCM-15 first CPU smoke receipt (November 14, 2026)

**Classification:** executed synthetic finite-ring mathematical program experiment. This source does not instantiate the EvalMath formal-series grader. Its 64-class operators and modular arithmetic are independent plumbing fixtures, not a benchmark result on EvalMath or Lean. No pretrained `LOOP-T`, `FF-U`, Qwen, or Goodfire VPD weights were used.

**Run:** Python 3.13, PyTorch CPU, seed 15, hidden 32, three tied visits, shared TopK SAE width 64 with k=6, rank-four AT-VPD, 160 training tasks, 40 discovery tasks, 48 periodic-family heldout tasks. Optimization: 40 base, 25 SAE, 45 AT-VPD and 20 discovery-generated replay steps; exact candidate budget 12. All four report arms reused the same frozen base checkpoint.

| Arm | Exact heldout future correct | Future accuracy | Mean paid prefix queries | False future acceptances |
| --- | ---: | ---: | ---: | ---: |
| Strictly tied base | 25/48 | 0.5208 | 9.25 | 0 |
| SAE-conditioned Atlas-Transport VPD | 25/48 | 0.5208 | 9.3333 | 0 |
| Reversed visit gates | 25/48 | 0.5208 | 9.3333 | 0 |
| Zero SAE codes | 25/48 | 0.5208 | 9.25 | 0 |

The discovered failure-ridge proxy produced four selected finite-cell maxima and eight freshly seeded public evaluation cases, with separate hidden gold sidecar. The eight generated evaluations were kept out of adaptation and frozen readout scoring.

**Exploratory dimensional atlas:** singular values of the two-direction finite-difference Jacobian at successive visits (task `discovery:220000`) were approximately `(2.046,1.669)`, `(3.931,2.438)`, `(6.103,2.841)`. Neighboring tangent-space alignment singular values were `(0.911,0.894)` and `(0.976,0.947)`. These depend on the fixed random/SAE readout axes and finite step `epsilon=0.01`. They are observational diagnostics, not certified tangent-rank lower bounds, braid invariants, or a causal mechanism claim.

**Decision:** the engineering gate passed, and the initial skill-gain gate did not. Next: rerun against the pinned EvalMath ARG-01 oracle, add a genuine source-level causal Jacobian and matched compute baseline, and evaluate generated difficulty families with independent seeds. Retain the null result and all controls.

Reproduce from the repository root:

```bash
python -m unittest discover -s research/dimensional_atlas/gym/tests -v
python -m research.dimensional_atlas.gym.run --backend toy \
  --steps 40 --sae-steps 25 --vpd-steps 45 --replay-steps 20 \
  --output /tmp/scm15_atlas_vpd.json
```

## Hosted pinned EvalMath ARG-01 smoke

**Executed on GitHub Actions:** November 14, 2026, workflow `SCM-15 dimensional atlas and math gym`, run #4, job `pinned-evalmath-oracle-smoke` (success). The job checked out `MoralityLabAI/EvalMath` at the pinned commit `88bb6cf8d2fe172e37d25964dfb2c285d9033841`. Source hash checking against `c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b` passed before training. The public EvalMath roster was loaded after the short adaptation and discovery phases.

This is a **minimal integration smoke**, with 24 freshly generated training examples, 12 discovery examples, two recurrent visits, three baseline optimizer steps, two SAE steps, three AT-VPD steps and two frontier-replay steps. The finite-query evaluator tested 192 already-public cases, including source-family shifts, negatives and intentionally paired observational-prefix collisions. Its cases are not all independent.

| Arm | Exact future correct | Prefix acceptance | False future acceptances | Mean verifier queries |
| --- | ---: | ---: | ---: | ---: |
| Tied baseline | 81/192 (42.1875%) | 94/192 (48.9583%) | 13 | 8.7083 |
| AT-VPD | 81/192 (42.1875%) | 94/192 (48.9583%) | 13 | 8.7135 |
| Reversed visit gates | 81/192 (42.1875%) | 94/192 (48.9583%) | 13 | 8.7135 |
| Zero SAE codes | 81/192 (42.1875%) | 94/192 (48.9583%) | 13 | 8.7083 |

The discovery-only proxy selected two local peaks and generated four fresh public challenge cases with their hidden labels kept in a separate runner-local sidecar. No improvement over the tied baseline was established. This run validates the externally pinned independent oracle integration, the exact paid-query accounting, and the generation interface.
