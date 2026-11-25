# DCF-15 results: LDT mesh routing

Setup:
- Repository run, 2026-11-25, one CPU process, no GPU.
- Held-out: 6 conditions (cross-coupling ρ ∈ {.2, .4, .6} × noise {low, high}); ρ = .6 never appears in development. 64 fresh episodes × 32 steps per condition (2,048 queries per router per condition), batch 32.
- Frozen before the held-out run (`PRE_TEST_SHA256`): kernel cost model, router thresholds, κ and the MLP.
- The batch sweep (256 episodes) overlapped in time with DCF-16 data generation on four other cores. Its routing-time column carries that load; kernel costs are modelled and unaffected.

## Gates

| Gate | Result |
|---|---|
| E1: the exact operator equals independent elimination over the relevant component; local equals exact without pending dependencies | PASS (`tests/test_mesh15.py`; HybridTRMLDT port checks exact against brute force over all 2^16 states) |
| P1: learned beats the strongest dev-selected symbolic router on objective_total in more held-out conditions than it loses | **FAIL: 0 wins, 6 losses** |

## Mean over the 6 held-out conditions

| Router | EU | Regret | Coverage | Selective risk | Kernel ms/query | Routing ms/query | Objective (kernel) | Objective (total) | Launches/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle (privileged regret) | .1195 | 0 | .221 | .104 | .054 | 7.30 | .1190 | .0460 | 1.08 |
| fixed exact | .1195 | 0 | .223 | .104 | .485 | .002 | .1147 | .1147 | 1.00 |
| **fixed BP** | .1165 | .0030 | .214 | .106 | .087 | .003 | **.1157** | **.1156** | 1.00 |
| symbolic (dev-tuned) | .1165 | .0030 | .214 | .106 | .087 | .498 | .1157 | .1107 | 1.80 |
| learned (MLP, 547 params) | .1130 | .0065 | .205 | .104 | .174 | .680 | .1113 | .1044 | 2.75 |
| random | .1052 | .0144 | .198 | .109 | .220 | .045 | .1030 | .1025 | 3.00 |
| symbolic J (κ·KL_hat) | .0808 | .0388 | .137 | .090 | .027 | .460 | .0805 | .0759 | .85 |
| fixed local (slow merges on) | .0727 | .0468 | .146 | .098 | .001 | .002 | .0727 | .0727 | 1.00 |
| fixed local, no slow timescale | .0517 | .0678 | .108 | .106 | .0004 | .002 | .0517 | .0517 | 1.00 |

Findings:
- **No router beats a fixed operator once routing is charged.** Fixed BP has the best total objective on average. Fixed exact is best in the two strongly coupled low-noise conditions (ρ = .4 and .6, low noise: .1936 vs .1935, and .1886 vs .1835). The best fixed policy therefore varies by regime, but no per-query router captured that.
- **The dev-tuned symbolic router collapsed to near always-BP**: 81.5% BP, 18.5% local, no exact. It matches fixed BP in utility but pays 0.5 ms of Python feature extraction per query.
- **The learned router selects more varied operators** (37% BP, 13% exact, 32% abstain). It loses utility (regret .0065 vs .0030) and pays more kernel time and routing time. Training cost is negligible (0.13 s, 0.01 ms per decision amortised). Against symbolic J it wins 6/6, so it beats a weak symbolic baseline.
- **The slow timescale earns its keep.** Merging the most correlated unit pair every 8 steps cuts the true discarded correlation `KL(p ‖ ∏ p_B)` from .359 to .229 nats (−36%). It raises local-answer utility by 41% (.0517 to .0727) and helps in all 6 conditions. The benefit grows with coupling: at ρ = .6, low noise, local utility rises from .034 to .083.
- **Exact requests ("global inference") are rarely worth their fixed cost.** The exact kernel carries a fitted 1.3 ms launch overhead (cost-model R² .75, the least accurate of the four kernels). At λ = .01/ms that overhead costs more than the .003 regret BP leaves.
- **The privileged oracle loses on total objective** (.046) because computing true regrets takes 7.3 ms per query. Even perfect knowledge of regret is not worth paying for at this granularity.

## Batching economics (ρ = .4, low noise, 256 episodes)

| Router | B = 1 | 8 | 32 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| fixed exact, kernel ms/query | 1.797 | .644 | .520 | .489 | .484 |
| symbolic, kernel ms/query | .101 | .093 | .092 | .092 | .092 |
| learned, kernel ms/query | .439 | .299 | .221 | .193 | .188 |
| symbolic, launches/step | 1.00 | 1.52 | 1.71 | 1.91 | 2.00 |
| learned, launches/step | .74 | 2.20 | 2.86 | 2.98 | 3.00 |

Batching amortises the exact kernel's launch overhead 3.7× from B = 1 to 256. A homogeneous fixed policy gets the full benefit, at one launch per step. Adaptive routers fragment the batch: the learned router launches three distinct kernels per step at B ≥ 32, so its overhead is shared across fewer episodes. Its kernel cost per query still falls 2.3× with batch size, but stays 2× the symbolic router's. Utility does not depend on B, because γ fragmentation charges never flipped a decision here.

## Verdict

The mesh interface works: units expose state, dependencies, uncertainty, operators, costs, verification and estimated information loss, and an exact escalation path is verified. What earns promotion in this simulator is structural, slow-timescale reorganisation (merging correlated units), not per-query learned routing. That repeats the DCF-10–14 pattern: compiled, homogeneous execution beats fine-grained adaptivity unless the routing decision is nearly free.
