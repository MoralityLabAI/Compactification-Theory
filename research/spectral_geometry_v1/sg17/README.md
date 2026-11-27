# SG-17: chain-polytope LP and exact-separation constraint generation

Synthetic source-bound spectral routing, additive integer utilities and fixed nonnegative coordinate gaps, existing d<=32 and T<=16 validation. The unchanged SG-15 integer certificate verifier is the trusted arithmetic path. Both numerical LPs are optional proposal generators in the research gym; existing default and core dependencies are unchanged.

## Chain LP

Let y_ij indicate a read and h_ij indicate a retained label before that visit. The LP maximizes sum u_ij*y_ij minus lambda*sum b_j*(x_i-x_(i-1))*h_ij subject to sum_j y_ij=k_i, y_ij<=h_ij, h_(i+1)j<=h_ij and 0<=y,h<=1. With a binary schedule, minimal feasible h is exactly the suffix OR of reads; its price telescopes to the last-consumer price. Nonnegative gap prices ensure a minimal h is always sufficient. For one label, any fractional point in the order polytope is the average over uniform threshold theta in [0,1] of its binary indicators 1{coordinate>=theta}; each such pair satisfies all per-label order inequalities. Thus the per-label relaxation is the convex hull of binary feasible read/retention pairs. Relaxing the row equalities with multipliers eta and optimizing labels independently recovers the SG-15 dual bound. LP strong duality proves equality of the optimal LP relaxation values. SciPy minimizes the negative objective, so the budget RHS marginals are negated before reconstructing candidate rational eta. Numerical equality/status is diagnostic only; every accepted bound is recomputed in scaled exact integers. Row coupling can still cause an integrality gap (triangle-cover fixture).

For M=d*T, the chain LP has 2M variables, d*(2T-1) inequalities, T equalities and d*(5T-2) matrix nonzeros including equality rows, excluding variable bounds. SG-15 direct epigraph has T+T*d+d variables and d*T*(T+7)/2 nonzeros. At d=32,T=12: 1856 vs 3648 nonzeros; T=16: 2496 vs 5888. Chain LP has more variables, fewer nonzeros; no claim of a universal complexity or timing advantage.

## Constraint-generation comparator

The restricted dual has T+d variables (eta free, w>=0), and w_j >= sum_{i in mask}(u_ij-eta_i)-price(last(mask),j). Initialize each component's columns from a verified legal incumbent, the full mask and a maximizing root-vector mask. Including the feasible incumbent's masks guarantees a finite lower bound on the restricted dual, even with free multipliers. At each numerical solve, reconstruct rationals and compute the exact maximizing mask for every component in O(dT), by enumerating last uses with cumulative positive adjusted utility. Add absent maximizing masks, retain the tightest exact full bound seen. Empty masks are represented by w>=0. Stop at the round cap, a numerical failure, no new masks, or a bound whose floor matches the initial incumbent. No new masks is not itself an optimality certificate. This implementation rebuilds and solves the LP each round, without simplex-basis reuse. Any returned vector yields a sound bound. The campaign uses SG-15 exact face recovery and legal rounding; if necessary it falls back to the original LP and then SG-12 block search. No numerical solver status certifies optimality.

## Reproduction

Install DataGoblin and optional NumPy, SciPy, pytest. From repository root:

    export PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient:training/rlt_gym/sg15_spectral_face:training/rlt_gym/sg17_spectral_chain
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q training/rlt_gym/sg17_spectral_chain/test_sg17.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg17_spectral_chain/sg17.py

Four targeted tests passed: 64 separation fixtures (2048 mask objective evaluations), 64 chain/direct exact dual parity fixtures checked against exhaustive integer optima, 32 additional tiny tasks with two capped cut proposals and four complete acquisition arms (128 complete runs), fractional triangle-cover fixture. The isolated tests were rerun successfully. The eight directly relevant dependency blob hashes match the current SG-16 branch. Full repository tests and Ruff not run; Rust and Lean not compiled.

## Campaign

36 fresh synthetic development roots, 12 per persistent/rotating/mixed regime, d=32, half T=12 and half T=16, alternating read counts 7/8. Two randomized arm orders per root; CPU time is the mean of each root's two-run median, includes shared root-vector and feasible incumbent construction, all proposed LP solves, recovery, exact certificate checking and final receipt, excludes post-hoc rechecks. All 288 evaluation runs checked against the SG-15 integer verifier; cross-policy intervals intersect at an exact optimum on every root. No independent exponential oracle at d=32. No LP solver warm start. No matched deadline; compare end-to-end CPU at equal final certificate quality. No paid inference, GPU or native model claims.

| Arm | Mean CPU ms | Certified /72 | Primary certified /72 | Original-LP fallback | Solver calls | Paired CPU wins /36 |
|---|---:|---:|---:|---:|---:|---:|
| direct |19.126868|72|72|0|72|0|
| chain |12.726726|72|72|0|72|36|
| cuts8 |41.628217|72|32|40|596|1|
| cuts2_chain |20.646398|72|72|0|214|7|

No block-search fallback on evaluation roots. The cuts8 arm had 50 capped cut stages, 14 floor stops and eight no-new-mask stops; its final primary certification count includes feasible repair after the cut stage. Cuts2_chain includes up to two cut rounds before chain LP; 2/72 runs certified without the chain LP. The chain arm has 72 exact primal/dual equality certificates; cuts8 has 68 exact-face and four floor-only certificates. Mean CPU savings chain vs direct 33.46%, wins all 36 roots; T=12 mean 15.889618 vs 10.733794 ms, T=16 mean 22.364118 vs 14.719657 ms. Cut matrices are smaller per solve but repeated solver overhead dominates. Recommend chain LP as next deterministic benchmark, keep cut generation experimental and existing production defaults unchanged. Single host, two timing repetitions, fresh exposed development (not a sealed confirmation cohort). Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0; each numerical solve has its own 2s time limit, not an end-to-end cap. highspy not available; no direct warm-started native HiGHS experiment.

## Evidence custody

Source blobs sg17_lp 83f1121f03ad990a6d87c13e2b78927ee38ed52b; sg17 730f7eba133afe2586180b2e1937d50397a56020; tests 646ae9b402ad62c4aff0a74b2a53fa858094a653. Raw results JSON SHA256 4eebf3feab37adc7a27bb885443c3340fe1c512b4a9a0203080ea9a0a5ebc73c. Generated raw runs supplied separately. Next: reduce the cost of obtaining useful rational multipliers with a compact exact combinatorial primal/dual proposal, or compare HiGHS basis reuse for related tasks under matched CPU budgets; keep integer verification independent.

## Implementation custody

[DataGoblin campaign and tests](https://github.com/MoralityLabAI/DataGoblin/tree/6ffffe2aa01e632387ddd7b4907418768f5102e4/training/rlt_gym/sg17_spectral_chain) · [Chain and cut LP proposals](https://github.com/MoralityLabAI/DataGoblin/blob/6ffffe2aa01e632387ddd7b4907418768f5102e4/training/rlt_gym/sg17_spectral_chain/sg17_lp.py).

Complete archive `sg17_cpu_campaign.zip`, SHA256 `d9ce8026150e88e441dd951a0b8d2da6516e628b1805fe5ce31e28ba871d6de3`. Contains 26 files, including executed sources and dependency copies, raw results, logs and a SHA256 manifest. Separate post-run integer replay checked all 288 final receipts and 398 candidate receipts; all four policies' certified intervals coincide on every one of 36 roots. Archive integrity check passed. Raw timing data are supplied separately. All three committed Python source blob hashes match the executed files.
