# LSV-01: Levin-inspired sieve for applied mathematics

**Status: 2026-11-04.** Exact-rational CPU implementation, ten unit tests, five written mathematical results, and an evaluation protocol. No GPU resources required.

Michael Levin's multiscale agency and goal-directed morphogenesis motivate a **question-selection prior**: prioritize precise mathematical problems about local components preserving or recovering a globally declared goal under interventions. The validity of the results depends on their explicit mathematical hypotheses, not on a philosophical assumption about biology or consciousness.

Existing repository research on [ingression dynamics](../ingression_dynamics_v1/THEORY.md) and [goal recovery](../ingression_dynamics_v1/GOAL_RECOVERY.md) studies quotients and goal visibility. LSV-01 adds asynchronous repair under bounded coupling and disturbances.

## The mathematics

[THEOREMS.md](THEOREMS.md) contains full written proofs:

1. A geometric finite-time goal-recovery bound under fair asynchronous local updates and weighted small-gain constraints.
2. A constructive exact-rational certificate from the resolvent (I-A)^(-1)*1 whenever a nonnegative rational gain matrix has spectral radius below one.
3. A complete two-agent feedback-cycle stability criterion ab<1, with limiting optimal contraction rate sqrt(ab).
4. A common-envelope guarantee for arbitrarily switching interaction modes, with an exact counterexample where individually stable modes cause explosive switching.
5. A Jacobian-envelope lemma connecting differentiable repair rules to the local componentwise contract.

These use known mathematical tools and do not claim new general theorems. Current evidence is a written proof and exact-rational finite computational tests, with no Lean audit or neural/bio benchmark.

## Reproduce

From the repository root on Python 3.10+:

    python3 -m unittest discover -s research/levin_sieve_v1 -p 'test_*.py' -v
    python3 research/levin_sieve_v1/small_gain.py
    python3 research/levin_sieve_v1/benchmark.py

The certificate checker [small_gain.py](small_gain.py) synthesizes nonnegative rational weights, validates the matrix inequalities, computes the disturbance tube, supplies sufficient fair blocks to a stated tolerance, and audits finite update traces. It can form a conservative common gain envelope for arbitrarily switching modes. Actual deployment requires a separate proof of the local gain bounds on all allowed states.

The demo uses two components with gain matrix [[0,3],[1/10,0]], noise at most 1/100 per update, and alternating local updates. Weights (11/2,1) certify rho=11/20 although the ordinary maximum row sum is 3. The asymptotic radius is 1/45.

## Proof-coverage baseline

[benchmark.py](benchmark.py) enumerates 36 two-component nonnegative gain matrices with a,b chosen from {0,1/10,1/2,9/10,2,4}.

| Criterion | Exactly certified cases |
|---|---:|
| Ordinary unweighted sup-norm contraction | 16/36 |
| Weighted rational small gain | 24/36 |
| Additional cases reached by weighting | 8/36 |

The 12 other cases have ab>=1 and fail strict weighted small gain. These counts compare mathematical certificates on a selected toy family. They measure neither previously unknown mathematics nor real-world model improvement.

## The philosophy-sieve experiment

[PROTOCOL.md](PROTOCOL.md) describes a blinded, cost-matched comparison of Levin-inspired and neutral problem generation, with independent proof review, novelty checking and held-out applied benchmarks. The present known-theory example is a positive control, excluded from the future comparison.

## Next questions

**LSV-02, stale observations:** derive a bounded-delay version of the fair-schedule guarantee and find failure cases at long delays.

**LSV-03, switching topology:** move beyond the conservative elementwise-max gain envelope, seeking tighter common weights, mode-dependent storage functions and switching-dwell guarantees.

**LSV-04, recurrent neural repair:** bound Jacobians on a declared invariant region around a frozen TRM checkpoint, perturb hidden states and test predicted recovery/abstention against actual task utility.

**LSV-05, embodiment transfer:** test whether an exact goal-visible quotient lets a certificate found in one simulated morphology transfer to another under matched disturbances.
