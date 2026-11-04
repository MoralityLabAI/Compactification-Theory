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

## LSV-02: certified delayed observation

The [bounded-staleness theorem](DELAYED_REPAIR.md) extends the fair asynchronous repair guarantee to messages delayed by at most D steps. An aligned B-fair schedule has certified superblock length L=B*(1+ceil(D/B)); the usual small-gain contraction applies at each complete superblock. The [delayed checker](delayed.py) audits individual stale reads and the finite-time bound using exact rational arithmetic; [tests](test_delayed.py) cover a sharp scalar witness and a changing-delay network. The theorem is a written proof with finite CPU tests, awaiting any Lean formalization or neural validation.

## LSV-04: learned-checkpoint repair

The [checkpoint-certified neural repair experiment](NEURAL_REPAIR.md) trains a CPU-only two-parameter nonlinear recurrent network, exports a rational checkpoint, and derives global error bounds from the frozen weights. Its 128-task held-out development roster tests clean and two-step-stale recovery with exact rational auditing and untrained, swapped-weight, zero-coupling, and oracle controls. The learned model achieves 128/128 tasks within absolute 0.2 error by 8 clean cycles or 16 stale cycles; the written certificate also guarantees those bounds for the declared domain. See [code](neural_bridge.py), [benchmark](benchmark_neural.py), [checkpoint](checkpoints/repair_two_node_20261104.json), and [receipt](results/repair_two_node_20261104.json). This is a constrained two-node proxy, not a pretrained TRM/LLM result or a Lean kernel proof.
