# Dynamical closure under coarse-graining: a finite-state math sprint

## Task-preserving extension

[TASK_CLOSURE.md](TASK_CLOSURE.md) gives the constructive finite-state companion to the existing EvalMath behavioral quotient. It proves a persistent fixed-target loss threshold, derives the minimum state-code bit budget, and supplies a nonconstant task-preserving reclosure counterexample. The implementation in [task_closure.py](task_closure.py) constructs minimal quotients and returns intervention words witnessing lost task distinctions.

The extended suite has 14 tests. Its controlled-system audit covers 5,898 systems and 29,290 candidate partitions. [receipt.json](receipt.json) records the executed checks and source hashes. The original four tests below were rerun against their fetched GitHub blob hashes.

## Theorem 1: exact deterministic closure

Let X be finite, F:X→X a deterministic transition and C:X→Z a surjective coarse-graining. There exists an autonomous macro transition G:Z→Z satisfying C∘F=G∘C **if and only if**

    C(x)=C(y)  =>  C(F(x))=C(F(y)).

*Proof.* If G exists, equality of C(x),C(y) implies equality after applying G. Conversely the displayed implication makes G(z):=C(F(x)) independent of the selected representative x with C(x)=z. This G is unique on C(X). QED.

This is a **partition congruence** condition. A macroscopic state may be informative about a target while failing to support autonomous dynamics.

## Theorem 2: minimum squared-error closure risk

Let A:X→R be a scalar macro-observable, X be a random microscopic state and Y=A(F(X)). For every measurable g with finite second moment,

    E[(Y-g(A(X)))²]
      = E[Var(Y | A(X))]
        + E[(E[Y | A(X)]-g(A(X)))²].

Thus the unique optimal macro predictor up to null sets is g*(a)=E[Y|A(X)=a], and its residual risk is

    R(A) = E[Var(A(F(X)) | A(X))].

If every microscopic state has positive probability and X is finite, R(A)=0 if and only if the partition induced by A satisfies Theorem 1. Under a restricted empirical support this equivalence holds only on that support.

*Proof.* Expand the square around conditional expectation; the cross term has zero expectation by the tower property. Finite conditional variance vanishes exactly when Y is constant on every positive-mass fiber of A. QED.

## Counterexample: closure is not monotone under coarse-graining

Take X={0,1,2}, F(0)=2, F(1)=0, F(2)=2. Consider

    A₀=(0,1,2)  [identity]
    A₁=(0,0,1)  [merge states 0 and 1]
    A₂=(0,0,0)  [constant]

A₀ is closed. A₁ aliases states 0 and 1 although their next macrostates are respectively 1 and 0; closure fails. A₂ is closed with its trivial constant dynamics. Under uniform state sampling, R(A₁)=1/6 while R(A₀)=R(A₂)=0.

Consequently the first failure index t* is a **path-dependent threshold**, rather than an irreversible information-loss phase transition. Information about the microstate (e.g. Shannon mutual information I(X;A_t(X))) is non-increasing under deterministic successive coarse-graining, while autonomous closure risk need not be monotonic.

## Interpretation for mechanistic interpretability

Low prediction error on observational trajectories can hide failures under interventions and distribution shifts. A useful experimental extension should preregister (i) a coarse-graining path, (ii) a state distribution, (iii) a specified target observable, (iv) a causal intervention family, and (v) a fixed error tolerance. Restricting the macro-model class also changes the measured threshold.

The constant A₂ example motivates a **nontriviality constraint**, such as task-relevant predictive information, to prevent trivial macrostates from being declared a successful interpretation. The extension shows that preserving a nonconstant task still permits reclosure; it treats behavioral loss and autonomous closure separately.

For stochastic Markov chains, exact strong lumpability requires equality of total transition probability into each coarse fiber for every pair of microstates in the same fiber. The current executable tests concern deterministic finite-state systems only.

## Reproduce

    python -m unittest discover -s research/dynamical_closure -v
    python -O -m unittest discover -s research/dynamical_closure -v

Standard-library Python only. The baseline tests exhaust every deterministic transition and partition for 1–4 states. The controlled extension exhausts two-action systems with binary tasks for 1–3 states and fixed-target risk checks through four states. No GPU/model access required.
