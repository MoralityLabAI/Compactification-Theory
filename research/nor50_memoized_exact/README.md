# NOR-50: memoized exact-native verifier mechanism test

The [preregistered protocol](REGISTRATION.md) was frozen in commit `a3b7a211e312235f8e9363612aff4173f50a3a4e` before first source generation or timing. It continues the archived [NOR49 historical 3/5 engineering FAIL](../nor49_reciprocal_count_dispatch/RESULTS.md).

The key mechanism is repeated exact reciprocal `R` execution. NOR45's original executor separately calls the same quadratic inverse for base, u, v, and active hidden suffix. NOR46/47/48 trace execution deduplicates each distinct unary transform. [cached.py](cached.py) adds an exact per-call memoized native control with the identical output formula and Merkle root. [test_cached.py](test_cached.py) uses only sixteen original NOR48 training-source programs and injects a local unary counter to verify exactly one invocation per distinct active operator.

The source-only 64-case [cohort.py](cohort.py) derives entirely new SHA-disjoint RRC25-native public input identities using replicate blocks 200–215 and OFFSET=140050000. Its public-only first fixture is upstream of any true hidden-program recovery or CPU timing. The separate [benchmark_cpu.py](benchmark_cpu.py) then compares complete output+root timings for NOR45 original, NOR50 cached, NOR47 reentrant GMP trace, and unchanged NOR49 count-R dispatcher over five interleaved repetitions per source.

The five frozen engineering gates quantify multi-R caching savings, remaining GMP advantage over cached native, low-R caching overhead, exact vector/root/security parity and source-first reproducibility. Conditional R count medians are primary; matched-workload total timing and 16-block bootstrap are predeclared secondary. Shared signing/network costs are excluded equally from each timed arithmetic core, with the prior pinned authority/session/provider controls run as separate CI checks. Full exact traces are nonsuccinct, and the trusted reference provider is an external assumption.

First CPU evidence must be archived byte-for-byte regardless of PASS or FAIL. The workflow contains a marker-gated source-first and CPU stage, avoiding first-source timing until predata CI and the preregistration are in place.

**First original source-and-CPU trigger:** the source-free trained-program unit and signed-control suite passed on the preregistered code. This commit starts the frozen 64-source public-only roster job, with its dependent first CPU run. Preserve the raw original result regardless of the five engineering gates.
