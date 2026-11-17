# SCM-04 exact triple Milnor / compactification receipt

**Execution date:** 2026-11-13. **Scope:** CPU-only exact braid algebra and a mock-model capture interface. There are no trained RMP checkpoint or SAE/dictionary measurements in this experiment.

Run commands:

    python -m unittest discover -s research/seifert_activation/tests -q
    python research/seifert_activation/scm04_reference.py --out research/seifert_activation/results/scm04_cpu_reference.json

Results verified locally on Python 3.13, SymPy 1.14, Torch 2.10 CPU:

- **46/46 unit tests passed:** the 33 existing SCM-01/02/03 tests plus 13 new SCM-04 tests.
- The exact normalized Artin longitudes and the cubic nilpotent Magnus quotient independently return `mu_123(C)=+1` and `mu_123(C^-1)=-1`; the links have zero pairwise linking and normalized Alexander polynomial `(t-1)^4`.
- The 25 possible sample-frame phase cuts of the synthetic `C` braid have raw position-order triple linking `+1` or `-1`, whereas the original-token-order correction consistently returns `+1`. An odd initial x-rank permutation accounts for the raw sign reversal.
- The cubic quotient evaluates `C^1000` and `C^-1000` exactly, returning `+1000` and `-1000` with 6 nonzero cubic Magnus monomials per meridian in this family. The general representation has at most 40 monomial slots per meridian. This establishes exact low-degree *coordinate* compactification for the specified observable, with growing integer coefficient precision.
- 200 seeded short pure braids satisfy exact cyclic-coefficient consistency modulo the standard linking-number gcd. A separate 128-word test compares the fast degree-two mixed Magnus formula against direct noncommutative polynomial multiplication. Pure conjugation, nonzero-linking indeterminacy, invalid braids and explicit expansion-budget failures have their own regression tests.
- The RMP `analyze_cycle` adapter now records token-order-corrected Milnor measurements and `compare_double_resolution` reports a separate closure-invariant comparison alongside the existing based-Artin check. Equality of the **tested** closure invariants is only a necessary consistency check and does not certify link isotopy.

`results/scm04_cpu_reference.json` was generated without network access, and is immutable under the supplied script. SHA-256 of the file: `c3be674b464d74c66e2e7cc7e11013848c766656a7014586d245d6f218b872cb`.

**Honest limit:** no empirical evidence that a neural model computes Milnor invariants; even a positive projected-link observation would require chart controls and a causal test. This remains downstream of SCM-02/03 and outside Miller's 4D same-boundary, same-genus Seifert surface problem.
