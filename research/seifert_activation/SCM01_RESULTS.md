# SCM-01: exact synthetic topology instrument, first receipt

**Run date:** 2026-11-13. **Execution:** local CPU, Python 3.13.5, SymPy 1.14.0. **Input:** synthetic braid fixtures only; no model weights or activation arrays. **Executed test command:** `python -m unittest discover -s /mnt/data/seifert_activation/tests -v` (8/8 tests passed). Additional randomized checks: 15 seeded (`seed=33`) B3 braid words of length 4, each checked for positive Markov stabilization invariance, conjugation invariance and exact crossing extraction; 15/15 passed.

| Braid presentation | Closure | Alexander (normalized) | Canonical Seifert genus | Surface components |
| --- | --- | --- | ---: | ---: |
| `B2: [1]` | unknot | `1` | 0 | 1 |
| `B2: [1,1,-1]` | unknot | `1` | 1 | 1 |
| `B2: [1,1,1]` | trefoil | `t^2-t+1` | 1 | 1 |
| `B2: [-1,-1,-1]` | mirror trefoil | `t^2-t+1` | 1 | 1 |
| `B3: [1,-2,1,-2]` | figure-eight | `t^2-3t+1` | 1 | 1 |
| `B2: [1,1]` | Hopf link | `t-1` | 0 | 1 |
| `B2: []` | two-component unlink | `0` | 0 | 2 |

**Interpreting the first pair:** `sigma_1^2 sigma_1^-1 = sigma_1` in `B2`; hence both presentations close to the exact same unknot. Their oriented Seifert-algorithm surfaces use different numbers of half-twisted bands. The extra canceling pair makes an additional handle in the chosen surface, changing its genus without changing the boundary. The pair is a presentation-sensitive constructed-surface control, *not* an example of Miller's equal-genus non-isotopy.

**Negative instrument controls passed:** open endpoint configuration rejected, unresolved over/under crossings rejected, positive affine changes of observed x/y coordinates preserve the extracted braid word. Checks against simulated sparse observations make no assertion that a black-box neural trace is free from sub-sampling collisions. The Alexander polynomial also fails to distinguish trefoil mirrors; it is an incomplete link invariant by design.

**Scientific decision:** Gate A passed in local synthetic tests. No neural phenomenon was tested. The next gate is repeatability and causal relevance in RMP residual traces under cyclic interventions, with charts, thresholds and holdouts registered beforehand. Four-dimensional embedded-surface construction and Khovanov cobordism maps remain future research.