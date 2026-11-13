# SCM-03 CPU regression receipt

**Date:** 2026-11-13. **Execution:** local CPU, Python 3.13, PyTorch 2.10 CPU, SymPy 1.14. **Scope:** synthetic braids plus mock visit-hook interface. Trained RMP checkpoints, dictionaries and monitor bundles are held outside GitHub and were unavailable to this execution environment.

Ran:

    python -m unittest discover -s research/seifert_activation/tests -q

**Result:** 32/32 tests passed: 23 existing SCM-01/02 cases and 9 new SCM-03 tests. The nine test methods cover free word cancellation, braid relation and inversion, seeded random cancellations, same-winding nontrivial commutator, based-versus-conjugate scope, 16 commutator variants, analyzer discrimination, and size-budget refusal. All 16 commutators [sigma1^(2a), sigma2^(2b)] for a,b in {-2,-1,1,2} had zero pairwise winding and nonidentity exact Artin action.

| B3 history | Pairwise winding | Alexander | Canonical genus | Artin based action |
| --- | --- | --- | ---: | --- |
| identity | (0,0,0) | 0 | 0 | identity |
| C = [sigma1^2,sigma2^2] | (0,0,0) | (t-1)^4 | 2 | nonidentity |
| inverse C^-1 | (0,0,0) | (t-1)^4 | 2 | nonidentity, distinct from C |

The C/C^-1 comparison has matching winding, Alexander and canonical Seifert genus while exact free-group images distinguish the based braids. SCM-03's resolution comparison rejects a deliberate replacement of C by C^-1 even when its abelian comparison passes. Markov-conjugate braids were explicitly tested to document why Artin-image inequality is not a link-isotopy obstruction.

The RMP capture defaults to three observed downstream positions, records the exact short Artin images and a content fingerprint, and reports image-budget exhaustion as inconclusive. The older independent winding/Alexander/Seifert checks remain intact.

**Neural validation status:** pending. No actual LOOP-T/FF-U activation capture, causal result, compression gain, or 4D Seifert-surface distinction was measured.
