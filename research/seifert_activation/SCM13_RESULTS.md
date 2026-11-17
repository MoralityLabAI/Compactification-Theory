# SCM-13 synthetic reference and audit receipt

**Run date:** 2026-11-14. **Scope:** new exact-rational topological transition threshold, deliberately synthetic sampled trajectories, and format-only RMP capture consistency audit. No trained model weights, D-shared dictionaries or monitor bundle were available to this execution host. Prior G0 and RMP registrations stay unchanged.

## Certified mathematical statements

- Each full-rank projected pair has exactly one affine collision preimage in the two-dimensional intervention disk. The pair winding changes from zero to the determinant orientation when the intervention radius crosses that point's norm.
- A strict exact-rational clearance inequality preserves the pair winding and the complete free-loop braid class between the true continuous response and its affine surrogate, provided uniform source-proved nonlinear error bounds.
- When all collision preimages are outside the disk, radial contraction proves that the complete pure braid is trivial. The nonabelian zero-winding commutator from SCM-03 is therefore an obstruction to a *sufficiently accurate full-rank affine-disk representation*.
- Source Hessian norms could provide `eta_i <= H_i r^2/2`; those RMP Hessian norms and floating-point chart errors remain unverified.

## Executed synthetic reference

- `python research/seifert_activation/scm13_reference.py --check --out research/seifert_activation/results/scm13_synthetic_reference.json`
- Immutable JSON SHA-256: `60cd7d493461a83b4102033ab209d628fa9c65955b676ac70c95fcc39a52d661`
- Seed 1313: 80/80 nondegenerate affine fixtures matched independent polygonal winding, comprising 55 null and 25 nonzero pair-winding cases. The two-strand fixture crosses its critical radius at exactly 2.
- The three-strand commutator `[sigma1^2,sigma2^2]` has three zero pair windings and a faithful nonidentity Artin action. This is synthetic braid algebra, with no inferred network nonlinearity.
- The sampled first-harmonic screen predicts the correct synthetic winding while always reporting `empirical_only_unverified_full_circle_error`; a sparse four-turn alias retains that label.

## Campaign provenance hardening

`scm13_receipt_integrity.py` guards four-arm capture comparisons by checking source classification consistency, campaign-wide fixed checkpoint/dictionary/feature/chart hashes, identical token sequence hashes across the two original models for each task, and complete real RMP 32-angle eight-visit capture structures. A read-only CLI is included. The existing SCM-12 launcher now invokes this stricter analyzer after capture. A syntactically valid hash is still not proof that its underlying bytes came from the claimed checkpoint; the SCM-12 frozen-manifest preflight supplies an independent identity check.

## Interpretation

No neural topology, learned latent reasoning, compression advantage, causal improvement or 4D Seifert-surface isotopy obstruction is established. The next empirical step remains the registered trained-model capture using external, hash-verified RMP artifacts, followed by an independently verified Hessian error bound if a full continuous certificate is sought.
