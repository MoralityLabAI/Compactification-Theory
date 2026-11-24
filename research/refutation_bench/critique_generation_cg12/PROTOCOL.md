# RB-CG-12: original-source provenance pilot

## Purpose and registered scope
CG-11 tests kernel-derived dependencies in generated source. CG-12 moves the same extractor to original MathCan declarations, at commit 6d02b611c7358216379cb36952512a104e0bdb8e (Lean 4.35.0). This is an engineering and corpus-development pilot. No learner is evaluated and no source author is accused of a mathematical defect. A proof that uses a legitimate classical axiom is valid in its declared foundation.

Four unchanged Init-only source modules, nine fixed roots:
- ObservableQuotient: factor_iff_fiber, factor_unique.
- AuthorizationQuotient: classifier_exists_iff_fiber, collision_forces_error, coarse_compatible_implies_fine.
- NOR19StreamedVerifier: streamed_exactly_iff_materialized_equal, streamed_rejection_iff_unequal.
- MatchedStepMargin: positive_margin_stable, negative_margin_stable.

Compile each original module without changing any bytes. Independently query #print axioms of its selected roots from a separate importing module, outside the original namespace. Separately elaborate an instrumentation copy with an additional import Lean and appended CG-11 extractor and queries. Record both hashes and compare the complete imported original axiom set with the instrumentation-copy root axiom set, #print axioms, and independent reachability closure. Reject any discrepancy. The instrumentation copy is not asserted to have an identical proof term to the original; the native original axiom-set comparison bounds the provenance claim. Only local namespace definitions/theorems/opaque/axiom nodes are expanded, and imported or out-of-scope constants remain summary boundaries. Abort on an unsupported local kind or missing node. The pilot deliberately chooses roots with no anticipated reachable local inductive declarations.

## Matched policies
Every unchanged selected declaration is evaluated under two explicit foundations: {propext, Classical.choice, Quot.sound} and {propext, Quot.sound}. These are hypothetical policy variants, not allegations about the theorem. Any native root axiom outside the first set fails the pilot pending manual review. The second policy excludes Classical.choice. All native axioms remain visible in independent receipts, and a policy violation needs a real dependency path ending in the excluded axiom or a boundary carrying that axiom. A clean verdict requires complete graph traversal. The policy checker uses the same CG-11 independent certificate contract. Archive both policy cases with original source hash and certificate; no training data/test leakage claim. Report counts by root, native axiom set, number of expanded nodes, boundary count, certificate length and any mismatch. All cases are development-only.

## Interpretation
A missing axiom in a particular policy means this submitted proof exceeds that policy. It does not prove the target unprovable constructively. A direct local edge identifies a dependency of the instrumented proof. Imported boundaries expose transitive axiom summaries only. The pilot must pass all gates before expanding to the Nielsen historical case or large original modules. Keep initial failures and do not change selected roots after viewing the results.
