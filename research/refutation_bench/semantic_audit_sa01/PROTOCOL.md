# SA-01: sorry-free semantic proof audit

## Motivation and scope

The pinned Nielsen–Semita audit in [CASE_STUDY.md](../CASE_STUDY.md) identifies source-level undeclared mathematical obligations, an abstract-to-classical statement bridge, circular consistency reasoning, unsupported overspill and a soundness inference error. Its archived source build failed. That source family remains a public historical case, not evidence of a kernel-accepted counterexample. The present independent synthetic fixtures are designed to compile with no admitted proofs, including every defective advertised claim's submitted declaration.

A submission is (E, d, T, A): a pinned Lean environment E, named declaration d, precise independently specified advertised proposition T, and allowed axiom set A. Check:
- K: replay the exact source in a pinned kernel, and inspect warnings and axiom output.
- D: require the transitive axiom closure of d to be contained in A. A declaration elsewhere in the source is irrelevant to this check unless it is used.
- C: require a separately checked bridge from the submitted declaration to T, preserving the original premises, quantifiers, domain and intended interpretation. A checked proof of T independent of d can establish T but is not a certificate of what d proves.
- S: when existential applicability or a nonempty domain is part of the advertised claim, check that claim independently. A vacuous universal implication is a legitimate theorem.

Record a vector, not a single "valid proof" flag. A counterexample to T in a finite toy establishes falsity of that toy T. A countermodel to the claimed inference establishes nonentailment in its modeled fragment only. A missing correspondence proof is an open obligation, not falsity. An extra axiom flags reliance on a premise outside A, even when T happens to be true. Build failure is a distinct outcome. No claim about RH's truth, proof, or independence follows from these fixtures.

## Seed protocol (public, no blind evaluation)

Eight paired source families; 16 input records. Each pair shares exact source bytes and differs in selected declaration or advertised proposition. The fixtures cover transitive axiom dependence versus a dead axiom; abstract encoding drift; vacuous eligibility; lost domain restriction; conclusion supplied as a premise; reversing implication; definitional identity versus independent meaning; and witness strengthening.

Inputs are source text, selected declaration, advertised Lean proposition and common axiom policy. Gold labels and certificate proof terms are excluded from input export. Source and independently appended certificate files are compiled separately. The source is checked for unexpected axioms at the selected declaration; gold certificates are independently required to depend only on propext, Classical.choice and Quot.sound. No native_decide, admit or sorry. The native transcript is authoritative for the small fixture; no parser-derived dependency graph is treated as kernel output. Retain all failed compiles and certificate checks.

Supported controls have proof terms explicitly using the selected declaration. The axiom-dependent case has a kernel axiom-closure receipt and no purported counterexample to its true advertised proposition. The other defective claims have independently compiled negations of their explicitly advertised toy propositions. This initial pilot intentionally chooses easy negative witnesses; it cannot establish open-domain semantic reasoning ability.

## Next preregistration: SA-02

Generate matched families with fresh identifiers, randomized independent definitions and proof aliases, mixed valid decoy axioms, different depths of transitive dependencies and independently sourced Mathlib targets. Split by generator family and original theorem ancestry before generating examples. Hold out entire mechanism combinations and a separate private source-family test. Balance supported and unsupported claims and include both true but unsupported claims and unknown correspondence obligations; keep real-source data and public synthetic gold separate.

Audit actions: read declaration or definition; inspect dependency closure; unfold a chosen constant; inspect a type; request a bounded witness; propose a bridge or countermodel. Every action has a logged cost, including Lean invocations, fetched bytes, tokens and wall time. Compare axiom-only, statement-only, symbolic closure/type checker, unrestricted verifier-assisted search and recurrent learned audit controller at identical budgets. Unrestricted symbolic checking is the oracle for the decidable fragment, not for general Lean entailment.

Primary proposed gate: independently verified and scope-correct audit evidence on 90% of sealed defective cases, false rejection at most 5% on valid controls, and separately report abstention and coverage. Freeze exact thresholds and budgets before training; do not tune them on test data. Report per-mechanism recall, verified evidence precision, smallest certificate bytes, cost per verified audit, and matched-pair joint accuracy. For unknown claims, abstention is correct when no sufficient evidence has been found; it is not a verified refutation.

## Mathematics to develop

For a finite explicit DAG of declaration dependencies, a forbidden-axiom path is a certificate of a violation of the axiom policy. Reverse reachability gives exact closure in linear graph time. A single path certifies use; absence of use requires checking the full reachable slice. For a modeled entailment Gamma entails T, a structure satisfying Gamma and falsifying T is a semantic refutation certificate. General proof search has no completeness guarantee under finite resources. Minimize the evidence certificate subject to correctness; arbitrary source graph metadata does not establish genuine Lean dependency edges.

Learned policy hypothesis: recurrent state can retain the current claim, open correspondence obligations, dependency slice, and checked evidence while a verifier chooses which obligations to close. Test acquisition cost and matched controls, not a reward for rejecting proofs. Later attach the original Nielsen source audit as an external case study, with its recorded build status unchanged.
