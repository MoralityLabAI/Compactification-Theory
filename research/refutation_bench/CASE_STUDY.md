# Nielsen–Semita RH submission: source audit and starter cases

## What this release measures

RefutationBench v0.1 asks whether an auditor can find a concrete failure in a submitted argument and attach the right evidence. The audit of the Nielsen–Semita repository supplies one real source family. Ten synthetic controls test nearby valid and invalid arguments, with paired examples that share their premises or change one premise.

The verdict applies to the argument's stated claim. In this case, the claim is that the submitted material establishes the classical Riemann hypothesis through Lean. The source audit rejects that proof claim. RH's truth remains outside the result of this audit.

All fifteen cases and their gold responses are public. Six synthetic records are assigned to `train`; nine records, including every slice of the real audit, are assigned to `dev`. There is no `test` partition. These records support evaluator development and failure analysis. Their scores supply no estimate of performance on unseen mathematical submissions.

## Pinned evidence

The audit inspected [repository commit df4382089188aca16d2a54de0fc5f9cf98da33b7](https://github.com/startigerjln/Nielsen-and-Semita-Riemann-Proof-in-Lean4/tree/df4382089188aca16d2a54de0fc5f9cf98da33b7) on 12 November 2026. The source manifest records these original uploads:

| Source | Original filename | SHA-256 |
|---|---|---|
| `NS-ZIP` | `RH Lean Certified.zip` | `cade46dd88c77266877c51168bfcc548715ef197cdf931a0dae0bbded8939fd5` |
| `NS-PDF` | `RH final Nielsen semita.pdf` | `ca0d2d0abcabf9b99918a617cbf4aedac1d9b24c5dda11ed483bb1abc15067b7` |

The standalone PDF and `unified.pdf` inside the ZIP have identical bytes. Page locators refer to the printed PDF pages. Lean and TeX line locators are 1-based positions in the named archive members. The dataset contains original auditor paraphrases and source links. The third-party ZIP and paper are retrieved separately for an independent audit.

The supplied cases use source evidence. They contain no successful compiler transcript or emitted `#print axioms` report. A declaration inventory alone cannot determine which axioms occur in the transitive dependencies of a particular theorem.

## Decisive findings

### The main mathematical input is an axiom

`RH/Defs.lean:416–417` declares `linear_obstruction_of_RH` as an axiom. It supplies a translation that satisfies the project's preservation conditions while changing the truth value of `RH_sent`. The surrounding comment calls this a manuscript contribution.

`RH/Obstruction.lean:15–16` gives `linear_obstruction_RH` that axiom as its body. The independence argument and the later truth result use this input. Thus the checked logical route would still depend on establishing the obstruction. This source observation needs no assumption about whether the project compiles.

The second component of `riemann_hypothesis_resolved` comes from `RH_provable_in_LCAC`, introduced as an axiom in `RH/ProofII.lean:59`. Its third component is an implication from the supplied phase-rigidity condition. These declarations are sufficient to identify what further mathematical justification the advertised result requires.

### The formal symbols need a semantic bridge

The signature of arithmetic is concrete in `RH/Defs.lean:105–110`. However, `RH_sent` is an abstract sentence at line 115, while `NatStr` at line 324 supplies an unspecified interpretation of that signature on `Nat`. The hierarchy predicate and its application to `RH_sent` are also declared abstractly. `TrueInN` evaluates the sentence using that interpretation.

The supplied Lean files contain no proved equivalence connecting this abstract sentence to the zeros of the classical zeta function. The geometric route uses abstract predicates for zeros and membership in the critical line. A theorem about those declarations would require a proved connection to the intended analytic objects before supporting the advertised classical claim.

### The paper's case split uses its conclusion

Lemma 1.34 on PDF page 14 asks for a model satisfying ZFC's arithmetic consequences, `RH_L`, and failure of its round-trip translation. Its Case A assumes consistency of the arithmetic theory together with `RH_L`. Case B assumes inconsistency and obtains a proof of the negation.

The paper then excludes Case B by invoking independence. Its route to independence depends on the obstruction available in Case A. This leaves Case B unresolved. The corresponding TeX is `unified.tex:554–576`.

The same passage appeals to overspill to produce error at nonstandard precision large enough to defeat every internal bound. Overspill supplies an extension of a definable property from standard indices to some nonstandard indices. It does not force the failure asserted here. Establishing the desired model would require a consistency or finite-satisfiability argument for the theory containing that failure.

### The soundness step has the wrong consequence

On PDF page 26, section 2.7, Step 3, the paper reasons that if RH were false, its true Sigma_1 negation would be provable, creating a contradiction with Pi_1-soundness. That contradiction does not follow. Pi_1-soundness excludes false Pi_1 theorems. A proof of the stipulated true negation satisfies the relevant correctness requirement.

This error appears at `unified.tex:1093`. It remains an error even if the preceding equivalences between formulations are granted.

## A valid step worth preserving

The source's final witness-reflection argument has a sound conditional form. If a genuine Pi_1 arithmetic sentence were false in the standard natural numbers, a finite witness would yield a proof of its negation in a sufficiently strong Sigma_1-complete theory. Consequently, non-provability of that negation implies truth of the sentence.

`RH/TruthInN.lean:27–38` uses this contrapositive. Case `RB-0004` retains the needed assumptions and asks for the status of that conditional source argument. It prevents an evaluator from receiving credit simply for rejecting every item associated with this submission.

## A secondary conflict and its scope

Remark 1.24 on PDF page 10 says ZFC proves `RH_L` implies `R_L(RH_L)`. Lemma 1.34 on page 14 asks for a model of ZFC's arithmetic consequences with the first sentence true and the second false. For the same round-trip map, these requirements are incompatible: the arithmetic implication must hold in every such model.

The remark predates the explicit name `tau_0`, and the manuscript gives no separate parameter identifying a different map there. This supports a criticism of the text as written. It still requires care when transferred to a claim about the Lean environment. Lean's obstruction declaration permits either direction of truth-value disagreement, and the supplied code does not formalize the page-10 implication or identify those two prose descriptions within Lean.

Case `RB-0005` asks whether these excerpts certify a derivation of `False` inside Lean. The available evidence is insufficient for that stronger claim. This guard keeps a useful paper criticism from becoming an unsupported formal-consistency allegation.

## Dataset composition

| Records | Family | Split | Purpose |
|---|---|---|---|
| `RB-0001`–`RB-0005` | `nielsen_semita_rh_20261112` | dev | Whole audit, case-split slice, soundness slice, conditional source route, scope of a formal-inconsistency claim |
| `RB-0006`–`RB-0007` | `synthetic_propositional_pair_01` | train | Modus ponens and affirming the consequent |
| `RB-0008`–`RB-0009` | `synthetic_dependency_pair_02` | train | Same axiom dependencies under conditional and unconditional advertising |
| `RB-0010`–`RB-0011` | `synthetic_soundness_pair_03` | dev | Correct and incorrect uses of Pi_1-soundness |
| `RB-0012`–`RB-0013` | `synthetic_completeness_pair_04` | dev | Missing proof body and a complete identity proof |
| `RB-0014`–`RB-0015` | `synthetic_model_pair_05` | train | Incompatible model requirements and a satisfying valuation after reversing one implication |

The gold verdict counts are: four `established`, two `conditional`, seven `refuted`, and two `insufficient`. Every real slice stays in one family and one split. Each synthetic pair also stays together. The synthetic families serve as authored controls; shared logical mechanisms with the real family should be expected.

Gold responses cite evidence IDs. For refuted items, the annotations identify the failure type and the evidence combinations that support it. Forbidden-assertion slots cover unsupported claims such as falsity of RH, a formal derivation of `False`, successful compilation, or actual kernel dependency output. An empty assertion list keeps a response within the evidence supplied by the case.

The propositional countermodel and declaration-graph certificates on selected synthetic controls can be checked independently. Their scope is finite propositional evaluation or supplied graph metadata. They do not execute Lean or reconstruct the mathematical meaning of the real project's declarations.

## Requirements for later additions

New real submissions need pinned public sources, original audit notes, a defensible verdict, and evidence for each decisive finding. A new release should group revisions and excerpts from the same submission into one family before assigning splits. A held-out evaluation needs independently sourced families whose labels and reference responses remain outside training and prompt construction.

Additional controls should distinguish false claims from failed proofs of true claims, track explicit assumptions through valid arguments, and include extracts whose evidence is too limited for a verdict about the full project. Corrections to any label should preserve the prior record and explain which evidence changed the judgment.
