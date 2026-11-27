# DISTILL-05 / research suite v0.4.0

DataGoblin should choose informative experiments, preserve research credits, diagnose negative results, reject unearned claims, and propose the cheapest decisive follow-up. This version creates a research-specific examination lane rather than only increasing numerical problem sizes. The 25-gym legacy registry, v0.3 compiler, v1 packets and earlier pilot plan hashes remain unchanged. Research packets have library_version 0.4.0 and response.v2: the exact answer is native JSON, with surrounding worked explanations in strings.

## Research families

| Family | Source | Exact decision |
|---|---|---|
| adaptive_evidence | DataGoblin MTR-02D | Optimal contingent policy under test costs, prior weights, a per-path budget and costly false publication |
| robust_discrimination | BPD-DT01 | Minimum-cost mechanism-distinguishing intervention panel with prerequisites and bounded bit errors |
| research_portfolio | MoLT Goblin | Maximize worst-scenario gain under CPU, RAM, dependencies and permission constraints |
| paired_causal_gate | SCM-23 | Exact root-weighted paired effects and matched-resource eligibility |
| proof_obligations | EvalMath RH-PROOF | Select audits maximizing conditional claim value without promoting unresolved or refuted claims |

These are new finite synthetic planning models, inspired by source-pinned research. They are not native gym executions or independent empirical findings. Standard and frontier settings expand hypotheses, tests, jobs, roots or noise tolerance. Actual scientific hypotheses remain the responsibility of the researcher.

Each research case has four questions: exact decision, scientific interpretation, follow-up experimental design, and structural obstruction/repair. Decision certificates accept multiple optima. The adaptive policy is checked by explicit per-hypothesis replay; an independent conditional recursion tests the optimum. The oracle records feasibility, claimed-value consistency, optimality and gap for the optimization families. Open-ended prose requires separate review. The second section does not receive the answer to the first.

## Research evidence docket

The suite also includes one fixed public MTR-02I negative-result docket at DataGoblin commit cbc028c2525b148269e14b43b5f379f5c10a9b72, report MTR_02I_REPORT.md. At the 8192 logical-work cap the learned scorer and forced scored route coincide, while cheap route leads both cohorts. The exercise asks for the displayed utility loss, causal interpretation of the forced-route control, resource accounting and a defensible next experiment. It does not assume a wall-time gain or promote a learned scorer. This is a public historical development result, excluded from synthetic curriculum export.

## Free Inky pilot

Run the existing free-only Tinker transport on the authenticated operator host:

    python -m Distill.inky_free prepare --out runs/research-05 --mode live --per-family 0 --research-per-family 6 --research-section-size 2 --research-difficulties standard,frontier --seed 2026112805 --max-generation 8192 --effort 0.6
    python -m Distill.inky_free run --campaign runs/research-05 --confirm-free-tier --limit 1
    python -m Distill.inky_free status --campaign runs/research-05

After checking the live model and billing, run the remaining 61 requests and replay:

    python -m Distill.inky_free run --campaign runs/research-05 --confirm-free-tier --limit 61
    python -m Distill.inky_free replay --campaign runs/research-05

The pilot contains 30 generated cases plus one fixed docket, 62 two-question sections and 124 question slots. All decisions precede the research-design sections in the round-robin queue. Keep the complete raw response archive local. No paid fallback, Fable, training, or checkpoint writes. Provider identity verification is now mandatory. Existing context and journal constraints still apply. For offline verification change --mode to fixture and omit --confirm-free-tier.

## Research-taste rubric

The original four dimensions remain. Research questions additionally receive applicable scores for decision value, resource accounting, alternative hypotheses, evidence scope, experimental identification, falsifiability and structural counterexamples. The review form includes per-dimension denominators because different questions ask for different competencies. Reviewer identity is claimed and review quality requires independent audit. An exact optimum is not proof of good research judgment outside the stipulated model. Separate format failures, feasible-but-suboptimal plans, wrong certificates, and scientific proposal quality.

## DataGoblin curriculum

    python -m Distill.research_curriculum --out runs/research-curriculum-v1

The default export builds 48 train, 16 calibration, and 32 public-eval synthetic adaptive-evidence roots with disjoint content hashes and fixed split seeds. Each public row contains only the public task and legal first actions, with test costs and outcome masses. The evaluator-only referee has exact optimal-continuation values, one-step values, regrets and replayable policy trees. Stop, one-step, and maximum-outcome-entropy-then-stop are cheap reference baselines. All candidate policy values are replayed across the hypotheses and their maximum agrees with the DP optimum. This directly measures opportunity cost of selecting a research experiment. Oracle continuations are evaluator resources, not deployable model performance. Neither actual provider responses nor the fixed research docket enter the dataset. All exports explicitly set training_admission=false pending admission by the owning pipeline. No model training is implemented in this version.

Keep original live pilot scores immutable. This suite changes prompt and task distributions, so a comparison to the old pilot is exploratory. New native JSON answer handling applies only to research packets; the earlier strict v1 scores are preserved. Independent held-out structure families, downstream student utility and research validity outside these finite models remain future evidence requirements.
