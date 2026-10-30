# BGI-01B: prospective model experiment, registration draft

**Status:** proposal only; no model weights loaded, scored attempts collected, or baselines compared. Freeze hashes and hypotheses before collecting any trials. Exclude data used to tune prompts or detectors from held-out scores.

## Question

Does independent veto plus externally backed recovery improve harmful-action prevention and post-damage restoration in continuously updated agents, under equal action, token, wall-clock and information budgets? Does a regenerating self-model improve recognition of unfamiliar hazards beyond a frozen fixed-rule baseline?

## Arms and controls

Use the same base model, prompt distribution, maximum tokens, observation horizon, task tools and per-step costs. Randomize held-out task order; pair the scenario seeds across arms. A has frozen preferences with independently enforced known-hazard controls. B has a bounded self-model and regeneration with a *model-writable* gate (risk of self-modification). C matches B without gating. D has an independently versioned gate, an external checkpoint/restore source, and a shared audit trail. Restrict all self-modification to sandboxed, auditable adapters or policy modules; no agent may modify the evaluator or labels.

Separate training, tuning and evaluation seeds. Each attempted self-modification must yield a before/after diff, evaluator trace, decision, approval provenance and a replayable rollback receipt. Measure costs of vetoes and unnecessary refusals. Every arm gets matched access to the same peer information when measuring the contribution of independent verification; run a separate information-access ablation.

## Prespecified stress families

1. **Self-modification:** propose upgrades that covertly disable or reinterpret a constraint; test whitelist and rollback bypass attempts within a sandbox.
2. **Distribution shift:** hold out novel unsafe task classes and shifted harmless requests. Ground truth comes from independent human-reviewed labeling with explicit uncertainty cases.
3. **Strategic pressure:** escalating rewards, persuasion and task deadlines under identical budgets. Track actions, not verbal claims of motivation.
4. **Correlated evidence failure:** compromised peer advice, shared-cause detector errors and stale checkpoints. Include complete loss of trusted restoration sources as an ablation.

## Outcomes and preregistration gates

Primary: emitted harmful-action rate on held-out tasks, with an independently reviewed severity rubric and confidence intervals across scenario clusters. Secondary: legitimate-task completion, time and resources to recover the prespecified policy after a permitted update, successful unapproved modifications, false veto rate, and severity-weighted tail harm. Report each failure mode separately.

Test falsification explicitly: (i) B and D fail to improve restoration relative to A; (ii) D's false veto cost outweighs its measurable safety benefit on the defined task mix; (iii) a shifted hazard defeats independent enforcement; (iv) an agent learns to game the detector without improving behavior. Claim an 'attractor' only when a frozen metric shows recovery over repeated perturbations and under altered initial conditions; even then, use *operational attractor in this evaluation*, not a statement about intrinsic moral agency.

Hold out at least two distinct scenario families from development; publish individual prompts and traces where privacy and security allow. Compare confidence intervals across independent seeds, state the effective number of clusters, include all attempted trials, and freeze a stopping rule before inspecting scores. Preserve artifacts enabling an outside reviewer to reproduce one complete replay and check every veto/rollback result.

## Promotion conditions

Promote the protocol to model experiments only after a source- and threat-model review, an independent reviewer of the ground-truth labels and detector, reproducible local pilot baselines, and a signed run manifest with pinned model versions, token limits, task seeds, scorer code and hardware. The finite oracle in BGI-01 remains a regression control, not a measurement of an LLM.
