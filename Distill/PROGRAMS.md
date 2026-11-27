# DISTILL-06 / longitudinal research programs v0.5

DISTILL-05 evaluates individual research decisions and explanations. DISTILL-06 introduces four linked synthetic investigations under a depleting experimental budget, inspired by MTR-02D/I evidence DAGs and certificate memory. This is a finite model, not a native scientific result.

## Research environment

- Anchor: investigate a local claim through fallible screens or an exact audit.
- Transfer A and B: investigate related claims. A positive audited law and an exact anchor certificate allow cheaper checked transport using the project's offset.
- Out of scope: no transport. The shifted-control family changes the screening mechanism in this final project.

Six binary variables with public factored prior weights define 64 worlds. The actual world is private. Screens share a nuisance, so repeated observations must not be treated as independently reliable. A control reveals the nuisance; a law audit verifies the transport premise; direct audits establish local certificates. Transport verifies both premise receipts and scope. Recommendations and publication decisions advance to the next project; the agenda can evolve after every observation, but earlier finalized decisions are immutable.

Experiments deduct credits and utility. A correct recommendation earns +6, a wrong recommendation -12, a valid certificate +10, and defer 0. These are stipulated synthetic utilities. Recommendation truth and reward stay private until termination. The actor receives the public prior, rules, evidence ledger, certificate provenance, compatible-world masses and legal actions. It returns one native JSON action, research agenda, evidence interpretation, budget rationale, and revision rule. State IDs bind answers to the public observation. The referee owns actual-world selection and outcome labels.

## Exact evaluation

An integer-weighted dynamic program computes whole-program expected utility. Each policy is independently replayed across every possible world to validate observations, costs and rewards. Baselines: stop, local one-step replanning, direct-first, reuse-first and an exact oracle with transport disabled. The transport ablation restricts only the action set. Environment credits are matched, but planner work is separate; no matched-runtime or learned-policy claim follows.

## Synthetic curriculum

    python -m Distill.program_curriculum --out runs/program-curriculum

Default: 12 train, 4 calibration, 8 ordinary public-eval, 8 shifted-control public-eval roots. Each contains up to two selected reachable states per project and exact continuation values and regret for every legal action. Oracle and local-policy trajectories supply selected observable states; they are not IID. The shifted mechanism is absent from training/calibration. Public root/state hashes reject exact overlap. The benchmark covers the complete prior distribution per evaluation root.

Per-root public/ JSON documents contain observations only; referee/ documents contain oracle labels. Keep them separate and never send labels, private world selection or evaluator files as model input. Synthetic exports have training_admission=false. Native DataGoblin ingestion, student fitting and cost-matched ranking experiments are separate follow-ups.

## External DataGoblin and free Inkling runner

Use fresh private run directories. The host selects a world using a separate seed and binds the referee file to the run manifest. The salted content hash is local integrity rather than external authentication. No referee fields appear in model-facing observations.

    python -m Distill.program_runner prepare --out runs/dg-program --mode external --seed 2026112820 --family shifted_control
    python -m Distill.program_runner next --campaign runs/dg-program
    # Send only the returned prompt; save its JSON response.
    python -m Distill.program_runner submit --campaign runs/dg-program --response action.json

Repeat next/submit until complete. Offline fixture and live free-tier lanes share the durable journal:

    python -m Distill.program_runner prepare --out runs/program-fixture --mode fixture --world-seed 7
    python -m Distill.program_runner run --campaign runs/program-fixture --limit 32
    python -m Distill.program_runner replay --campaign runs/program-fixture

    python -m Distill.program_runner prepare --out runs/inky-program --mode live --seed 2026112820
    python -m Distill.program_runner run --campaign runs/inky-program --confirm-free-tier --limit 1
    python -m Distill.program_runner status --campaign runs/inky-program
    # After verifying actual provider model and billing:
    python -m Distill.program_runner run --campaign runs/inky-program --confirm-free-tier --limit 31
    python -m Distill.program_runner replay --campaign runs/inky-program

The runner uses the existing free-only Tinker sampler, 12,288 context, 4,096 generated-token cap and effort 0.6. No paid fallback, checkpoint writes, training or Fable calls. Keep the journal, referee and responses private and outside Git. Pending or invalid actions stop the run; no automatic resubmission or invented substitute action. STOP blocks new claims. Malformed external JSON leaves a pending claim. Replay reconstructs accepted actions and the actual trajectory. A terminal result is a single realized world, not exact expected utility across the prior.

Research agenda, scientific interpretation and revision rules still need independent review. No provider-output training export. Older DISTILL-01–05 packets and results are unchanged; the sampler receives an additive raw-prompt entry point.
