# DISTILL-09: teacher qualification before paid distillation

Neon saturates the original 30-decision pilot; its visible derivations still require review. DISTILL-07/08 found no learned promotion over reuse-first. The hand-authored budget rule preserved raw expected utility under the declared work metric on the tested roots.

## 09A: audit original operator-side journal

On the operator machine, using the compatible original roster:

    python -m Distill.teacher_qualification review --legacy --campaign runs/neon-pilot --directory runs/neon-audit --create

Provide the independent reviewer with public packets, visible answers, source material and templates. Fill every score and evidence field, then repeat without --create. Report coverage over all 215 slots. Review bindings cannot authenticate independence or correctness. A fixture-mode replay cannot prove a model's identity. Keep journals and reviews outside Git.

## 09B: external research exam

    python -m Distill.teacher_qualification prepare-exam --out runs/neon-research09 --teacher "Gemini 4 Neon"

Send only the 62 prompt files in public/, in independent contexts. The fixture configuration compiles the roster only, and does not represent provider sampling. Keep evaluator code and labels away from examinees. Save each visible answer verbatim in an operator-owned directory by full packet ID plus .json; use an empty file for an omitted response. No retries after grading. Default seed matches the existing research development roster.

    python -m Distill.teacher_qualification freeze-exam --campaign runs/neon-research09 --responses runs/neon-research-raw
    python -m Distill.teacher_qualification grade-exam --campaign runs/neon-research09 > runs/neon-research-grades.json
    python -m Distill.teacher_qualification review --campaign runs/neon-research09 --directory runs/neon-research-audit --create

Freeze copies and hashes original bytes before grading; repeat grades verify all hashes. Invalid packets count in denominators. Local integrity is not authenticated provider custody. Distinguish exact pass, feasibility, certificate consistency, format and semantic review coverage. Match against the existing free Inky research runner with identical public packet IDs; retain actual provider and billing evidence locally.

## 09C: longitudinal teacher pairs

    python -m Distill.teacher_qualification prepare-pairs --out runs/teacher-pairs09

Eight paired frontier programs, four shared-control and four shifted-control. Specs and private draws are paired. Example:

    python -m Distill.program_runner next --campaign runs/teacher-pairs09/shared_control-0/neon
    python -m Distill.program_runner submit --campaign runs/teacher-pairs09/shared_control-0/neon --response action.json
    python -m Distill.program_runner run --campaign runs/teacher-pairs09/shared_control-0/inky --confirm-free-tier --limit 1

Check free tier, actual provider identity and billing. Keep separate contexts and send only model prompts, not evaluator files or other arms' trajectories. Finish all 16 arms before comparison:

    python -m Distill.teacher_qualification compare-pairs --campaign runs/teacher-pairs09

Reports realized utility, credits, false recommendations, transport, and sum of conditional one-step optimal-continuation gaps at visited states. Reuse-first is replayed in the same world. The DISTILL-08 budget-rule work benchmark is separate: these teacher runs do not match its work meter. Eight paired worlds are descriptive, not expected utility over the prior. Review agenda, evidence, budget and revision statements separately.

## 09D/E: paid and training gates

Proposed maximum Fable spend: $80 original research, $50 negative results, $40 adversarial experiment design, $20 independent criticism and repair, $10 reserve. No price assumed. Verify current model, rates and output-use terms before paid calls. Use the existing spend ledger or equivalent atomic reservation and actual provider receipts. Select hard development cases; freeze independent held-out structure and root splits before selection. Require checkable public artifacts and falsification criteria. Do not assume provider output is eligible for training.

DataGoblin transfer requires native owning-pipeline admission, matched-resource comparisons with cheap strong baselines and held-out evaluation. No paid calls, automatic escalation, training or provider-output admission in this change.
