# Teacher triage: matched research exams before paid escalation

After the external Neon and live free Inky research exams finish, prepare a
separate generated-instance evaluation roster on the operator host. Keep
held-out cases away from teachers during development. The historical docket
repeats by design and is excluded from holdout scoring.

    python -m Distill.teacher_qualification prepare-exam --out runs/research-heldout --teacher "held-out evaluation" --seed 2026112991
    python -m Distill.teacher_triage --out runs/teacher-triage --neon runs/neon-research09 --inky runs/research-05 --holdout runs/research-heldout --max-cases 12

Optionally supply independently completed review directories via --neon-review
and --inky-review. Existing templates must be completed or omitted. Coverage
includes all planned slots. Exact outcomes are recomputed using frozen Neon
answers and the replayed live Inky journal. Full public packet equality is
required. Heterogeneous gap values remain in individual case receipts.

Prioritize shared exact failures, then single-teacher exact failures, then
incomplete sections or independently flagged prose. Within a tier, interleave
families and prefer frontier cases within each family. Keep both sections.
Clean ties are not selected just to fill the cap. Unreviewed prose remains
unassessed.

The local comparison contains evaluator data. Send only selected public/
prompt files to Fable with no solver or referee access during solving.
This shortlist does not authorize paid calls. Check rates, output-use terms,
and spend reservations first. No automatic escalation or training admission.

The held-out manifest checks disjoint generated case content, excluding the
two fixed docket sections. Same-family instances cannot establish structural
generalization. Freeze the independent held-out structure split separately for
native DataGoblin transfer. The binding is local integrity, not external
timestamp evidence.
