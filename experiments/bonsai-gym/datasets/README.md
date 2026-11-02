# Datasets

## meta_sft v0.1 (`meta_sft_v0_1/`)

A small, versioned meta-learning SFT corpus for the Bonsai gym: each record is a decision an experiment controller
has to make (which task next, which adapter placement, which control, how to read a failure), grounded in a result or
a registered plan from the Morality Lab repositories. It is built by `scripts/build_corpus.py` from pinned git commits
listed in `configs/corpus_sources.json`. Rebuild: `python experiments/bonsai-gym/scripts/build_corpus.py`
(`--check` rebuilds to a temp dir and compares). Tests: `python -m pytest experiments/bonsai-gym/tests/test_corpus.py -q`.

Files: `train.jsonl`, `val.jsonl`, `heldout.jsonl`, `manifest.json` (counts per split, family and evidence type;
sha256 of every file; source commits; builder sha256; heldout families and lineages). The older `canary/` and
`parity/` directories here belong to other work and are untouched.

### Counts (see `manifest.json` for the authoritative numbers)

| family | train | val | heldout | total |
|---|---:|---:|---:|---:|
| curriculum_selection | 15 | 1 | 0 | 16 |
| experiment_design | 27 | 1 | 0 | 28 |
| failure_diagnosis | 38 | 2 | 0 | 40 |
| learning_gain_prediction | 32 | 3 | 0 | 35 |
| quantization_aware_adaptation | 13 | 0 | 0 | 13 |
| skill_composition | 9 | 1 | 0 | 10 |
| task_graph_navigation | 9 | 4 | 0 | 13 |
| specialist_routing | 0 | 0 | 11 | 11 |
| transfer_across_families | 0 | 0 | 12 | 12 |
| **total** | **143** | **12** | **23** | **178** |

Evidence types: 125 observed, 27 hypothesis, 26 synthetic (14.6%, cap 30%). Heldout is 22 observed + 1 hypothesis, no synthetic.
Lineages: 74 train, 8 val, 12 heldout (94 in total).

### Record schema

One JSON object per line. `schema_version` is `meta_sft/0.1`.

| field | meaning |
|---|---|
| `id` | `ms01-` + first 16 hex of sha256(`lineage_id`\|`family`\|index within that lineage and family, in spec order) |
| `family` | one of task_graph_navigation, curriculum_selection, learning_gain_prediction, skill_composition, experiment_design, failure_diagnosis, specialist_routing, quantization_aware_adaptation, transfer_across_families |
| `task_context`, `available_skills` | what the controller is told |
| `proposed_action` | the action taken or proposed |
| `predicted_improvement` | the prediction stated in the source, or null when the source states none |
| `observed_outcome` | what the source reports happened; non-null only for `observed` records |
| `verification_method` | how the claim is or would be checked |
| `evidence_type` | `observed` (a measured result is quoted, with path and section), `hypothesis` (a registered prediction or proposed next step with no result in the source), `synthetic` (templated, see below) |
| `lineage_id` | experiment chain, for example `q1-adapters/reverse-text`; all records from one chain share it |
| `provenance` | `repo`, `commit` (branch@sha7), `commit_sha` (40 hex), `path`, `lines_or_section` (section name plus the line span of the quoted evidence in the pinned file), optional `also_paths`; synthetic records carry a `basis` block |
| `messages` | `system`, `user` (context, skills, family question), `assistant` (action, predicted improvement, how to verify, and for observed records a final "What happened (recorded in the source)" line) |
| `prediction_basis` | extra field, not in the requested schema: `preregistered`, `mechanism_rationale` (the source gives a mechanism but no number), `proposed`, or `none` (then `predicted_improvement` is null) |

The outcome is the last line of the assistant message. Train a proposer on the action, prediction and verification lines only
(truncate or loss-mask the "What happened" line) unless the aim is to teach outcome recall.

### Provenance and faithfulness

Every non-synthetic record is generated from a spec in `build_corpus.py` that lists verbatim evidence strings. At build time each
string is searched in the file as it is at the pinned commit (`git show <sha>:<path>`; whitespace, `*` and backticks are ignored
when matching) and the build stops if one is missing; the line span in `lines_or_section` is computed from those matches.
So the numbers in `observed_outcome` can be traced to quoted text, but the prose around the quotes (contexts, actions, outcome wording)
is the builder author's summary of the source; read the cited section when a nuance matters. `configs/corpus_sources.json` lists every
source repo, local path, commit and the files used, and the builder refuses to run if that list and the specs disagree.

Sources used: Compactification-Theory (main@9f0783e plus siml01, bgi01h, cb02, rc01, ct02 and classical-learning-theory branches),
q1-adapters, BitAgent `experiments/ternary-lora-5080`, HybridTRMLDT (master and the S0 real-row-parity branch), TRMStoryworld (cloned to
`C:/Users/patri/Documents/Codex/bonsai-gym/sources/TRMStoryworld`), ConstitutionalAlignment, EvalMath (eval-reuse toys), StableRateStudy,
loop-capacity, jev-qwen (xbench summary).

### Split rules

1. Heldout families are `transfer_across_families` and `specialist_routing`: every record of those two families goes to `heldout.jsonl`, nothing else does.
2. Any lineage that appears in heldout is removed from train and val, whatever the family of its other records ("quarantine"; 0 records needed it in this build,
   because heldout lineages were defined so that they contain only heldout-family records).
3. The remaining lineages are ranked by sha256(`seed`|`lineage_id`), seed 20261102; the first round(0.10 x lineages) go to val (8 of 82), the rest to train.
   Val is about 10% of lineages but only 8% of records (12 of 155) because lineage sizes vary.
4. Near-duplicates inside a (lineage, family) group are collapsed (word-set Jaccard of action, prediction and outcome at least 0.90); none were collapsed in this build.

### Why these heldout families

`transfer_across_families` was asked for. `specialist_routing` was chosen over `failure_diagnosis` (the family with the most observed records) because every
failure analysis shares a lineage with the experiment that produced it, so holding the family out would have quarantined most of the corpus; routing results live in
their own lineages (adapter-library routers, TRM/LDT routing, commit/veto gates) with measured numbers and cost tables. Its weakness is size: 11 records.

### Leakage controls and their limits

- No lineage is shared across splits; no heldout family appears in train or val; heldout lineages are not mixed with other families.
- Heldout lineages are separate experiments, but several sit in the same papers as train lineages (for example the q1-adapters paper, the jev-qwen xbench summary,
  the HybridTRMLDT paper). Base-model scores and setup descriptions (for example the 0.208 4-bit baseline on reverse-text, MATH-500 base scores) can therefore appear in both train and heldout contexts.
  Use heldout to measure behaviour on new decisions, not to claim zero shared facts.
- Synthetic records exist only for non-heldout families, each scenario has its own lineage, and the allocation and next-node answers are computed from the stated rule, not measured.
- The builder never reads a working tree; the manifest records the builder hash (CRLF folded to LF) and the sources config hash.

### Synthetic records (26)

Templated from procedures the sources state: 12 guarded task-allocation scenarios (65% frozen family weights, 35% accuracy gap, from SIML-01N), 8 next-node scenarios on
the proposed task graph, 6 pre-registration checklists (adapter placement, ternary LoRA learning rate, router ablation, curriculum comparison, repair loop, runtime parity).
Inputs such as development accuracies are invented and labelled HYPOTHETICAL in the context; no synthetic record contains an `observed_outcome`, and none claims that the guarded policy beats random
allocation (in the source random allocation scored higher).

### Known gaps

- Size: 178 records, 94 lineages. Several sources contribute 1 to 4 records.
- No Bonsai 2 27B adapter-training outcome exists yet: the S0 records are runtime parity and VRAM measurements, and the 27B trainability gate is recorded as open.
- Not mined: LoopedTransformers, Hermes-Skills/hermes-lite and Control-Harness (only second-hand through jev-qwen), the remaining HybridTRMLDT feature branches (loop-schedule, RLM mesh), the EvalMath papers (only the sealed toy runs were used), most of ConstitutionalAlignment, the Compactification-Theory physics branches (ws01-ws07), plan-r, hnet, sch01, irc02, selfmodel-gym.
- TRMStoryworld contains little measured outcome data; 3 records, one a procedure.
- `hypothesis` records include theory-derived statements (learning-theory crosswalk) and proposed procedures, not only registered numeric predictions.
- Contexts for a few records rely on author paraphrase of tables; line spans point to the quoted evidence, not to every sentence of the record.
- Val has 12 records and no quantization_aware_adaptation record.
