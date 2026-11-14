# FC-11: first original MathCan multi-step Lean tactic-state experiment

**November 14, 2026. First valid source-pinned CPU-matched evaluation completed.**
[Immutable model evaluation #43543404745](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43543404745)
passed the actual Lean4.35 proof checks, using original 62,401-parameter
LR-01 checkpoints frozen before the test selection became available.

The preregistered proof-count criterion **technically passed for seed17**
against a per-theorem *deadline-limited* Lean-native tactic search,
by a margin of one proof. The separate compute-efficiency
criterion **failed for both seeds**. The narrow count advantage is
**deadline/startup sensitive**: the native control timed out on two
GraphAuthorization arithmetic theorems that independently compiled
successfully with a 20-second allowance. It does not support a robust
learned proof-search superiority claim.

## Original comparator defect and post-hoc validation

The earlier [first completed model CI #43541890610](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43541890610)
produced genuine model-selected kernel proofs (seed17 4/14,
seed29 1/14), but its native comparator emitted invalid Lean
source by declaring a tactic macro before a subsequent import.
Its recorded native success count of 0/14 was **not a valid
baseline observation**. The source/test cases and original
frozen trained models were retained without modification.

[Corrected frozen-weight replay #43543404745](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43543404745)
reused the original SHA256-verified weights from model
training run #43541360928. The corrected native control
compiled 3/14 within seed17's model-derived per-case
allowances and 4/14 within seed29's allowances.
The learner's 4/14 versus 3/14 edge for seed17 depends
on two native timeout cases whose total wall-time overrun
was just 115.70 milliseconds. The comparator repair
happened **after** the original evaluation and is
therefore an explicitly post-hoc follow-up.

The separate [relaxed native search CI #43543846501](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43543846501)
compiled **5/14** original held-out theorems with a
20-second allowance each and zero timeouts: all four
GraphAuthorization arithmetic proofs plus
Intervention.obsEqOn_refl. This exceeds both trained
actors' four and one certified proof successes.
It provides a stronger comparison within the same
eight-script registry.

The [cross-run action and model timing audit](fc11_probe_and_replay_audit_20261114.json)
replayed all 248 training-stage action booleans from two
independently compiled artifacts, validated original
checkpoint SHA256s and checked the 28 model and 28
matched native test records. Eleven of 31 training
states had no progressive action in the registered
vocabulary. Native time allowances matched model
elapsed times case by case, and all model positives
were recorded as distinct sorry-free Lean compilations.
The source reference proof oracle remains 14/14.

## Source, provenance and Lean kernel

FC-11 preregistration was frozen FIRST at
`b57e2500e829b12fdfc257077b8e410171c93b98`.

Source is the original unmodified `patrickdugan/MathCan`
repository at commit
`c84e2025334e9a967c1297425a490f98d5b3d189`,
under its own `leanprover/lean4:v4.35.0` toolchain.

- **Train-only selector**, before any held-out selection code existed:
  `2e7e749d4c7f8610a4083f4f8551d084102ea55a`.
  [Source CI #43540113159](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43540113159)
  compiled the reference proofs of **16 original research theorems**
  in FiniteHorizonGap, TOECosmology, TOEGraph, and DeletionMemory,
  and extracted **31 real intermediate compiler-emitted tactic states**.
  Their canonical JSONL SHA256 is
  `47fae96eac1f9b176519faaa0a64cbf016203b6f23ba56e40a2295e9fedaa97d`.
- **Test selector**, committed later at
  `b821e0d85423606cf971b1f80db8092ede693d81`.
  [Source CI #43540299693](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43540299693)
  compiled **14 original theorems** from the disjoint
  Intervention, HookState, and GraphAuthorization modules,
  extracting **29 real intermediate states**; canonical
  JSONL SHA256:
  `a8e7bee63f80c03c640ccdbb9bada1f8d943e33fb403e37ad1fc7dbf536e11ad`.
- The original theorem bodies were left intact in MathCan.
  Separate instrumented source copies inserted `trace_state`
  before successive source top-level tactics. Compiler logs
  supplied every observed goal and its real local hypotheses.
  The original source proof oracle remains **14/14**, excluded
  from model inference.

## Genuine train-only action coverage

Eight preregistered actions:
`omega`, `decide`, `rfl`, `simp_all`,
`aesop`, `assumption`, `intro`, `constructor`.

[Training action-probe CI #43541101044](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43541101044)
applied **every action to each of the 31 real source training
states**, yielding **248 actual Lean4.35 compiler probes**.
Thirty-three attempts made accepted progress, of which
18 closed their scratch goal. Only **20 of the 31 training
states** had at least one registered action that made
accepted progress. All candidate action classes were
represented as trials, including failed actions.

The `aesop` tactic was *unavailable in MathCan's pinned
Lean `import Lean` environment* and therefore had zero
successful source probes. `assumption` also had zero
successful probes on the selected training states.
The candidate bank and training cases were kept
unchanged. The native comparator represents unavailable
`aesop` as an explicitly failing branch.

The source-probe JSONL SHA256:
`225337bcefd6138dc3cefeb28a5bede14710492c105bbab2cbd4aeff3c33c09e`.
The model training job regenerated the same 31 source
states and all 248 action probes independently. Its
probe artifact SHA256 differs because the logged wall times
differ between CI runners.

**Scratch runs may use `all_goals sorry` only to observe
a new Lean state.** These compiler probes are *never*
counted as positive theorem certificates. A model
proof must pass a separate, sorry-free Lean4.35 compilation
of its original theorem statement using the entire
chosen tactic sequence.

## Model, preregistration and frozen checkpoint

The actor directly imports the **original LR-01
`TypedLoop` cell** from
`LoopedTransformers/experiments/lean_reasoner/train.py`.
It has exactly **62,401 parameters**, four tied
recurrent passes, and eight active tactic candidate nodes.

Two seeds (17,29) trained for 300 fixed AdamW steps,
batch16, lr=.002 on actual compiler-emitted training
state tokens plus the eight source-trial action
progress/closing labels. Source gold reference proof
steps were used only to locate **training** prefixes
and never at evaluation time.

Frozen original model training: run
[#43541360928](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43541360928),
whose training job completed before the held-out selector
was read by its dependent evaluation job. Checkpoint
SHA256s (verified independently):
- seed17:
  `f79fcbbdd4d86edb77d7b43ade42198bdde3ac7b0f527d181ee781df848af365`
- seed29:
  `f9c052406e08d5bbe24712f649ce564c772a07e4b7e9f448e21d39eae848b182`

The original first model CI's native `first | ...` comparator
generated malformed Lean parser syntax and produced
**zero valid baseline certificates**. Those native
scores are **INVALID for a confirmatory comparison**.
Subsequent native smoke CI
[#43542473933](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43542473933)
proved that a validated Lean-native `first` syntax
can compile actual examples. A later source CI
[#43543131941](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43543131941)
validated tactic aliases that preserve the original
action order with the grammar-sensitive
`intro` and `constructor` actions.

The **first valid registered scoring** used the
exact original frozen model checkpoints and corrected
native syntax, through the separate
[eval-only CI #43543404745](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43543404745).
No model weights, held-out source or test answer
labels were changed.

## First held-out CPU-limited result

Each proof is one ORIGINAL MathCan theorem, with
at most three model-selected tactic steps. Every
positive result shown here passed a separate
Lean4.35 kernel compilation without `sorryAx`.

| Policy | Intervention /6 | HookState /4 | GraphAuthorization /4 | Total /14 |
| --- | ---: | ---: | ---: | ---: |
| LR-01 seed17 | 0 | 0 | **4** | **4** |
| LR-01 seed29 | 0 | 0 | 1 | 1 |
| Native first, seed17 model-time allowance | 1 | 0 | 2 | 3 |
| Native first, seed29 model-time allowance | 1 | 0 | 3 | **4** |
| Original source proof scripts (reference ceiling) | 6 | 4 | 4 | 14 |

Two native compiler calls timed out on seed17's
time allowance, and one timed out under seed29's
allowance. One independent post-hoc source audit
already recompiled **all four GraphAuthorization
arithmetic theorems** successfully with a
nonbinding twenty-second native limit.
The actor's only accepted outputs are one-step
arithmetic discharges. It completed **zero**
held-out genuinely multi-step Intervention or
HookState proofs.

Paired certified successes on exactly the same
held-out theorem instances:

| Seed | Student only | Native only | Both certify |
| --- | ---: | ---: | ---: |
| 17 | 2 | 1 | 2 |
| 29 | 1 | 4 | 0 |

**Preregistered learning criterion:** formally
seed17 4/14 > native 3/14, so the registered
`any seed` count predicate is **true**.
The single-proof margin is explained by deadline
sensitivity and does not substantiate learned
proof-search advantage over Lean-native automation.
Seed29 is materially worse.

## Measured end-to-end CPU time

The per-theorem native comparator's maximum runtime
was set to the **actual measured model time** spent
on recurrent inference, scratch Lean probes and
final Lean compilation on that same problem.
The full cost includes repeated Lean process
launches. The native baseline sees only that
time scalar, original theorem statement and
registered tactic bank, never model actions
or score values.

| Seed | Model total wall ms across 14 | Native total wall ms | Model ms/accepted theorem | Native ms/accepted theorem | Native timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 22,920.84 | **14,475.85** | 5,730.21 | **4,825.28** | 2 |
| 29 | 32,006.77 | **14,662.53** | 32,006.77 | **3,665.63** | 1 |

**Preregistered compute criterion FAILED for both
seeds**. The native search spends less total
time, with greater or comparable theorem
coverage in this tiny sample. The precise
CPU times and native timeouts depend on Lean
startup and CI runner scheduling; this is
not a hardware-independent FLOP comparison.

## Independent artifact audit

The first valid evaluation artifact
`fc11-verified-frozen-replay-43543404745`
contains the **14 source theorems**, their
29 compiler-printed tactic states,
28 original model response records
and 28 native-search records.
An independent container audit verified:
- the frozen train checkpoint SHA256s,
  preregistration SHA and test-goal
  JSONL SHA256;
- all 14 unique source theorem IDs
  in each model output, with a max
  of three registered tactic actions;
- each model accepted claim explicitly
  indicated a *sorry-free final Lean
  compiler return*, with accepted counts
  4 and 1 respectively;
- all native time-allowance scalars
  exactly equal the matching model
  theorem-run elapsed milliseconds;
- per-case elapsed totals, timeout
  counts and paired success counts
  reproduce the CI result ledger.

[Machine-readable independent audit](fc11_independent_audit_43543404745.json)
preserves the check. True kernel authority
is the successful pinned Lean4.35 CI run;
the archive replay checks consistency,
source hashes and unsafe-result leakage.

## Meaning and next experiment

This is the first RefutationBench FC
campaign to train and evaluate
the original compact LR-01 on
actual intermediate tactic states
of **existing research proof files**.
The source dataset and certificate
pipeline are real, and the negative
transfer result is informative.

The training action vocabulary lacks
the compositional `exact`, `rw`,
`simp only`, `have`, `rcases`,
`calc` and `induction` scripts
that the held-out proofs often use.
The original test is narrow, and
its action bank cannot express
many reference derivations within
three simple tactics. It offers no
evidence that the recurrent model
learns general Lean proof search.

A follow-up should define typed,
parameterized tactic actions,
use original theorem subgoal
states for training, and evaluate
against Lean-native automation
using a realistic shared runtime
and no source-reference proof
leakage. The original theorem
source itself is a 100%-correct
reference, distinct from student
search performance.

No claim concerning an unformalized
mathematical manuscript or an
open conjecture is made.
