# RRC-14: measured reversible-edit program search

**Executed 2026-11-10, CPU pilot, one training seed.** The registered 12-query
reversible edit search recovers some withheld `S/R` graph compositions.
The primary recurrent-superiority gate **fails**: the matched feedforward
and an untrained residual-frontier policy both do better on unseen graphs.
The exact interpreter and teacher's neighbor scans remain substantial
sources of computation.

## Frozen executable-program results

All figures are **exact observed-prefix accepted programs**, independently
verified and then graded on degree-13–32 query continuation.

| Evaluation | Frozen RRC-13 one shot | Trained GRU edits | Same GRU, no history | Same GRU, no residuals | Feedforward edits | Untrained residual frontier | 64 previews / 4 checks | Full exact search |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Familiar graph classes (80) | 7 | **66** | 62 | 54 | 57 | 36 | 79 | 80 |
| New S/R compositions (160) | 0 | **53** | 36 | 18 | 63 | **78** | 159 | 160 |
| New S/R plus product-source shift (64) | 0 | **16** | 17 | 0 | 14 | 16 | 60 | 64 |
| Selected inconsistent observed pairs (20) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Identical-visible future collision pairs (16) | 0 | 0 | 2 | 0 | 2 | 6 | 16 | 16 |

All accepted programs in the first three rows forecast the held-out
future correctly. The sixteen collision records comprise eight matched
base/mutant pairs; the exact search accepts all visible prefixes but
predicts only eight continuations correctly, as demanded by the
information limit. When a weaker policy abstains on both pair members,
it also avoids future false acceptance but obtains no positive credit.

The shuffled edit-target GRU recovers 13/160 unseen compositions and
0/64 shifted cases. Untrained lexical frontier recovers 55/160 and 19/64;
the seeded random frontier recovers 50/160 and 27/64. A fixed catalogue
of the first 12 canonical graphs recovers 21/160 and 3/64. Exact search
is complete **only within the declared 64-syntax finite grammar** through
degree 12. It is not a proof of an infinite symbolic identity.

## Recurrence, feedback and evidence

The 7,353-parameter GRU and 7,393-parameter feedforward model trained on
exactly the same 960-case familiar-graph roster, matched minibatch
indices, 420 Adam updates and teacher-generated edit trajectories. The
GRU's development edit-label accuracy is 49.2%, versus 37.5% for the
feedforward model, but full-program identification is the decisive gate.
On the 160 unseen-composition cases, retaining GRU memory raises exact
recoveries from 36 to 53 within the same checkpoint; masking residual
input reduces them to 18. This shows dependence on the supplied recurrent
memory and exact-feedback path in the frozen pilot. It **does not** show
an advantage over matched feedforward (63/160) or the untrained
residual-frontier controller (78/160). On the 64 product-source-shift
cases the recurrent memory effect also disappears (16 versus 17).

The trained recurrent policy uses the RRC-13 initializer, which itself
solves **0/160** new compositions and **0/64** shifted examples on this
new roster. The gain from adding reversible edits therefore depends on
spending up to 12 fresh interpreter queries. The process is a learned
search-order heuristic over a graph language and a public algebraic
interpreter, not primitive or graph-semantics invention.

## Per-case inference resources on 160 new compositions

| Method | Two-demo graph queries | Integer-loop proxy | FP MAC proxy |
|---|---:|---:|---:|
| Frozen RRC-13 initializer | 1 | 1,014 | 12,936 |
| Recurrent edit policy | 10.63 | 10,780 | 87,590 |
| Feedforward edit policy | 10.09 | 10,235 | 84,056 |
| Untrained residual frontier, same initializer | 9.49 | 9,627 | 12,936 |
| Untrained 64-preview search, four exact checks | 65.04 (64 low-degree previews) | 10,630 | 0 |
| Complete early-pruned 64-graph exact search | 37.04 candidate attempts | 19,307 | 0 |

All edit policies have the strict **12-query maximum**; these means
reflect early verified success. Each edited graph query covers both
public degree-12 demonstrations, with one exact integer verifier and
1,014 nominal loop units. The preview procedure instead executes all
64 candidates to degree four before up to four degree-12 verifications,
so its query count is at **different input degrees**. Complete search
short-circuits after a mismatch on the first demo. Table units are
**not** wall-clock times or mutually fungible. Costs of training, exact
big-integer bit operations, storing the case data, the interpreter and
Python heap/restore overhead are omitted from the proxies. The RRC-13
prior contributes 6,148 additional FP32 parameters to every edit arm.
The supervised teacher additionally executed 35,631 neighboring training
graphs on two demonstrations apiece; its source labels are a separate
training privilege.

## Source and leakage audits

The fresh RRC-14 seed is `2026111014`, with no full observed-prefix
train/test alias. The held-out eight S/R semantic classes and test-only
source generator were excluded from optimization. All 20 selected
inconsistent observed examples are rejected by the complete finite
solver, and the adaptive random-control seed is based on **visible
coefficients**, keeping every collision pair identical under inference.
All 15 new tests pass locally; RRC-13's 16 tests also pass. PyTorch and
portable-GRU inference logits agree within 2e-5. A fresh independent
single-thread CPU training replay reproduced all three frozen checkpoint
JSON files byte-for-byte. The source-hashed JSON receipt replays the
entire 488-case development-and-evaluation score table.

This is one exploratory seeded pilot. The shared primitive semantics and
fixed graph topology were known from RRC-11/12/13. Held-out examples have
been exposed here and should not be reused as blind evaluations for
future model selection.

## Decision

The reversible-edit interface supplies a working path from zero direct
recovery to **53/160** unseen compositions with 12 interpreter queries,
while its strongest untrained 12-query search control obtains **78/160**
and the matched feedforward obtains **63/160**. The architecture and
search-efficiency hypotheses remain unconfirmed. Future work should
compare against cost-matched *learned* lookahead, test topology/primitive
shifts rather than more seeds of the same DSL alone, and account for
training-time candidate executions before claiming computational savings.
