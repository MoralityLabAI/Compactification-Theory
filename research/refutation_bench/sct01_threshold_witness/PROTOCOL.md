# SCT-01: exact finite SAT-witness monitoring protocol

Date: November 14, 2026.
This records the executed finite reference's frozen
rules and the streaming gym. It was assembled
during the campaign and is not presented as a
prospective registration preceding the first
source-code experiments.

## Clause model and independent distributions

Six Boolean variables are divided into two
disjoint three-variable groups. For each group,
all eight distinct signed 3-clauses are present
in the source universe, one for every forbidden
local assignment.

Bernoulli inclusion experiment: each of the
sixteen clause types is included independently
with probability p. The exact increasing event
is that at least one complete eight-clause
group is present.

With-replacement stream experiment: at each
step one of the sixteen clause types is drawn
uniformly and independently. Repeated clauses
are meaningful events but do not add any new
formula constraint. The first-UNSAT time is
the first prefix whose set of distinct clauses
contains a complete group.

The distributions have separate probability
laws, threshold objects and sample-space
definitions. Never identify a Bernoulli
inclusion probability with a clause density
or the random-SAT asymptotic threshold.

## Formal state and witness access

The exact decision monitor has 65,026 states,
encoded within sixteen live bits. Every
incomplete clause-presence set receives
its unique base-255 code; all winning states
share one absorbing state.

The certifying monitor has 65,027 states,
also within sixteen bits, distinguishing
which of the two gadgets first completed.
Its output is SAT or a valid gadget index,
which the host can check against the
original clause-presence set. The public
static decoder expands the winning gadget
index to its eight known signed clauses.

Compare two eight-bit controls: a monitor
tracking only the first gadget's eight
distinct clause types, and a monitor
counting raw arrivals to each group.
The first can miss UNSAT caused by the
second gadget. Arrival counting can give
false-positive UNSAT claims after repeats.
All errors must be recorded separately.

The source clause stream, host complete-set
SAT oracle, static gadget codebook and
certificate verifier are not included
in the declared live monitor state.
They are charged as separate resources.
The in-process Python gym is an evaluation
interface and cannot securely isolate
arbitrary malicious Python introspection.

## Executed cohorts

The seeded streaming runner uses seed
1752352026, horizons 8,16,24,31,32,40,48,64,
and 1,024 independently generated sequences
for each horizon. Each policy receives
identical clause sequences.

Per horizon report the sampled UNSAT rate,
the independent exact hitting-time CDF,
the number of correct/false-positive/
false-negative classifications, the verified
positive proofs and each monitor's state
budget. Retain SHA-256 source and dataset
fingerprints. The source CDF is a population
quantity; sample frequencies may fluctuate.

## Proof boundaries and promotion

The exact finite SAT oracle enumerates all
2^16 clause subsets and all 64 assignments.
It independently certifies that the two
eight-clause cores are precisely the minimal
contradictions. Complete enumeration supports
finite conclusions only.

The cover optimum proof includes the empty
set in the fractional-cover domain.
The Myhill-Nerode argument concerns
deterministic exact streaming with arbitrary
length and repetitions. Distributional,
randomized and approximation-permitted
monitors require distinct lower bounds.

A positive answer in a larger proof task
must include a host-verifiable contradiction
witness. A failed search for one is not
a proof of satisfiability. Later RefutationBench
experiments may train a compact reasoning
model, but must freeze source clause cases,
original model and optimizer hashes, and
independent held-out formula families before
making empirical generalization claims.

The source #175 universal comparison theorem
and #235 large-n SAT hitting-time results
remain distinct research claims. This
finite control does not reprove them.
