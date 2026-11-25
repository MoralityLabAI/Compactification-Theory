# COL-COMP-01: how many bits a quotient loses about an attractor

This package measures the price of compressing a finite dynamical state to
its conserved arithmetic quotient. It enumerates every odd core state for
172 admissible parameters `d <= 1025`. The sample contains 44,204 states.
Across 85 of those graphs, one quotient label contains several basins.

The exact best decoder using q alone gets 37,317/44,204 states correct under
the pooled uniform-core distribution, or 84.420%. The parameter-macro score
is 85.065%. Every count and cell membership appears in the
[readable summary](results/summary.json).

## Setup

For positive `d = 1 (mod 4)`, odd n goes to the odd part of `3*n+d` when
`n = 1 (mod 4)` and to the odd part of `n+d` when `n = 3 (mod 4)`.
The odd core `X_d = {1,3,...,d}` is finite and closed. Each seed has a unique
eventual odd cycle. Its minimum labels the target basin B(n).

When `gcd(d,6)=1`, multiplication by 2 and 3 permutes residues modulo d.
Let q(n) denote the least representative of its orbit under those two
multipliers. Transitions preserve q. This package computes that orbit
partition directly, including residues whose gcd with d exceeds one.

A target count c(q,b) records the number of core states in basin b with
quotient q. These counts define a distribution once the seed measure is
specified. The code reports both uniform odd seeds and uniform cycles.

For example, at d=13, q identifies two classes among three cycles. A
uniform-cycle classifier reaches 2/3. Uniform odd seeds instead give 4/7.
The distinction matters whenever basin sizes differ.

## Exact finite tradeoff

Suppose a decoder receives q plus a side message with at most M possible
values. Allow the encoder to know the true basin, and order the basin counts
inside each q from largest to smallest, `c_(1)(q), c_(2)(q), ...`. The greatest
possible number of correct seed decisions is

\[
C_M=\sum_q\sum_{j=1}^{\min(M,k_q)}c_{(j)}(q),
\]

where k_q counts the basins represented in q.

**Proof.** For a fixed q, every message has one decoded basin label. At most
M basin labels can be predicted correctly. Choosing the M largest masses
bounds the correct count from above. An encoder can attain the bound by
assigning one distinct message to each chosen basin and assigning the
remaining basins to any existing message. Summing over q gives the formula.
Randomized decoding cannot improve the optimum because expected success is
linear in its output probabilities.

Consequently the least number of fixed side bits needed for zero error is

\[
b_* = \left\lceil\log_2\max_q k_q\right\rceil.
\]

This is an information budget with an oracle encoder. Computing its message
from an unfamiliar seed may require solving the basin problem itself. The
frontier deliberately excludes that computational cost and identifies it as
a separate experiment requirement.

The code also computes the minimum expected binary prefix-code length for
basin identity conditional on q using the integer Huffman merge cost. It
reports the weighted bit sum and its exact rational mean. Conditional
entropy appears as a separately marked floating-point diagnostic.

## Sweep results

| Extra fixed bits needed for perfect basin labels | Parameter graphs |
| ---: | ---: |
| 0 | 87 |
| 1 | 44 |
| 2 | 28 |
| 3 | 10 |
| 4 | 2 |
| 5 | 1 |

These are exact counts over the declared 172-parameter domain. This audit
exposes every target in that domain. Future claims about a learned encoder
need a new evaluation population and a fixed computational budget.

### Previously inspected stress case: d=2021

The anchor has 1,011 odd states, 39 cycles, and six quotient classes. Its
largest quotient contains 18 basins. A gcd-only decoder gets 110/1,011
correct; the full quotient raises that ceiling to 190/1,011. Under uniform
cycle weights, q reaches 6/39, or 2/13.

| Extra bits | Messages per q | Oracle correct / 1,011 | Fixed hash controls, correct range |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 190 | 190 |
| 1 | 2 | 285 | 190 |
| 2 | 4 | 454 | 190–191 |
| 3 | 8 | 709 | 196–203 |
| 4 | 16 | 987 | 217–227 |
| 5 | 32 | 1,011 | 253–268 |

Five ideal side bits can identify every basin. A five-bit hash of the seed,
combined with q, gets at most 268 states correct among the four fixed hash
controls. The gap quantifies how the encoder's content matters once the
message budget is fixed. Hash controls are descriptive finite encoders;
their decoder also has access to all labels for the hindsight ceiling.

For this anchor, `H(B|q)` is approximately 3.715100272743 bits. The exact
optimal prefix-code cost is 3,789 weighted bits, or 1,263/337 bits per seed.
The code selects no learned representation from this result.

## Relation to the existing compactification work

[NOR-15C](../nor15_late_source_ceiling/README.md) bounds a shared decoder's
success inside a late-source feature fibre. COL-COMP-01 supplies complete
arithmetic graphs whose fibres and target collisions can be enumerated.
The c(q,b) table gives a finite witness for each unit of decoder error.

This lets a later representation study compare a learned side message with
an exact oracle frontier at the same bit budget. The report must keep the
cost of producing the message and the decoder's label exposure visible.
A bit budget alone leaves the difficulty of discovering the correct code
unspecified.

## Next experiment contract

Use all audited parameters as development sources. Select new parameters
before training a code, retain every seed from one parameter in its assigned
split, and record families with shared factors. Compare a learned b-bit
message with fixed hash messages, branch-prefix codes, the q-only decoder,
and the oracle C_(2^b) frontier. Keep each encoder's access to d, n, exact
transitions, and target labels explicit.

Report error as an exact rational count for complete cores. Report
parameter-macro and state-weighted scores, computational cost per seed,
and the gap to the oracle at each bit budget. When the question concerns
transfer or a neural mechanism, use the repository's statistical evidence
gates with parameter families as clustered roots and fresh confirmation
sources. The current result has evidence class `exact finite domain`.

## Reproduce

Python 3.10 or newer and its standard library suffice.

```sh
python -m unittest research.collatz_quotient_frontier.test_frontier -v
python research/collatz_quotient_frontier/frontier.py \
  --check research/collatz_quotient_frontier/results/exact_receipt.json.gz
```

Regenerate with `--out /tmp/col-frontier.json.gz --summary /tmp/col-summary.json`.
The compressed [full receipt](results/exact_receipt.json.gz) contains all cell
counts, every oracle side-bit point, and all four hash controls. Six tests
include exhaustive enumeration of tiny encoder/decoder pairs, monotonicity,
a zero-error endpoint check, known distribution-sensitive ceilings, cycle
invariance, and deterministic replay.

## Source and implementation audit

The common arithmetic module comes from
[MathCan's Collatz resonance program](https://github.com/patrickdugan/MathCan/tree/research/collatz-resonance/research/collatz_resonance).
The vendored source SHA-256 is
`24ff52171aedf890c33439896ab5e71de65d9b05d557ba61d162f3623aa5dbca`.
A separately written traversal and quotient BFS agreed with it on every
state and residue in this sweep. The propositions above are mathematical
arguments recorded here; their machine-checked formalization is separate
work in MathCan.
