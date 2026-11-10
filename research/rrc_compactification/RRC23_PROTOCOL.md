# RRC-23: executed scaling and resource-audit protocol

**Executed November 10, 2026.** This is an exploratory CPU benchmark planned with RRC-22's already exposed outcomes in view. It reuses known frozen checkpoints and the exact degree-12 graph classes, and introduces new longer input series and a published four-token grammar. It is a systems comparison with fixed source choices, not an externally blinded or multiply replicated machine-learning study.

## Questions and frozen interfaces

Does extending the finite coefficient horizon create a CPU break-even for selective neural deferral? How does a staged exact verifier change the answer? What happens when the number of syntax programs increases from 64 to 256? Report absolute accepted-prefix counts, withheld query-future accuracy, and separate computation costs.

The RRC-21 detector and RRC-19 model weights are untouched. At inference they see the first 13 exact coefficient values of two public input/output demonstrations. The heldout coefficient verifier receives longer prefixes at degrees 12/24/48/96/192, and at degree 384 on a smaller fresh sample. Its known graph topology is public, its candidate execution is exact, and its source-program label/query future remain grading-only. The frozen RRC-18 class representatives are certified **only to degree 12**, and higher-degree reuse is a tested optimization, with full syntax search as an independent reference.

## Primary deterministic roster

- Primary: 84 unique full-length examples, exactly three per topology/source-family cell. Seven families are independent signed units, unrestricted periodic partition products, sparse signed units, reciprocals of sparse polynomials, squares of sparse polynomials, sign-twisted products and high-degree-onset series. Four public wiring types are crossed with these families. All source series are generated once to degree 216, then truncated repeatedly. Eight examples share an identical **degree-12 visible signature** with a different longer series, deliberately retained as late-onset ambiguity. Full-length series are distinct.
- Five horizons: 12, 24, 48, 96, 192. For each use the same first 12 coefficients in the frozen neural models and public demonstrations through the new degree in the exact verifier. Independently grade a new query's degree-(T+24) output.
- Secondary long-horizon stress: eight fresh cases to degree 408, two per topology, with degree-384 verification and withheld coefficients through 408. This is a separate small-sample check, rather than an extension of the exact same 84 source trajectories.
- Enlarged grammar: forty fresh four-token source programs, ten per topology. The extra public pre-operation belongs to `I,N,S,R`; all 256 four-token spellings are available to exhaustive search. Test observed degrees 12, 48 and 96. The frozen model's old three-token ranking is reused and expanded by a hand-coded operator-ordering rule. The original model was never trained for this new fourth token.

## Methods and actual resource contract

- Complete certified ordering: class-mass-prior order from RRC-18, each candidate checked on both public demonstrations through degree T.
- Progressive certified ordering: same exact candidate order and equality criterion, with reexecutions at degree 4, 12 and T and honest counts for all stages.
- Selective full deferral: the frozen RRC-21 OOD/entropy router proposes either a 12-candidate learned list or the complete certified class list. The same direct/progressive verification comparison applies to it. Note that **nondeferred cases can abstain** before a matching candidate is reached.
- Full 64-syntax search: independent larger-work comparator to ensure true-grammar coverage. For the four-token grammar, include 256-syntax complete search, 4×old-class candidates and a frozen twelve-candidate four-way expansion.
- Each candidate execution on two demos counts as one candidate *query*. Every visited short or long verification stage counts two separate interpreter executions. Store distinct attempted candidates, arithmetic-work proxies by topology/degree, accepted and future-correct flags, and input/trace SHA-256 identities.

The measured profilers use warm `perf_counter` wall-clock samples, two rounds over each 84-case horizon, and independent setup costs for model+calibration loading and symbolic-class construction. The four-token physical sample uses eight cases and one round at degrees 12/48. Degree-384 physical timing uses eight cases and one round. Fresh-process memory samples compare progressive full and selective on 84 degree-192 cases and 8 degree-384 cases, using Linux peak RSS. Host-specific times and memory never enter byte-for-byte CI comparison. The reference CPU profile was assembled from three shorter timing sessions with unchanged source hashes after a combined profiler invocation exceeded the execution window.

## Verification gates

1. Exact source generators return normalized integral unit series. Metered arithmetic agrees with the public graph interpreter across all four topologies. The four-token wrapper composes precisely with the three-token interpreter.
2. For a fixed order, direct and progressive verification yield exactly the same accepted graph, attempted candidates and future result. Every stage execution is counted; no free final verifier call is available.
3. All 84 primary cases are reachable by full 64-syntax search at every horizon, all 40 extended cases by full 256-syntax search, and all eight degree-384 cases by full 64-syntax search. Distinguish finite acceptance and withheld future correctness.
4. The policy cannot use source graph labels, unseen outputs or future predictions. Same-visible future mutants produce identical predictions. Model observations remain degree-12 at every horizon.
5. Check the degree-12 class members against independent longer inputs, but do not promote finite probe checks to a new universal equality proof. Report differences between class-ordered and full-syntax future predictions.
6. Store a source/checkpoint-hashed deterministic JSON receipt and a separate source-bound *physical* timing/MiB profile. Unit tests validate schema, measured positivity, work/call counts and limits without freezing nanoseconds or RSS.
7. A computational break-even claim requires reduced actual whole-path latency under an explicitly reported accuracy loss. A strict performance claim requires matching the exact reference's coverage and accounting for neural training/calibration and symbolic class compilation. Where the measured crossover is absent, retain that result.

## Reproduce

From repository root, Python 3.10+ standard library, with the previously committed frozen checkpoint JSON files:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_scale23.py' -v
python research/rrc_compactification/bench23.py --check
python research/rrc_compactification/bench23.py --smoke
python research/rrc_compactification/memory23.py --out /tmp/rrc23_memory.json
python research/rrc_compactification/bench23.py --profile --rounds 2 --out /tmp/rrc23_physical.json
```

The deterministic replay rechecks the 84-case, five-horizon data; the 40-case expanded grammar; the eight degree-384 stress cases; and source/checkpoint hashes. The full physical profile may be split by horizon on compute-constrained runners; the checked physical record lists its different measurement batches. The CI workflow runs the exact-replay and small physical smoke, without retraining or network downloads, and does not impose a speed threshold.
