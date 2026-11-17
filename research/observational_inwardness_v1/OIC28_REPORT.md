# OIC-28: Robust seven-phase adaptive sensing

**Executed November 11, 2026.** Exact finite-state experiment extending OIC-27 to seven hidden phases and one possible substituted sensor bit.

| Horizon | Outcome | Evidence |
|---:|---|---|
| 14 | Exhaustive UNSAT | 6,719,523 affine-canonical states; OIC-T44 pairwise pruning |
| 15 | Inconclusive | Search reached 10M-state cap |
| 16–18 | Inconclusive | State and wall-time limits |
| 19 | Inconclusive | Larger searches reached limits |
| 20 | Constructive SAT | 1,097,803 states; 1,697-node verified policy |

**Certified result:** \(15\le Q^*_{\mathrm{flip}}(7)\le20\).

The independent verifier evaluates 773 terminal leaves and all 1,029 hidden-world/fault-position scenarios. The positive policy's public-history branch structure is checked separately from the C++ solver. Six Python integrity and symmetry tests pass in ordinary and optimized modes. Pinned source and compressed receipts reproduce the results.

A separate SciPy/HiGHS binary integer solver reports no feasible **fixed-address Q20** sensor schedule with minimum cross-shift Hamming distance three. The LP relaxation is feasible; no independently checkable integer-infeasibility certificate was generated. The fixed Q21 case timed out. The fixed/adaptive separation is therefore supported by integer-solver evidence, with a stronger separately verified constructive adaptive upper bound.

**Resources:** Adaptive Q20 requires 20 returned sensor bits and up to 60 online address bits, or 80 total communication bits. Two exact candidate layers require 98 logical bits. Policy storage and search work are charged separately. The source archive contains the complete decision tree, C++ planner, Python verifier and exact replay manifest.

No neural model was trained. Next steps are in OIC29_PROTOCOL.md.
