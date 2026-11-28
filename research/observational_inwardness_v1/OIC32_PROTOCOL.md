# OIC-32: Certified global Q20 feasibility or infeasibility

**Prospective, 11 November 2026.** OIC-31 resolves the mixed-fault optimum among fixed schedules omitting at least one marker, while the seven-phase 20-query unrestricted fixed-schedule integer decision remains without a checkable negative certificate.

## Exact objective

Find a 20-query schedule using any of all seven available sensor labels with cross-shift minimum Hamming distance at least three, or provide a complete independently checkable proof that none exists. Work with the 49-integer time-residue occupancy matrix from OIC-T61 and 1,029 exact linear distance inequalities. Separate an integer solver's own UNSAT verdict from a checkable proof. A valid SAT witness must expand to a time-address word and pass a pure-integer 1,029-pair plus every single-fault replay.

## Certificate strategy

Try a proof-producing PB/SAT encoding with an independently verified DRAT/LRAT-style unsatisfiability trace, or an exact branch-and-bound with a compact replayable search DAG. Gauge transformations and cutting planes need explicit soundness proofs and regression tests; preserve all-seven-label schedules. Keep approximate LP relaxations only as pruning bounds, using outward-rounded rational certificates for claims of impossible branches.

## Adaptive branch

Revisit OIC-28 Q19 under strict state/RAM/wall caps and the 42-element affine phase gauge. Any new pruning rule must be proved and tested against complete smaller-n searches. Save a full policy tree for SAT and a complete conclusive search record for UNSAT; cap exhaustion is inconclusive.

## Resource ledger

Charge response bits, address commands or preprogrammed instructions, policy storage, candidate belief bits, proof bytes, runtime and peak host RAM separately. The finite synthetic benchmark cannot by itself certify a learned transformer observer.


## Resolution (11 November 2026)

[OIC-T64](OIC32_THEORY.md) proves that `Q_fixed^*(p,one substitution)=3p` for every prime `p>=5`, **including full-marker schedules**. The former unrestricted Q20 n=7 SAT/UNSAT certificate gap is closed analytically: Q20 impossible, Q21 attained by the known three-block code. The separate adaptive seven-phase query minimum remains 15–20 and is the subject of [OIC-33](OIC33_PROTOCOL.md).
