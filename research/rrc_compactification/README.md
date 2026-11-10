
## RRC-19: family-balanced certified quotient ranking

[RRC-19 report](RRC19_REPORT.md), [executed protocol](RRC19_PROTOCOL.md), and [theory](RRC19_THEORY.md) compare equal-budget balanced and skewed 6,240-parameter GRUs and a matched 6,235-parameter feedforward model. Training spans signed, product and sparse sources with the exact RRC-18 class checker. On fresh product sources, balanced GRU solves 134/140 and skewed GRU 119/140; on **new rational and squared families**, the skewed model wins 226/280 to 215/280. The [CPU replay](bench19.py) retains the failure of the broad source-robustness gate, exact negative/collision audits, class-compiler costs, independent checkpoint retraining and post-primary seed checks.

## RRC-23: measured scaling and exact verifier break-even

[RRC-23](README_RRC23.md) holds the existing neural router and degree-12 class certificate fixed while scaling exact formal-series checks to degree 192 (plus an eight-case degree-384 stress) and expanding the public syntax to four tokens/256 programs. The staged complete symbolic method is faster than staged selective neural deferral at every measured horizon. Exact results, physical timing, peak RSS, and source-hashed replay are stored separately; higher-degree class equality is empirically checked rather than newly proved.

## RRC-24: delayed q-adic factors and exact candidate search

[Results](RRC24_REPORT.md), [algebraic analysis](RRC24_THEORY.md), [protocol](RRC24_PROTOCOL.md), and [reproducibility guide](README_RRC24.md) compare a learned four-base ranker with an exact degree-16 ordering and complete delayed-factor recovery. Thirty-two delayed choices per base have identical coefficients through degree 63. The two-query GRU identifies 195/240 held-out programs, while exact prefix ordering identifies 217/240 and uses less measured CPU time. The suffix parameters are reconstructed by a supplied exact algorithm; the experiment's trained model ranks base operators only.
