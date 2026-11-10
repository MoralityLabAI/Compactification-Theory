
## RRC-19: family-balanced certified quotient ranking

[RRC-19 report](RRC19_REPORT.md), [executed protocol](RRC19_PROTOCOL.md), and [theory](RRC19_THEORY.md) compare equal-budget balanced and skewed 6,240-parameter GRUs and a matched 6,235-parameter feedforward model. Training spans signed, product and sparse sources with the exact RRC-18 class checker. On fresh product sources, balanced GRU solves 134/140 and skewed GRU 119/140; on **new rational and squared families**, the skewed model wins 226/280 to 215/280. The [CPU replay](bench19.py) retains the failure of the broad source-robustness gate, exact negative/collision audits, class-compiler costs, independent checkpoint retraining and post-primary seed checks.
