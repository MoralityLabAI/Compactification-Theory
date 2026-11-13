# SUD-01: reproducible finite receipts

Date: 2026-11-13. Standard-library Python; 7 unittest cases pass.
Code uses exhaustive finite sums, with binary64 entropy evaluation.

The command

    python reference.py --n 4 --p 0.1 --exhaustive

evaluates all 12,870 balanced truth tables on four Boolean inputs.

| Statistic | Enumeration | Analytic bound |
|---|---:|---:|
| Minimum H(S given Y) (bits) | 0.4689955935892813 | h2(.1) = 0.4689955935892812 |
| Minimum Bayes bit error | 0.10000000000000003 | p = .1 |
| Maximum I(S;Y) (bits) | 0.5310044064107187 | 1-h2(.1) = 0.5310044064107188 |

At p=.1, Cantor ratio r=1/3, and T=12:

| Balanced semantic bit | Bayes bit error | Posterior Hausdorff dimension | Optimal 12-bit recovery |
|---|---:|---:|---:|
| Coordinate | .100000 | .295903 | .282430 |
| XOR of 2 bits | .180000 | .429081 | .092420 |
| XOR of 3 bits | .244000 | .505772 | .034855 |
| XOR of 4 bits | .295200 | .552281 | .015024 |
| XOR of 8 bits | .416114 | .618058 | .001570 |
| XOR of 16 bits | .485926 | .630569 | .000341 |

The unconditional Cantor measure has dimension
1/log2(3) = .630930. Its posterior still has positive dimension
even with an infinite sequence of bit-noisy observations.

The parity formula uses \(p_n=[1-(1-2p)^n]/2\).
Success is \((1-p_n)^{12}\), while posterior dimension is
\(h_2(p_n)/\log_2 3\).

These are null-model results. No neural checkpoint, Lean kernel
or real semantic-probe noise model was evaluated.
