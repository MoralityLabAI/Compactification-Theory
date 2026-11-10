# OIC-20: minimum labeled sensors for hidden modular shifts

**Executed exact integer reference, November 10, 2026.** The test is deterministic; it does not train a model.

The original one-marker observer only recovers gcd(n,d). OIC-20 adds labeled marker sensors and asks for the minimum number required to identify **every** shift d from a complete length-n active observation word, given a uniform unknown initial phase.

| n | Minimum sensor labels | Witness positions |
|---|---:|---|
| 2 | 1 | 0 |
| 3 | 2 | 0,1 |
| 4 | 2 | 0,1 |
| 5 | 2 | 0,1 |
| 6 | 2 | 0,2 |
| 7 | 2 | 0,1 |
| 8 | 3 | 0,1,2 |
| 9 | 3 | 0,1,3 |
| 10 | 2 | 0,2 |
| 11 | 2 | 0,1 |
| 12 | 3 | 0,1,4 |
| 13 | 2 | 0,1 |
| 14 | 2 | 0,2 |
| 15 | 3 | 0,1,6 |
| 16 | 4 | 0,1,2,4 |

The prime-power sensor theorem gives a closed-form minimum: m=a for 2^a and m=a+1 for p^a with odd p. Every proper divisor marker plus the anchor at zero is a universal sufficient design with τ(n) sensors, although these searches show it is often not minimal.

**Verification:**

```sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic20_markers.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic20_markers.py' -v
python research/observational_inwardness_v1/oic20_run.py --check
```

The code enumerates complete observation-distribution histograms rather than treating a favorable initial hidden phase as known. It also tests invariance under global phase translation. Results belong to this finite labeled-sensor model and are separate from OIC-19's trainable recurrent observer.
