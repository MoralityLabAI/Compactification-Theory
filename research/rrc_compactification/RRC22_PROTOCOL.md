# RRC-22: reproducible physical-compute audit

**Execution:** November 10, 2026. This systems experiment reuses the published RRC-21 test roster, fixed calibration and existing neural checkpoints. It is a retrospective profiling run and does not claim an independent blind model result.

## Primary comparison

- `full_certified`: exact certified representative order with a complete up-to-33-program search. Does not deserialize the neural weights or run the detector.
- `selective_full`: source-shift p-value first, an early OOD branch to full certified search, and for remaining cases a three-model score/entropy calculation. High-entropy cases defer to the full certified list; other cases use a 12-candidate neural ranking.
- Additional arms: uniform neural top-12; selective strict-12; OOD-only full deferral. Each preserves exactly the routing and candidate budget of the prior RRC-21 implementation.

Use the same 480 positive inputs (three familiar and six novel source generators), 16 inconsistent observed-prefix negatives and six visible-equal future-collision pairs (12 records). The complete symbolic solver and selective policy both receive the same public coefficients, public topology, certificate interpreter and class tables. Future outputs and hidden graph labels are used only for independent grading after each timed inference.

## Profiling contract

1. Generate/check the RRC-21 roster and certified polynomial class map. Load the frozen model checkpoints and saved calibration for the methods that require them, and record the **cold setup** wall time. The full certified baseline loads no neural checkpoint.
2. Warm each arm on the first twelve cases. For each of three measured rounds, shuffle exactly the same 508 cases using deterministic seeds `2026111022+round`, then time each policy's routing/order generation and its exact verifier separately with `time.perf_counter_ns`. Report total, median, P90 and P99 warm elapsed times, plus `time.process_time_ns` as an independent CPU clock.
3. Run each method's memory case pass in a fresh Python process. Record Linux `resource.RUSAGE_SELF.ru_maxrss` across all 508 cases. Record current resident memory after setup and after execution from `/proc/self/status`. Use Python `tracemalloc` only for 64 evenly spaced cases to limit interference, and identify the sample size.
4. Run exact arithmetic metering **separately from the timed path**. The alternate interpreter records integer multiply/add calls, operand bit widths, maximum intermediate width and coefficient comparisons for all executed graphs. Every result is checked for parity with the original exact verifier.
5. Crosscheck every observed-prefix acceptance and withheld-future score against RRC-21's fixed source-hashed receipt. The new semantic receipt records every program trace hash, input-roster hash and defining source/checkpoint hashes.
6. Keep time and memory measurements in a **machine-specific** profile. CI verifies the schemas, numerical positivity, actual call budgets and source hashes; it must not require a timing threshold or exact reproduction of nanoseconds/RSS from another machine. CPU smoke profiling must perform real policy and worker executions.

## Accounting boundaries

A complete graph query executes both degree-12 demonstrations. The independent degree-32 future grade is outside the timed inference path and charged only to evaluation. Separate model MAC counts and the integer-work proxy; do not sum dissimilar units. Offline class compilation, checkpoint loading, OOD calibration generation, previous model training, Python interpreter/runtime overhead and graph execution all remain explicitly attributed. Training/model costs are held fixed here; this benchmark trains no new parameters.

## Reproduce

From the repository root with Python 3.10+:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_meter22.py' -v
python research/rrc_compactification/bench22.py --check
python research/rrc_compactification/bench22.py --smoke
python research/rrc_compactification/bench22.py --profile --rounds 3 --out /tmp/rrc22_machine_profile.json
```

`--check` performs a complete deterministic recomputation and compares the frozen semantic JSON byte-for-byte. `--profile` records the machine-dependent physical measurement and never asserts that any policy is faster. The reference profile was assembled from three sequential batches using `--arms full_certified selective_full`, `--arms neural12 selective12`, and `--arms ood_only_full`, each with all cases and three measurement rounds. All inherited RRC-21 model and class source files remain unchanged.

**Success condition:** a computational benefit from the selector requires a reduction in measured total latency or total cost at an explicitly acceptable accuracy loss, after accounting for OOD inference and offline compilation. The previously observed 607 saved positive-case graph queries alone do not satisfy this criterion. Document neutral and negative outcomes verbatim.
