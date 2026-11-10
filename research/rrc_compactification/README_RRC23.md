# RRC-23: coefficient-horizon and graph-grammar CPU scaling

- [Measured outcome and operation ledger](RRC23_REPORT.md)
- [Exact finite-series theory and break-even inequality](RRC23_THEORY.md)
- [Executed dataset, scoring and profiling contract](RRC23_PROTOCOL.md)
- [Deterministic source-bound semantic receipt](results/rrc23_deterministic.json)
- [Host-specific CPU clock profile](results/rrc23_reference_profile.json)
- [Independent-process Linux RSS record](results/rrc23_memory_profile.json)

RRC-23 reuses the already trained RRC-19 predictors and RRC-21 calibration. A complete exact symbolic class-order baseline and a selective router both verify public degree-12 model observations at five increasingly long exact coefficient horizons. A staged verifier checks degrees 4, 12 and T with all executions metered. A hand-coded additional unary pre-operation expands the grammar to 256 syntax programs.

**Outcome:** Progressive symbolic search beats selective progressive on all measured horizon medians, reaching 3.302 ms versus 4.606 ms at degree 192 and 17.209 ms versus 21.209 ms on a separate eight-case degree-384 stress run. The symbolic arm accepts all 84 primary observed prefixes at every degree, while the frozen selective router accepts 81. This study trains no new neural model. Long-horizon class identities remain empirical past the original certified degree 12.

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_scale23.py' -v
python research/rrc_compactification/bench23.py --check
python research/rrc_compactification/bench23.py --smoke
# Full host-specific profile:
python research/rrc_compactification/bench23.py --profile --rounds 2 --out /tmp/rrc23_host.json
python research/rrc_compactification/memory23.py --out /tmp/rrc23_rss.json
```
