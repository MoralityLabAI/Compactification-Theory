# RRC-25: two-branch delayed suffix identification

[Measured CPU results](RRC25_REPORT.md) · [exact finite algebra and ambiguity proofs](RRC25_THEORY.md) · [executed protocol](RRC25_PROTOCOL.md) · [source-bound replay](results/rrc25_pilot.json) · [machine-specific timing](results/rrc25_reference_profile.json)

Exact programs have form `O(F)+c*q^K+c*q^(K+D)*U(F)+c*q^(K+2D)*V(F)`. The public exact header decoder finds `(O,K,D,c)` from observed coefficients, then learned rankers order the sixteen remaining `(U,V)` templates. Full integer verification follows every proposal. With four checks, recurrent recovers 101/192 future-correct, feedforward 105/192, exact previews 164/192. Complete factorized observed-prefix search accepts 192/192 and forecasts 171/192. The late-onset cases contain 28/32 genuine finite observational ambiguities.

From repository root:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_compound25.py' -v
python research/rrc_compactification/bench25.py --check
python research/rrc_compactification/bench25.py --profile --cases 4 --rounds 2 --out /tmp/rrc25_profile.json
```

Optional CPU training uses `python research/rrc_compactification/train25.py --out-dir /tmp/rrc25-retrained`. Checkpoints and exact data stay source-locked in the committed `results/` directory. The measured result is a negative recurrent-superiority test with a positive learned improvement over fixed ordering, conditional on a supplied algebraic decoder.
