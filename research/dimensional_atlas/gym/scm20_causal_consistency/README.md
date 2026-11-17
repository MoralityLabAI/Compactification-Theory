# SCM-20: causal-consistency training for native VPD mediation

This preregistered follow-up tests whether a differentiable donor-write
interchange objective can improve mediation while preserving SCM-19 clean
accuracy. The same rank-four adapter trains; the reasoning model, SAE and head
fiber stay frozen. A lambda-zero arm matches parameters, training pairs,
updates and three-forward training computation. Selection uses development
cases only, before all models/directions freeze and larger fresh OOD pools are
generated. See [PROTOCOL.md](PROTOCOL.md) and [config.json](config.json).

Completed: the combined clean-preserving mediation objective was not achieved.
The selected lambda-one, 4,000-update model reached 6.2–10.9% ID and 3.4–10.0%
OOD native-write mediation across seeds, versus zero for warm and matched
controls. Clean preservation failed in five of six seed/split comparisons.

| Mean fresh clean accuracy | SCM19 warm | Matched control | Causal |
|:---|---:|---:|---:|
| ID | 63.2% | 65.2% | 60.1% |
| OOD | 44.3% | 45.1% | 41.1% |

The 1,024-pair OOD pools yielded 120/142/116 common-eligible cases; the first
128 were assayed when available. This resolves the earlier sample-size shortfall
for these fixed pools but does not validate the causal hypothesis. See
[RESULTS.md](RESULTS.md), [THEORY.md](THEORY.md), and
[results/summary.json](results/summary.json).

All computation stays local on the RTX 3050: FP32, batch <=8, a 3000 MiB
allocator cap and 500 MiB sampled free-memory guard. Heavy artifacts and fresh
gold remain outside Git. Prior experiments and oracle bytes are unchanged.

```powershell
python -m unittest discover -s research/dimensional_atlas/gym/scm20_causal_consistency/tests -v
python -m research.dimensional_atlas.gym.scm20_causal_consistency.campaign --private ../scm20_private --oracle ../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py --old-private ../../outputs/scm19/private_artifacts
```

The registration seals implementation/configuration, oracle and warm-start
hashes before training/development scoring. Completed-stage receipts prevent
overwriting recorded outputs. A new study needs a separate results namespace.

The local worker completed without recovery or exceptions. Eight tests pass;
the independent CPU replay checked 126,624 predictions, 720 mediation cells,
4,096 fresh public signatures and nine frozen models. Peak PyTorch allocated/
reserved VRAM was 24.48/30.00 MiB, with 3245.7 MiB minimum sampled free. Worker
and verification oracle/cache accounting are recorded separately.

After private artifacts are delivered under local outputs, replay with:

```powershell
python -m research.dimensional_atlas.gym.scm20_causal_consistency.tools.audit --private ../../outputs/scm20/private_artifacts --old-private ../../outputs/scm19/private_artifacts --oracle ../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py
python -m research.dimensional_atlas.gym.scm20_causal_consistency.tools.report --private ../../outputs/scm20/private_artifacts
```

The OOD family/topology coupling and small encoded-demonstration overlap are
quantified in the results/audit. All public case signatures are disjoint from
earlier roles; exact encoder-observation novelty is audited separately because
the public query is not encoded. Historical `model.py`/`tasks.py` runtime CRLF
byte pins differ from their LF Git blobs; the delivered bundle preserves exact
registered runtime bytes with a manifest, leaving historical source untouched.
All publication commits use `[skip ci]`; no hosted experiment is requested.
