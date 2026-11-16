# SCM-20: causal-consistency training for native VPD mediation

This preregistered follow-up tests whether a differentiable donor-write
interchange objective can improve mediation while preserving SCM-19 clean
accuracy. The same rank-four adapter trains; the reasoning model, SAE and head
fiber stay frozen. A lambda-zero arm matches parameters, training pairs,
updates and three-forward training computation. Selection uses development
cases only, before all models/directions freeze and larger fresh OOD pools are
generated. See [PROTOCOL.md](PROTOCOL.md) and [config.json](config.json).

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
