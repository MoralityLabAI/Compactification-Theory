# Causal dimensional atlas and math-feedback gym (SCM-15)

[Theory](SCM15_THEORY.md) | [Executable gym](gym/README.md)

This track connects the SCM-01–14 source-certified Seifert/activation geometry campaign to sparse feature decomposition and a visit-conditioned, low-rank VPD-style adapter. It trains and evaluates a tiny recurrent transformer on exact synthetic algebraic programs, supports the separately pinned EvalMath ARG-01 oracle as an optional backend, and generates new training/evaluation challenges from independently checked discovery failures.

The primary evaluation remains exact mathematical correctness, split by source family and charged verifier calls. Geometric atlas measurements (SVD, feature supports and finite-difference curvature) remain observational until matched causal interventions and source certificates pass. No trained RMP checkpoint is bundled, and the current local demo is synthetic.

Run from repository root: `python -m unittest discover -s research/dimensional_atlas/gym/tests -v` and `python -m research.dimensional_atlas.gym.run --steps 40 --output /tmp/scm15.json`.
