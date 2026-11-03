# Algernon: a neural-net processor, a learning computer

**Status: research scaffold, 2026-11-03.** This folder specifies a proposed experiment. It does not claim a working Qwen graft, a trained recurrent processor, or an independently reproduced weight edit.

## Goal

Build and evaluate a compact model in which exact computation and trained recurrent reasoning cooperate *inside the model*. Work from three established ingredients and ask a falsifiable question:

1. **Qwenputer** has compiled an RV32 computer into 11 added transformer layers ahead of Qwen3.5-4B. Its DeltaNet state holds program and data memory; compiled weights execute arithmetic and instructions. The released driver initializes firmware and checks execution against a reference.
2. **HybridTRMLDT** has recurrent latent proposers and explicit projection/verification machinery. Earlier CoT-embedding and looped-transformer checkpoints are **candidates to inventory**, not presumed successful components.
3. **Weight-Space Programming** has shown a sparse intervention in Qwen3.5-0.8B that reduces pathological generation loops. The published fixed edit changes 64 MLP down-projection columns at each of layers 15 and 19; it does **not** demonstrate insertion of a new algorithm.

**Algernon's first question:** Can we graft a verifiable recurrent algorithm into a pretrained Qwen representation, preserve its previous competence, and increase performance with additional internal refinement steps at a measured compute cost?

The phrase "learning computer" here means an operationally tested adaptive neural system, with independently measured learning or transfer. An exact compiled CPU alone implements computation, not learning.

## Distinct tracks

| Track | Starting model | First milestone | Evidence required |
| --- | --- | --- | --- |
| **C: compiled computer** | Qwen3.5-4B / Qwenputer | Replay one exact circuit on a frozen workload | CPU reference match, exact trace receipts, bytes and latency |
| **R: recurrent graft** | Frozen Qwen3.5-0.8B initially | Insert a tiny trained recurrent refinement operator through a reversible bridge | 0/1/2/4/8-step ablations, matched-parameter feedforward control, held-out success |
| **W: sparse receiver repair** | Qwen3.5-0.8B edited/base pair | Reproduce authors' fixed patch, then analyze recipient layers | Checkpoint hashes, fixed roster, matched random edit, loop/cut and accuracy metrics |
| **J: joint processor** | Model chosen after the other gates pass | Route exact operations through a compiled kernel and heuristic tasks through a recurrent latent operator | Causal ablations of each component, retained base skills, total token and GPU cost |

**Important incompatibility:** Qwenputer's current RV32 construction targets the 4B architecture; its implementation notes say Qwen3.5-0.8B was skipped because its key/value-head layout provides only one value head per key head where the construction requires two. The successful sparse edit was demonstrated in the **0.8B** model. Treat these as distinct substrates until a new compiler/memory mapping has been verified.

Qwenputer's RV32 text-only BF16 merge is reported at 18.7 GB and requires about 31 GB host RAM to build. A 16 GB RTX 5080 cannot hold that full checkpoint entirely in its VRAM at BF16. Use its reference machine for CPU-first verification and budget any GPU deployment separately. Quantization changes require their own exactness checks.

## Why this belongs in Compactification Theory

The repository's adapter-depth experiments found that intervention **placement** and the **rank** of available features can dominate adapter parameter count. Algernon tests whether the same effect determines where a computational graft becomes accessible to Qwen. Measure representation rank, Jacobian sensitivity, effective bits per weight, and gradient conditioning at candidate receiver sites. Those measurements are hypotheses and diagnostics, not evidence that the old Qwen3-0.6B layer indices transfer to Qwen3.5.

## Files

- [ARCHITECTURE.md](ARCHITECTURE.md): mathematical state model, integration interfaces and controls.
- [CONTINUAL_LEARNING.md](CONTINUAL_LEARNING.md): proposed epistemic controller, research pruning and Pluto-Gym cross-reference (unrun).
- [PROTOCOL.md](PROTOCOL.md): pre-registered tests, leakage constraints, metrics and decision gates.
- [AGENT_WORKORDER.md](AGENT_WORKORDER.md): concrete build sequence for the next coding agent.
- [experiment_register.json](experiment_register.json): machine-checkable hypotheses and source status.
- [validate_register.py](validate_register.py): offline check of that registration file.
- [tests/test_register.py](tests/test_register.py): CPU-only schema and anti-overclaiming tests.

## Start here

From the root of Compactification-Theory:

~~~bash
python -m research.algernon.validate_register
python -m unittest discover -s research/algernon/tests -v
~~~

This first stage is deliberately **CPU-only**, without downloads or token expenditure.

## Upstream evidence and code

- [Qwenputer original implementation](https://github.com/divagr18/qwenputer) and [RV32 notes](https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/docs/rv32.md)
- [Takehi et al. experiment logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math)
- [Frozen weight-edit reproduction draft](https://github.com/MoralityLabAI/HybridTRMLDT/pull/5)
- [Hybrid LDT/TRM architecture taxonomy](https://github.com/MoralityLabAI/HybridTRMLDT/blob/master/docs/looped_transformer_trm_ldt_taxonomy.md)
- [Compactification adapter experiments](../../README.md)

Source revisions must be resolved to full Git SHAs and hashes written into execution receipts before running model benchmarks. The registration is a planning artifact only.
