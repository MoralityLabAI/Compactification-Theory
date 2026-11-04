# Algernon-02: Circuit Mirror

**Status (2026-11-03): CPU prototype executed. No Qwenputer, LOTUS, Circom binary or SNARK proof was executed here.**

## Hypothesis

A trained recurrent latent processor may implement the same state transition as a formally specified computational machine. For a decoder \(\pi\), learned transition \(F\), and exact machine step \(T\), the target equation is

\[\pi(F(h))=T(\pi(h)).\]

If it holds on every reachable latent state, then induction gives \(\pi(F^k(h))=T^k(\pi(h))\). Experiments test approximate commutation on finite, held-out states; they do not establish the universal premise. The R1CS-style backend is a relation checker over a finite field, while the latent network lives in a real vector space. Decoding to discrete states and public-input binding form the interface.

## Concrete machine

One-hot pointer chasing on N nodes. A public row-onehot transition matrix \(A\) encodes a total deterministic successor function. A trace consists of one-hot vectors \(s_0,\ldots,s_K\) and quadratic intermediate wires \(p_{tij}\):

\[p_{tij}=s_{ti}A_{ij},\quad s_{t+1,j}=\sum_i p_{tij}.\]

The exact checker enforces Boolean values, row sums of 1, initial-state binding, every intermediate product and next state, and a final-state output binding. It also checks the witness's public inputs against the requested problem. Signals use the default BN254 scalar field modulus specified by Circom. The JSON triples follow Circom's R1CS constraint shape, with *our own signal indices*; compiler-optimized constraint counts and indices will differ.

The included Python emitter also generates a Circom 2 source file with equivalent intended semantics. **Compilation/parity against the actual Circom compiler is an open gate.** A valid R1CS witness is not itself a zk proof. The `circom` source must be compiled and run using a supported binary and the generated witness checked before claiming compatibility.

## Run from the Compactification-Theory repository root

No Python dependencies are needed for exactness tests:

```bash
python -m unittest discover -s research/algernon/circuit_mirror/tests -v
python -m research.algernon.circuit_mirror.run --out runs/algernon/circuit_mirror
```

Optional tiny neural CPU pilot (PyTorch, no GPU):

```bash
python -m research.algernon.circuit_mirror.pilot --updates 300 --cases 40 \
    --out runs/algernon/circuit_mirror/pilot.json
```

The toy recurrent readout has `n^3 + n` trainable parameters. It receives explicitly constructed multiplicative features `state_i * edge_ij`. It learns how to assemble a next-state prediction from those features by one-step cross-entropy training; the same readout is reused for 1, 2, 4, 8, 12 and 16 steps. Training uses random-function graphs, calibration uses permutations and held-out tests use single long cycles and sinks. The random-weight readout is a control, and the exact R1CS trace checker audits the discrete trajectories.

This setup is **strongly biased toward the correct algorithm**: the supplied quadratic features already represent the circuit products and graph size stays constant. A perfect score is an interface sanity check, not evidence of emergent general reasoning, a neural compiler, or better scaling than classical pointer chasing.

## Reference execution receipts

The current CPU run with N=4, steps=8, seed=20261103 and 48 cases per generator family exports a full receipt. The optional pilot used N=6, 300 updates with batch 128, and 40 cases per family per evaluated recurrence depth. Both are local CPU runs. Exact reference and neural pilot results remain separately labelled.


## Actual LOTUS hidden-state bridge (implemented; GPU evaluation pending)

[lotus_bridge/](lotus_bridge/README.md) now provides a frozen 152-case pointer roster, a pinned upstream LOTUS hidden-state capture hook, and an exact constraint-mirror readout evaluation. The capture records the initial latent state and all sixteen post-refinement states, while the linear probe trains on stages 1–8 and is evaluated on held-out generator families at stages 9–16. A short on-device instrumentation parity check runs before extraction.

The CPU tests run without loading a pretrained model. None of the published LOTUS weights or their post-loop activations has yet been executed in our environment. Start with the two-item smoke extraction in the linked README.

## Next experiments, frozen before GPU spending

1. **LOTUS replication:** Pin `yingfan-bot/lotus` source and the published pretrained 3B model. Run its existing evaluation against the upstream instructions and retain a clean reference score, question roster, intermediate-state readouts and profiling. All source locks must include licenses and full resolved model revision. The 3B checkpoint is not claimed to fit within the 5080 VRAM until measured.
2. **Learn a decoder \(\pi\):** Train a small readout from actual LOTUS post-loop latent blocks to structured pointer states, using generator-disjoint data. Compare a one-step, multi-step and unstructured readout. Keep test labels and exact verifier outputs away from the readout during evaluation.
3. **Depth and structure:** Evaluate trained depths 1–8 and held-out 9–16. Record the commuting-square error per state, exact trace validity, last-state accuracy, text-CoT baseline and measured compute. A constant-state, shuffled-state, extra-token, matched-parameter feedforward and random-direction control are required for a strong causal claim.
4. **Constraint feedback:** Use constraint violations at *inference*, but account for their information content. A detailed residual can reveal the correct next state in deterministic pointer chasing. An oracle-assisted correction must be explicitly labelled and benchmarked against an oracle/control with equal access.
5. **Compiled backend:** Port the same typed transition into Qwenputer's arithmetic/IR primitives at its pinned revision. Check trace parity against this Python reference, including corrupted state and public input cases. Memory footprint, trace tokens, latency and exact precision are mandatory.
6. **Sparse repair:** Only after a working neural receiver is established, test local weight edits with fixed support/norm budgets and held-out data.

## Pinned primary sources

- [LOTUS](https://github.com/yingfan-bot/lotus/tree/eb77e2f7909c5006f58ff0ad7cd6629b942caa9e) code (paper [arXiv:2606.31779](https://arxiv.org/abs/2606.31779)). Checkpoint/model revision still needs an explicit resolved commit.
- [Qwenputer](https://github.com/divagr18/qwenputer/tree/bec1a0c9654ac8151b6f36ed0154a19c53141384), published RV32 implementation.
- [Circom docs](https://docs.circom.io/circom-language/formats/constraints-json/) for the constraint triplet shape and [BN254 Fr](https://docs.circom.io/background/background/) field.
- [LoopedTransformers xbench](https://github.com/MoralityLabAI/jev-qwen/blob/898781fd3c11ca5dc93f5efd9d85f1ae744afc22/reports/xbench/summary.md), historical OOD-depth reference.

The reference experiment is small enough to run without external services. It is deliberately independent of the missing D-drive files.
