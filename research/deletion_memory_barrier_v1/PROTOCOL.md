# DMB-01: prospective semantic trace monitor protocol

The executable mathematical phase uses an explicitly uniform n-bit source and iid independent deletion traces. This protocol governs later neural and proof-trace experiments.

## Inputs and observability contract

Freeze each source identifier, generator revision, original event encoding, deletion probability q, retention mask distribution, and the exact output alignment metadata exposed to the monitor. The default deletion observer sees trace words and their lengths, with retained positions hidden.

Track the monitor's B persistent bits, public clock, stopping time, cached raw traces, metadata from tools, static model/code parameters and any prompt-specific decoder lookup tables. A terminal decoder may read only the resources registered in the run. A model that can reread old prompts or inspect a complete server transcript has a broader interface.

## Benchmark axes

Use n={8,16,32,64} for controlled synthetic words, q={0.2,0.5,0.8}, m={1,2,4,8,16,32}, B={1,2,4,8,16,32,64} where each budget is feasible. For proof-event streams retain the native source distribution; estimate its support entropy and word correlations separately. Include uniform independent words, short-deck collision pairs, parity-constrained words, repeated words, and real source traces.

Use the full-trace MAP estimator only on small n/m where exact enumeration is possible, with a fixed approximation budget if generalized. Compare finite-state rolling hashes, source-trained classifiers with charged code/weights, terminal recurrent memory and a full raw-trace observer. Equalize compute and total stored bits separately. Record every decodable source and every indistinguishable hard pair.

## Falsification and statistics

For a fixed uniform source and declared terminal state, average recovery above the DMB-1 cardinality ceiling is a bookkeeping or implementation error. For a declared worst-case hard pair, success on both sources must respect the TV coupling ceiling. If a model appears to exceed either, audit raw-trace access, stopping time and model-side caches.

Only finite certified state spaces and exhaustively enumerated trace laws support exact claims. Sampled recovery rates require confidence intervals clustered by independent source word, with held-out generator families. Success of a structured source does not test the uniform-source ceiling.

Neural experiments should preserve original checkpoints and frozen source hashes. Real proof traces may contain structured redundancy; uniform-source conclusions require independent justification. Side channels, oracle verification, retrieval, and dynamic querying each need separate resource entries.

## Future trained model target

LoopedTransformers SEM-ECHO and compactification circuit oracles can emit per-loop symbolic labels. Freeze a semantic event alphabet and sample fresh problems. Apply deletion after event extraction while preserving the original input and checkpoint, then score exact trace reconstruction and continuation decisions. Report the ability to predict the right answer without reconstructing the deleted trace as a separate endpoint.

No trained model results are claimed by the initial DMB-01 branch.
