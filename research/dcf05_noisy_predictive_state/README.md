# DCF-05: Noisy predictive-state compactification

2026-11-25. Exploratory CPU-only study. 15 stochastic hidden-state systems, 66 trained checkpoints, 333 primary metric rows and 90 fresh quantization rows. No accelerator, energy or general language model measurement.

## Model and mathematical boundary

K initially independent fair bits. At each step choose action bit j, command c in {0,1}, read bit r (r=j half the time, otherwise uniformly sampled). Independent dynamics: flip bit j by c XOR Bernoulli(noise), with system-specific idle and commanded error rates. Coupled dynamics: replace c by c * s[(j+1) mod K] before XOR noise. Observe read bit XOR emission noise. Rates drawn from idle U(.015,.07), command-error U(.025,.15), emission U(.07,.20). Predict the current noisy observation before seeing it. Learners receive action/read indices and noisy prefix observations, without hidden bits or exact rates. Exact oracles receive the law. Both filter families receive strong structural priors; coupled mean field is given the actual neighbor graph. Empirical baseline is given bit identity and parity features.

For independent dynamics a product prior is preserved by each transition (single-coordinate kernel) and observation (single-coordinate likelihood). By induction the posterior remains a product. A general joint belief has 2^K-1 degrees of freedom, whereas the product family has K. For selected coordinate, transition q'=q+f(1-2q); prediction p=e+(1-2e)q'; after observation y, q''=q' l1/(q' l1+(1-q') l0). Others unchanged. Exact coupled oracle propagates the full joint distribution. Mean field substitutes neighbor marginal for the random controlling bit and ignores posterior correlations, so generally loses sufficient-state information. No result on the general dense matrix multiplication exponent follows.

## DCF-05A/B: training and held-out prediction

K=4,8,16 independent; K=4,8 coupled; seeds 211,223,227, independently drawn rate vectors by K and seed (shared rates across dynamics families where K matches). Each system has 2048 length-16 training words (32768 labels), shared across arms. Each learned model uses 200 Adam updates, batch64 sampled with replacement (204800 label uses). Factor and mean field learn 3K parameters, lr .035; GRU16/64 lr .003; BCE and gradient clipping. Empirical count model fits the same corpus with 96 pooled feature bins, Beta(1,1) smoothing, last observed bit, intervening counts of idle/command clipped at 3 and command parity. GRUs get one-hot action/read indices, command and previous observation. Independent test streams at L16/64/128, 256 words per system/length. Each L128 cell averages 3 systems / 98304 predictions; seeds, not tokens, are the unit for generalization claims.

L128 NLL (nats per noisy observation):

| system | exact oracle | learned factor | mean-field graph | empirical | GRU16 | GRU64 |
|---|---:|---:|---:|---:|---:|---:|
|4 independent|.519217|.519671|.593499|.571355|.689765|.688190|
|8 independent|.532913|.534424|.597363|.583435|.691907|.691898|
|16 independent|.553740|.557794|.607546|.597396|.692969|.694259|
|4 coupled|.531107|.591866|.549102|.610443|.680931|.681279|
|8 coupled|.545118|.596589|.562245|.617185|.687631|.689458|

Chance NLL=ln2=.693147. Conditional oracle KL for factor K4/8/16 independent: .000509/.001400/.003960; mean-field K4/8 coupled: .017987/.017508 vs independent filter .061693/.051709, recovering 70.84%/66.14% of KL gap. K16 factor 48 params, GRU16 2561, GRU64 19457. K16 factor accuracy 71.932%, oracle 72.138%; ECE .01094 vs oracle .00438. NLL, oracle KL, Brier, calibration and confidence coverage retained per system.

## DCF-05C: checks and sensitivity

Fresh L32 validation (256 words per system) selects lower NLL of two specified filter families. All 15 select appropriate structure. This is model selection, not discovery of unknown graphs. At L128 paired 2000-resample word-bootstrap intervals for factor minus mean-field NLL exclude zero in expected direction in every system; exploratory within-system intervals, no multiplicity correction or population-level inference. All 60 base learned checkpoints pass causal-prefix test: flip observations from step index8 onward, predictions at indices0..8 remain unchanged (max difference 0). Separate FP64 joint oracle agrees with factored oracle on six independent K<=8 systems, max abs diff 7.77e-16; joint oracle not executed for K16. Numerical cross-check separate from induction argument.

Sensitivity arm trains GRU64 for 1000 updates on same fixed corpus in six K4 systems. NLL worsens even at fresh L16: independent .688999 -> .774807, coupled .678325 -> .698681; at L128 independent .688190 -> .804711, coupled .681279 -> .710325. More updates alone do not close gap; cause not isolated. No hyperparameter search for stronger GRUs. Structured advantage conditional on supplied domain knowledge and training setup.

## DCF-05D: state precision

Separate fresh 256-word L128 stream per system for validation-selected filter. Replay in NumPy with learned FP32 params and FP64 arithmetic, max prediction discrepancy against PyTorch 2.694e-6. After each update round persistent marginals to multiples of 1/(2^b-1), initial .5 exact, weights and intermediates unquantized. Simulated budget K*b bits excludes parameters/temporaries; no packed hardware implementation.

| system | 2-bit NLL | 3-bit | 4-bit | 8-bit | full |
|---|---:|---:|---:|---:|---:|
|4 independent|.559421|.535356|.523872|.520562|.520537|
|8 independent|.574492|.543413|.537030|.534819|.534818|
|16 independent|.595781|.567010|.561138|.559245|.559246|
|4 coupled|.630105|.566564|.555402|.551843|.551784|
|8 coupled|.623111|.572438|.565186|.562341|.562344|

K16 at 4 bits =64 persistent state bits, NLL increase .001892. Eight-bit mean NLL changes by <=.000059 across families. Finite-horizon losses are empirical. Chain-rule route for future statistical verifier: expected cumulative conditional KL equals KL between trajectory laws under a fixed common action policy; Pinsker bounds total variation by sqrt(KL/2). Estimated KL here cannot certify a uniform bound over inputs or adaptive policies.

## Reproduction

Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, one thread per process. From this directory:

    python gym.py --out run
    python meanfield.py --out meanfield_run
    python gru_long.py --out long_run
    python audit.py --root .
    python quantize.py --root .

Training-loop wall-time sum 106.24s across 66 fits (processes overlapped, includes contention; not benchmark). No paid compute. Full raw receipts, checkpoints, source and SHA-256 manifest in conversation artifact dcf05_reproducibility.zip (SHA256 8dac42504ead4487bc913478096777bd784e6aafa2b842f4e707d6bc6c6987ac).

## DCF-06 proposed

Learn coupling graphs from noisy trajectories with graph hidden. Sparse edge hypotheses, held-out interventions, validation-locked graph-selection penalty, false-edge controls and exact small-system oracle. Compare learned, known, shuffled graphs, independent baseline, stronger neural predictors and adaptive cluster filters. Increase cluster size or abstain when predictive KL or calibration fails. Charge graph search, fitting, observation labels, persistent state bytes, uncertainty, wall time and query costs separately. Register a fresh family and quality threshold before evaluation. No claim of scalable AGI.
