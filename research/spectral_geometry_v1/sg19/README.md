# SG-19: selective multiplier retrieval without exact repeats

The SG-18 bounded certificate memory and SG-15 integer verifier remain unchanged. Each query starts with the same SG-18 feasible incumbent and exact root-vector bound. If the root gap is nonzero, retrieval may probe compatible cached vectors before chain-LP acquisition and its existing fallbacks. All candidate bounds are recomputed for the current task, and only independently certified rational vectors enter the four-record cache. A failed probe can improve the incumbent/bound; a skipped probe leaves the root certificate unchanged. No policy changes the proof checker or the production default. Memory caps count records, not bytes.

## Protocol

18 training trajectories, six per persistent/rotating/mixed regime, half cumulative drift and half anchor-relative. Each contains three interleaved families, shared structural parameters, 12 queries at d=32,T=12,k=7/8: original, 4 distinct +/-1 utility edits, 16 distinct +/-1 edits, 48 distinct +/-1..3 edits. Each phase randomizes family order; the cumulative cohort edits the immediately preceding family task. Assert every signature differs. Training uses cold acquisition to populate memory, independently probes each compatible resident candidate when the root gap is nonzero, and assigns the immediate cost proxy `net_ms=(cold_finish_ns*closed-probe_ns)/1e6`. This charges failed probes their cost and does not model their effect on later LP recovery or future cache content. Fit eight-feature root-weighted ridge (penalty 0.1), with within-trajectory shuffled targets as a control. Select the best positive predicted candidate or abstain. Tune a one-probe nearest-neighbor L1 gate over thresholds 0,4,8,16,32,64,96,128,1000000, maximizing summed training net for the nearest record. Selected threshold 128; 668 training labels, 86 closures. Fixed16 and nearest1/2 controls were defined before evaluation.

24 separate evaluation trajectories, eight per regime and 12 per drift mode. 288 distinct logical queries across the entire cohort, verified by full signature; two randomized whole-trajectory repetitions per arm, seven arms, 4032 runs. Memory resets each arm/repetition. Policies see task data, resident records and initial root gap, not family or phase identity. Timing includes initialization, retrieval, probes, fallback, exact checks and memory insertion; excludes post-timing audit and offline training. No hard deadline; CPU-only single host. After 15 completed trajectories the initial process timed out; the identical frozen model and per-root checkpoint were resumed with an added command-line resume path. The interrupted partial trajectory is discarded. The same training/model and evaluation seeds/policies remain; this is one cohort across two processes, not a confirmation run. Both executed source versions and logs are archived. No timing claim beyond this synthetic development workload.

| Policy | CPU ms | LP calls | Probes | Transfers | Certified /576 | Wins vs cold /24 |
|---|---:|---:|---:|---:|---:|---:|
| cold |10.814481|552|0|0|576|0|
| nearest2 |10.088710|338|752|214|576|17|
| nearest1 |9.286534|338|506|214|576|24|
| fixed16 |9.326490|396|188|156|576|23|
| tuned |8.889362|338|372|214|576|24|
| learned |8.925481|338|372|214|576|23|
| shuffled |10.850186|498|108|54|576|11|

Tuned reduces mean CPU 17.80% and LP calls 38.77% against cold. Paired trajectory mean CPU saving 1.925 ms; descriptive 10,000-resample trajectory bootstrap percentile interval [1.540,2.326] ms. Both tuned and learned selected the same probe timestamp (or abstention) on all 576 queries including repeats of timing runs. No incremental learned benefit demonstrated. Phase CPU cold/tuned in ms: base 10.630/11.051, small 10.739/5.993, medium 10.865/7.207, large 11.024/11.307. Transfer closures 0,102,88,24 respectively, all on changed tasks. Anchor-relative CPU cold/tuned 10.649/8.974 ms, cumulative 10.980/8.805 ms. All arms receive 24 root closures; all 4032 final and 5096 candidate receipts replayed; zero solver failures or block fallback. Independent units are trajectories, not queries or timing repetitions. Three targeted pytest tests passed: 24 no-repeat trajectory fixtures, gate and scope controls, and 224 small exact-oracle policy runs. Full repository tests, Ruff, Lean and Rust not run. No paid inference, GPU, model or native performance claim.

## Reproduction

Install DataGoblin and optional NumPy/SciPy/pytest. From repository root:

    export PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient:training/rlt_gym/sg15_spectral_face:training/rlt_gym/sg17_spectral_chain:training/rlt_gym/sg18_spectral_transfer:training/rlt_gym/sg19_spectral_selective
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q training/rlt_gym/sg19_spectral_selective/test_sg19.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg19_spectral_selective/sg19.py

Add `--resume` to use the saved model and root checkpoint from the same campaign. Do not mix checkpoints from unrelated campaigns. Timing training may select a different threshold on another host. The archive preserves the original frozen coefficients and results, initial/resume source versions, test logs and audit. Executed source blobs: original aa216bb60caa57c75fd23af105cd5b740690f25a, resume a86e4fabe5b8ea8f01ad260bda99bbcaa8afa25a, tests 9eb24eafcfaf701cfb850e1de62e8bb74e6ed813. Raw results SHA256 865f59ad2a18fde43122dd9ddd83daad45ceac9cb870d547ba4dfce6f3e9fdff; model SHA256 392691b61c492818bfa45f9abcd09494ff08dcc33b4538912680cd3406077278. Raw timing runs are supplied separately, not committed.

Next: use sensitivity of the relaxed component maximizers to cheaply update dual bounds and recovery schedules after sparse utility edits, instead of always recomputing all label contributions. Compare incremental exact evaluation and structural-task changes against SG-19 on fresh trajectories.

## Implementation custody

[DataGoblin campaign and tests](https://github.com/MoralityLabAI/DataGoblin/tree/e3391cfbfa5f50d1fae9e19e22834d176eb22f05/training/rlt_gym/sg19_spectral_selective). Complete archive `sg19_cpu_campaign.zip`, SHA256 `ccd848931a3389f670c5c75dd1750adc5c8fefa9232a15176d56a39e872da2ea`. Contains 31 files including the frozen model, executed initial/resume source versions, raw receipts, dependencies, test and interrupted/resumed logs, and SHA256 manifest. Archive integrity passed; both committed Python blobs match executed versions. The separate audit checked 288 globally distinct query signatures, all 4032 final and 5096 candidate receipts, and equality of all 576 tuned/learned probe choices. Raw timing runs supplied separately.
