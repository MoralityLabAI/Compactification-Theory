# DCF-08: Correlation-priced, per-trajectory partition routing

2026-11-25. CPU exploratory follow-up to DCF-07. Six fresh systems (width 6/8; seeds 503,509,521), 24 calibrated context graphs, 36 system-condition pairs, 198 evaluated arm-condition cells. No accelerator or energy measurement.

## Mathematical result

For an old product distribution p over partition P, projection to a new partition Q is q=product of p's marginals on Q. D_KL(p||q)=sum of entropies of new factors minus sum of entropies of old factors. Since old factors are independent, each new factor is itself the product of marginals on its intersections with old clusters. Thus the projection cost can be evaluated without constructing the full joint. This prices information lost from the *current approximation*, not information already absent from that approximation or future prediction error. Independent direct enumeration: 64 randomized partition fixtures (8 probability draws each), maximum KL error 4.57967e-15 and marginal discrepancy 6.66134e-16. The oracle and causality checks also passed in all six systems (oracle max discrepancy 4.441e-16, causal prefix max 0).

## Protocol

Reuse DCF-07 graph/rate calibration on fresh generated systems, recovering all 24 graphs. Structural priors and reset probes still supplied. Four contexts (independent, pairs, XOR triples, connected XOR cycle). Each trajectory has its own random context sequence and independent observations. Graph/rate calibration uses DCF-07's probe counts; no neural fits. Candidate partitions derive from each learned graph and union partition, cap 1..4. Static per-context held-out calibration: 128 length-48 words; estimate observation NLL for each partition. Freeze selection cost lambda=.0004 per belief degree of freedom, switch fee .02, projection weight 1. Each trajectory's router chooses among retain, context-specific optimum, and fixed union choice, minimizing NLL+lambda*degrees+(switch fee+mu*projection KL)/H for switches. H=clip((16+t)/(2+number of decoded context changes),1,32); causal cumulative hazard estimate, not a reliable residual-life estimate under arbitrary drift. Dwell-only sets mu=0; immediate routing selects static per-context optimum, fixed uses union choice selected by average static calibration. Full learned uses full joint with learned graphs/rates and same decoded context; oracle has true graph, rates, and labels. No use of oracle in routing. For unreliable labels, simple causal MAP context filter assumes .125 switch probability and .15 label corruption; raw comparator uses raw labels/last label carried forward on missing. No integration over context uncertainty. Test 32 trajectories x96 steps per system-condition, independent RNG: rapid switch hazard .75, slow .04, alternating 16-step blocks of .04/.65, noisy15/noisy30 and missing50 (underlying hazard .125). 110,592 underlying observation predictions before multiplying by arms. Equal-weight system-condition summaries; repeated predictions correlated. Pre-run local hashes, not external preregistration. Model conditions measured on same trajectories.

## Results

Mean predictive conditional KL vs exact privileged oracle (nats per step):

|scenario|full learned|fixed|instant|dwell|correlation|raw correlation|
|---|---:|---:|---:|---:|---:|---:|
|rapid|.000433|.010602|.009769|.008690|.008340|NA|
|slow|.000479|.010058|.004344|.004420|.004190|NA|
|bursty|.000420|.010275|.006735|.006143|.005264|NA|
|noisy15|.027375|.032294|.029561|.029753|.030272|.030306|
|noisy30|.044188|.045636|.043866|.044384|.044466|.053010|
|missing50|.025928|.030138|.028056|.028404|.028669|.021918|

All 36 condition means across each compact arm:

|arm|KL|NLL|composite objective|mean coordinates|switches/word|discarded KL/word|
|---|---:|---:|---:|---:|---:|---:|
|fixed|.023167|.516025|.521825|14.500|0|0|
|instant|.020389|.513298|.524344|13.402|27.292|.874483|
|dwell|.020299|.513185|.521277|13.869|12.211|.487455|
|correlation|.020200|.513424|.520869|14.220|8.432|.065741|

Correlation penalty reduces discarded KL 86.5% vs dwell and switches 69.1% vs instant. Mean K8 coordinates 15.922 vs 255 full; K6 12.519 vs 63 full. These are degrees of freedom, not measured bytes or arithmetic. On clean contexts mean KL .005931 vs .006418 dwell and .006950 instant. Noisy contexts worsen predictive KL relative to dwell. On missing labels last-label carry-forward outperforms assumed HMM (context accuracy .8878 vs .8590); at 30% corruption HMM helps (.7578 vs .6997). Do not treat the simple MAP decoder as Bayes optimal for observations. Strict joint gate requires correlation composite objective <= dwell, KL <= instant, and degrees <= instant: **FAIL, 1/36 (8,521,rapid)**. Composite-only gate 21/36. KL and state-budget gate 1/36. Additional retained state is a confound; cannot claim a matched-budget gain. Every system-condition has lower/equal discarded KL vs dwell (36/36). The correlation arm is a valid mechanism result but not a validated efficiency claim. No GPU/TPU advantage demonstrated. Timing receipts are single runs with concurrency and contention, include routing loops but exclude initialization/calibration; no benchmark claim. Candidate projection evaluations averaged 7760.6 per condition (32 words); explicit computation cost must be accounted for. No claim of production LDT routing.

## Reproduction

Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu. Dependencies copied byte-for-byte from DCF-07. From directory: sha256sum -c PRE_RUN_SHA256; bash run_all.sh; python audit.py. Receipts and logs in downloadable archive; no paid compute. Source code and protocol hashes in repo. DCF-09 candidate: uncertain-context belief (mixture or particles), local/change-point hazard estimates, budget-constrained Pareto comparison and amortized projection cost; compare to simple last-label baseline and full filter with same information access. Freeze new seeds and thresholds before execution. 
