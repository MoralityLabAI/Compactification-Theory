# DCF-14B: Budgeted decision inference

Exploratory CPU pilot. 180 generated problems, 720 arm/problem cells; n=12,16,24, families grid/sparse/dense/cycle/bipartite, seeds1501/1511/1523, four repetitions with fresh generated observations. Independent fair latent bits, known noisy unary and pair parity likelihoods, one queried bit. Reward correct +1, wrong -4, abstain 0; act at posterior >=.8 or <=.2. Full reference min-fill scope cap16 completes all fixtures. Smaller cap5 exact VE abstains on overflow; cutset greedily removes highest-degree nonquery variables until scope fits, maximum four cut variables/16 assignments; sums unnormalized masses across assignments. BP uses 20 synchronous iterations, damping .5. All timings single CPU measurements including ordering and inference, excluding data generation; no matched optimized backend or equal-operation budget. Local source hashes before execution, not external registration.

|arm|completed posteriors|expected utility|regret|mean ms|realized utility|acted|
|---|---:|---:|---:|---:|---:|---:|
|VE5|129/180|.413658|.225757|.558|.444444|100|
|cutset|167/180|.567296|.072119|1.973|.622222|132|
|BP20|180/180|.632674|.006741|10.303|.655556|143|
|VE16|180/180|.639415|0|.929|.694444|145|

60 independent brute-force checks at n12 max posterior error 8.88e-16. All completed bounded/cutset exact predictions agree with reference within 1e-11. Cutset recovers 38 additional posteriors, including all 17 previously uncompleted sparse cases. Dense regret .84674/.36060/.02928 for VE5/cutset/BP; grid BP regret .00442. Other families zero regret for all except sparse VE5 (.28205). Cap5 constrains scope, not actual working memory; cutset enumerates assignments sequentially. BP approximate posterior can be wrong; no safe confidence guarantee. No learned selector, neural training, hardware or energy measurements. Source and raw receipts in conversation archive dcf14b_reproducibility.zip; report published here. Next: compare a validation-trained selector to a simple scope-based rule, with optimized baselines and sealed graph families. No DCF-15/16 integration yet.
