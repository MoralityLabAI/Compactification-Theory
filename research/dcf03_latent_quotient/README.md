# DCF-03: Observation-only latent quotient discovery

Date: 2026-11-25. Exploratory CPU study following DCF-02. No GPU/energy or language model measurements.

## Task and information boundary
Hidden state s in Z/8, initial s=0, op o=0..31, transition f(s,o)=((2 floor(o/8)+1)s+o mod 8) mod 8; observe only y=1[s=0] after each operation. The training function returns operations and binary observations, never semantic state labels. Held-out operation tokens are {10,13,15,18,21,23,26,29,31}; 23 remain. Every primitive multiplication/addition is in training. Composed models receive the decomposition and identity skip, and the latent model receives known state cardinality 8, accepting/initial state anchor, and approximate permutation prior (Sinkhorn normalization plus entropy penalty). These are substantial task priors. No unseen primitive or unknown-state-cardinality claim.

## DCF-03A: passive neural experiment
Seeds 131,137,139; 600 AdamW updates, batch 96, length 4, 230400 binary labels per fit; separate evaluation RNG, 1024 random cases/seed/length/alphabet. Fresh random draws can repeat training words. Dense32, rank4 use shared continuous core, monolithic32 does not compose operations, latent8 uses learned stochastic matrices. Optimizer lr .003 for RNNs, .015 for latent, positive BCE weight 3, latent entropy penalty .02. Different model and training priors preclude isolating factorization as the cause. No prospective blinded registration.

| arm | params | seen L4 trace % | L16 | L64 | held-only L4 trace % | L16 | L64 | exact full-alphabet table |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|dense32|1857|83.56|6.93|0|45.74|2.67|0|0/3|
|rank4|1089|78.06|11.00|0|48.47|6.28|0|0/3|
|monolithic32|3137|75.26|4.88|0|18.20|0.03|0|0/3|
|latent8|768|94.50|51.60|2.21|80.05|39.91|0.03|1/3|

Complete trace accuracy requires all prefix outputs correct. Final accuracy can be misleading due to imbalance; raw receipts also include positive/negative counts. For example L64 held-only balanced per-token accuracy for latent8 is 55.08%. For continuous models, extract representative states using eight predicted future observation signatures (empty plus seven additive probes). BFS on candidate signatures stops at 64 states. Dense32 and monolithic32 all overflow; rank4 extracts 22,18,39 states and fails exact checks (short counterexamples [1,26], [1,13], [1,13]). This extraction method's failure is not a proof of no quotient.

For latent8, harden primitive transitions by row argmax, compose them into integer table, certify with BFS over reachable oracle/candidate state pairs. The oracle is used in this audit, not table extraction. Seed131 passes full alphabet, seeds137,139 fail with witnesses [1,13] and [1,1]. Extra training on fresh seeds149,151,157, 2400 updates and 921600 labels each, yields one additional exact table (157); 149,151 fail. Across both training schedules 2/6 latent fits pass. Across all 15 fits, 2 pass. 600-step and 2400-step groups use different seeds; no causal training-budget comparison.

## DCF-03B: finite proof and hard compilation
The product BFS starts at (0,0). If every reachable pair has the same acceptance bit, induction on the input word proves equality of observations for every finite word. If a mismatch is reachable, BFS returns a shortest distinguishing word. Independent integer audit derives the bijection from candidate transitions on single additive operations. It verifies all 256 transition edges and acceptance bits with an independent bitwise oracle. Seed131 maps oracle states to [0,2,4,6,7,5,1,3]; seed157 maps to [0,4,3,7,2,1,5,6]. Each certificate visits 8 product states.

At length256, for each passing seed and each of mixed/held-only alphabets, 256 new random words: hardened table 256/256 complete traces in each cell (1024/1024 total); continuous probability propagation 0/1024. At length64, soft trace counts seed131: mixed 2/256, held-only 0/256; seed157: both 256/256. These are selected passing checkpoints, not unconditional training performance. Hard-table theorem is independent of floating-point execution; soft network not certified.

Minimality: all 8 reference states are reachable by single additive operation i, and distinct i,j are distinguished by suffix add(-i mod8). The 28 pair witnesses are checked in active_audit.py. Myhill-Nerode gives at least 8 states for any exact deterministic recognizer; three bits of persistent state suffice and are necessary in this discrete representation. Generic complete transition table = 8*32*3=768 bits =96 bytes (accepting/initial state convention fixed). The known composition can store ten nonidentity primitive tables in 30 bytes; neither figure is a lower bound for program storage. Direct modular arithmetic is a simpler exact oracle.

## DCF-03C: active symbolic control
Given chosen membership queries to the binary observation oracle and the fixed additive probes, construct signatures of histories and enumerate primitive transition edges via BFS. No hidden-state labels supplied. 656 distinct queries (1838 cumulative operation tokens, longest word length3), 8 states, exact product audit and independent bijection. Compose learned primitive tables to cover the held-out composite ops without querying those tokens. Unlike passive learning this control gets chosen suffix probes and knows their proposed structure. The comparison is useful for this task but not an equal-information or equal-compute benchmark.

## Reproduction
From directory:
```sh
python gym.py --out run --steps 600
python gym.py --out long_run --seed 149 151 157 --arms latent8 --steps 2400
python active_audit.py --root .
```
PyTorch 2.10.0+cpu, NumPy 2.3.5, Python 3.13, single-thread CPU. Aggregate training loop times 14.32s +17.62s, excluding audit and evaluation. Full checkpoints, raw receipts and SHA-256 manifest are in the conversation's dcf03_reproducibility.zip. Source and aggregates in repo. No paid compute.

## DCF-04 proposed
Hide cardinality (candidate k=4,8,12,16), remove supplied primitive factorization in a separate arm, and use permutation systems with unknown transition laws and randomized observation mappings. Discover suffixes from counterexamples instead of providing all eight distinguishing probes. Compare neural proposal + active repair, passive model, query-only learner and exact symbolic oracle. Charge queries, trace labels, wall time, state bits and code storage separately. Freeze new seed/system families and hold out verification systems, operation combinations and longer words; use a separate audit oracle and report rejected hypotheses. Primary outcome: exact equivalence on independently generated systems within registered query and compute budgets, with failure retained. No extrapolation to general AGI or dense matrix exponent implied.
