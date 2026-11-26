# ALG-DESCENT-02 development result

2026-11-26. Companion [source and detailed receipt](https://github.com/MoralityLabAI/DataGoblin/blob/ca78b42e324837ac854ef11cfa7a242ff7d40999/docs/alg_descent_v2.md), DataGoblin PR #89.

40 training roots, 1,817 supervised examples, three model seeds, 36 evaluation roots and 1,332 rollouts. Degrees 24/32/48 are OOD relative to 8..16 training. The experiment uses public synthetic polynomial certificates; no factoring or DLP discovery. Primary outcome is exact full closure under the fixed work and call budgets.

OOD no-cache: MLP 50/54 closed, mean spent work 1546.61; GRU 51/54, 1458.07; public fingerprint heuristic 18/18, 1075.56; exact privileged offline DP 18/18, 912.22. Among 50 jointly solved model/root pairs the GRU has 16 wins, five ties and 29 losses (median GRU-minus-MLP work +5). Learned policy advantage over the strong heuristic fails.

Logical 128-byte LRU closure memo cuts GRU mean work to 1336.20 and fingerprint mean work to 895.06, without changing closure. Utility eviction based on observed work per byte loses to matched LRU for all five online arms. This is consistent with reuse probability being necessary for a useful retention policy. The offline no-cache DP is not a lower bound on cache-enabled costs.

Next hypothesis: exact fingerprint admissibility plus learned residual ranking; train on expected total cost-to-go/regret rather than only the teacher's chosen action, and test history reset/shuffle against normal recurrence on fresh roots. Account for preprocessing, inference, and RAM as separate measured resources. Retention target is expected future reuse times avoided work per retained byte, under a matched LRU control.

No native BPD or MMS implementation, independent inference replay, real RSS measurement or neural win is established by this pilot. The mathematical potential bound is independent of the development failure.
