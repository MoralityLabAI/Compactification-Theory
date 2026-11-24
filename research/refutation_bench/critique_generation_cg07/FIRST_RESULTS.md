# CG-07 first complete results

[Run 45337826780](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45337826780), job 138254000845, head cb803fa362e5c6e15693a3c901a5a4affc05f116. Original [run 45337485838](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45337485838) failed before native certification or training: epsilon_sixth_clearance lacked eleven distinct benign omission boxes within the original coordinate range. The correction increases only that calibration root's first Nat coordinate lower endpoint ceiling from 8 to 24. Train and test roots, seeds, widths, budgets, quotas, model and scoring gate were unchanged.

Native links: 11 original roots, 25 source modules including dependencies. All 1120 reference certificates and 777 distinct accepted actor certificates passed Lean replay. Both 257-parameter rankers trained on 1737 interval nodes, 990 positive nodes. Registered gate **FAIL**: combined 15/31-node test defect counts FIFO 170, DFS 234, slack 235, volume 232, learned-121 238, learned-127 237. Both seeds required at least 239. All policies had identical support totals 583; no invalid certificates. At 15 nodes the learned policies certified 118/120 and 117/120 false advertisements versus 115/120 for interval slack. All three reached 120/120 at 31 nodes. At 127 nodes every policy achieved 479/480, with the remaining case supported but unresolved; full-budget exact interval diagnostic certified every case. The model ranks already-inspected intervals; its scoring work is recorded separately. No CPU-matched improvement or arbitrary Lean comprehension is claimed.

| Split | Budget | Policy | Complete | Supported | Refuted | Nodes | Rank evaluations | Invalid |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cal | 7 | fifo | 86/128 | 72/96 | 14/32 | 508 | 0 | 0 |
| cal | 7 | dfs | 86/128 | 72/96 | 14/32 | 508 | 0 | 0 |
| cal | 7 | slack | 86/128 | 72/96 | 14/32 | 508 | 259 | 0 |
| cal | 7 | volume | 86/128 | 72/96 | 14/32 | 508 | 258 | 0 |
| cal | 7 | learned-121 | 86/128 | 72/96 | 14/32 | 508 | 258 | 0 |
| cal | 7 | learned-127 | 86/128 | 72/96 | 14/32 | 508 | 258 | 0 |
| cal | 15 | fifo | 104/128 | 83/96 | 21/32 | 790 | 0 | 0 |
| cal | 15 | dfs | 115/128 | 83/96 | 32/32 | 722 | 0 | 0 |
| cal | 15 | slack | 111/128 | 83/96 | 28/32 | 742 | 550 | 0 |
| cal | 15 | volume | 111/128 | 83/96 | 28/32 | 742 | 528 | 0 |
| cal | 15 | learned-121 | 115/128 | 83/96 | 32/32 | 714 | 493 | 0 |
| cal | 15 | learned-127 | 115/128 | 83/96 | 32/32 | 714 | 494 | 0 |
| cal | 31 | fifo | 123/128 | 92/96 | 31/32 | 1026 | 0 | 0 |
| cal | 31 | dfs | 124/128 | 92/96 | 32/32 | 860 | 0 | 0 |
| cal | 31 | slack | 124/128 | 92/96 | 32/32 | 892 | 756 | 0 |
| cal | 31 | volume | 123/128 | 92/96 | 31/32 | 924 | 705 | 0 |
| cal | 31 | learned-121 | 124/128 | 92/96 | 32/32 | 852 | 670 | 0 |
| cal | 31 | learned-127 | 124/128 | 92/96 | 32/32 | 852 | 667 | 0 |
| cal | 63 | fifo | 128/128 | 96/96 | 32/32 | 1052 | 0 | 0 |
| cal | 63 | dfs | 128/128 | 96/96 | 32/32 | 880 | 0 | 0 |
| cal | 63 | slack | 128/128 | 96/96 | 32/32 | 912 | 771 | 0 |
| cal | 63 | volume | 128/128 | 96/96 | 32/32 | 950 | 727 | 0 |
| cal | 63 | learned-121 | 128/128 | 96/96 | 32/32 | 872 | 685 | 0 |
| cal | 63 | learned-127 | 128/128 | 96/96 | 32/32 | 872 | 682 | 0 |
| cal | 127 | fifo | 128/128 | 96/96 | 32/32 | 1052 | 0 | 0 |
| cal | 127 | dfs | 128/128 | 96/96 | 32/32 | 880 | 0 | 0 |
| cal | 127 | slack | 128/128 | 96/96 | 32/32 | 912 | 771 | 0 |
| cal | 127 | volume | 128/128 | 96/96 | 32/32 | 950 | 727 | 0 |
| cal | 127 | learned-121 | 128/128 | 96/96 | 32/32 | 872 | 685 | 0 |
| cal | 127 | learned-127 | 128/128 | 96/96 | 32/32 | 872 | 682 | 0 |
| test | 7 | fifo | 230/480 | 197/360 | 33/120 | 2420 | 0 | 0 |
| test | 7 | dfs | 246/480 | 197/360 | 49/120 | 2400 | 0 | 0 |
| test | 7 | slack | 246/480 | 197/360 | 49/120 | 2406 | 1481 | 0 |
| test | 7 | volume | 244/480 | 197/360 | 47/120 | 2406 | 1474 | 0 |
| test | 7 | learned-121 | 250/480 | 197/360 | 53/120 | 2398 | 1470 | 0 |
| test | 7 | learned-127 | 250/480 | 197/360 | 53/120 | 2398 | 1470 | 0 |
| test | 15 | fifo | 331/480 | 259/360 | 72/120 | 4094 | 0 | 0 |
| test | 15 | dfs | 374/480 | 259/360 | 115/120 | 3824 | 0 | 0 |
| test | 15 | slack | 374/480 | 259/360 | 115/120 | 3808 | 3241 | 0 |
| test | 15 | volume | 371/480 | 259/360 | 112/120 | 3854 | 3177 | 0 |
| test | 15 | learned-121 | 377/480 | 259/360 | 118/120 | 3794 | 3321 | 0 |
| test | 15 | learned-127 | 376/480 | 259/360 | 117/120 | 3788 | 3277 | 0 |
| test | 31 | fifo | 422/480 | 324/360 | 98/120 | 5680 | 0 | 0 |
| test | 31 | dfs | 443/480 | 324/360 | 119/120 | 4912 | 0 | 0 |
| test | 31 | slack | 444/480 | 324/360 | 120/120 | 4896 | 4554 | 0 |
| test | 31 | volume | 444/480 | 324/360 | 120/120 | 4954 | 4280 | 0 |
| test | 31 | learned-121 | 444/480 | 324/360 | 120/120 | 4864 | 4642 | 0 |
| test | 31 | learned-127 | 444/480 | 324/360 | 120/120 | 4862 | 4570 | 0 |
| test | 63 | fifo | 468/480 | 350/360 | 118/120 | 6646 | 0 | 0 |
| test | 63 | dfs | 470/480 | 350/360 | 120/120 | 5578 | 0 | 0 |
| test | 63 | slack | 470/480 | 350/360 | 120/120 | 5558 | 5333 | 0 |
| test | 63 | volume | 470/480 | 350/360 | 120/120 | 5616 | 4979 | 0 |
| test | 63 | learned-121 | 470/480 | 350/360 | 120/120 | 5526 | 5473 | 0 |
| test | 63 | learned-127 | 470/480 | 350/360 | 120/120 | 5524 | 5368 | 0 |
| test | 127 | fifo | 479/480 | 359/360 | 120/120 | 6846 | 0 | 0 |
| test | 127 | dfs | 479/480 | 359/360 | 120/120 | 5758 | 0 | 0 |
| test | 127 | slack | 479/480 | 359/360 | 120/120 | 5738 | 5524 | 0 |
| test | 127 | volume | 479/480 | 359/360 | 120/120 | 5796 | 5169 | 0 |
| test | 127 | learned-121 | 479/480 | 359/360 | 120/120 | 5706 | 5697 | 0 |
| test | 127 | learned-127 | 479/480 | 359/360 | 120/120 | 5704 | 5586 | 0 |

- Seed 121: 257 parameters, final loss 0.43750759923190446, checkpoint SHA256 8c80f8eb511dfcae011db7a8724e2ed92b9d5a8e8e0a4382b759f817d7c07315.
- Seed 127: 257 parameters, final loss 0.4059293791008325, checkpoint SHA256 7185fafe7fb845449f1a6e92da248f65ea44614ae7790f5ce9c512e2cfda4459.

[Full artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45337826780/artifacts/14893859599), ZIP SHA256 c9e08b23bf41d9fc4103eadf8133c99f4e5ddd674d7edf7d29da588986aea260. The arithmetic interval task is close to saturation on defects. Next track returns to explicit mathematical bridges and the distinction between truth of the advertised conclusion and derivability through a selected formal source interface.
