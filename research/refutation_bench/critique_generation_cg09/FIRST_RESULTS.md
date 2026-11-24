# CG-09 first complete results

[Run 45339224815](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45339224815), job 138258557718, source head 6be746fd587071d10728f20586a81bd0419aec60. Original RHMainLogic source SHA256 ba2645a310f81497dfd35aa0e9e19c31b6e1cf5d261eb489e66415743d5153a2. Native weakening lemmas: four. Native reference replay: 512 certificates, 640 declarations. Native actor replay: 256 distinct certificates, 320 declarations. Both 737-parameter rankers trained on 256 trajectories, 2412 action sets and 13947 candidate rows. Zero invalid certificates.

Primary architectural gate **FAIL**. Paired full-budget goal-slice costs: depth 2474 isolated vs 2423 shared (2.0614% saving), bridge 1275 vs 1124 (11.8431% saving). Each split has 24 improved, 72 tied, zero regressed cases. Secondary learned gate **PASS**: both seeds 326 vs strongest fixed control 304, above the registered threshold of 310 on combined 16/32-firing depth/bridge test cases. This is improvement within the registered fixed control set, not over all possible symbolic schedules. On deep cases at 32 firings both learned actors certify all 96 cases with 1845 paid firings in total; strongest full-budget fixed shared-relevance control uses 2285. Full-budget learned and fixed policies all reach full coverage.

Gate receipt:
```json
{
  "architecture": {
    "test_depth": {
      "isolated": 2474,
      "shared": 2423,
      "saved": 51,
      "relative_savings": 0.020614389652384824,
      "improved": 24,
      "ties": 72,
      "regressions": 0,
      "passed": false
    },
    "test_bridge": {
      "isolated": 1275,
      "shared": 1124,
      "saved": 151,
      "relative_savings": 0.11843137254901959,
      "improved": 24,
      "ties": 72,
      "regressions": 0,
      "passed": true
    }
  },
  "architecture_passed": false,
  "totals": {
    "isolated:global_first": 182,
    "isolated:goal_slice": 265,
    "isolated:shared_relevance": 265,
    "shared:global_first": 304,
    "shared:goal_slice": 282,
    "shared:shared_relevance": 286,
    "dominant:global_first": 304,
    "dominant:goal_slice": 285,
    "dominant:shared_relevance": 286,
    "learned-141": 326,
    "learned-149": 326
  },
  "best_symbolic": 304,
  "learned_passed": true,
  "invalid": 0
}
```

| Split | Budget | Policy | Complete | Source-supported | Independent + gap | Refuted | Paid firings | Transports | Rule scans | Rank evals |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cal | 8 | isolated:global_first | 19/64 | 0/32 | 8/16 | 11/16 | 490 | 0 | 36468 | 0 |
| cal | 8 | isolated:goal_slice | 2/64 | 2/32 | 0/16 | 0/16 | 512 | 0 | 24458 | 5536 |
| cal | 8 | isolated:shared_relevance | 2/64 | 2/32 | 0/16 | 0/16 | 512 | 0 | 24458 | 5536 |
| cal | 8 | shared:global_first | 19/64 | 0/32 | 8/16 | 11/16 | 490 | 1319 | 34381 | 0 |
| cal | 8 | shared:goal_slice | 2/64 | 2/32 | 0/16 | 0/16 | 512 | 656 | 23865 | 5102 |
| cal | 8 | shared:shared_relevance | 2/64 | 2/32 | 0/16 | 0/16 | 512 | 939 | 26708 | 6088 |
| cal | 8 | dominant:global_first | 19/64 | 0/32 | 8/16 | 11/16 | 490 | 1319 | 34381 | 0 |
| cal | 8 | dominant:goal_slice | 2/64 | 2/32 | 0/16 | 0/16 | 512 | 719 | 24525 | 2640 |
| cal | 8 | dominant:shared_relevance | 2/64 | 2/32 | 0/16 | 0/16 | 512 | 939 | 26708 | 2894 |
| cal | 8 | learned-141 | 20/64 | 2/32 | 8/16 | 10/16 | 490 | 1231 | 31758 | 3099 |
| cal | 8 | learned-149 | 20/64 | 2/32 | 8/16 | 10/16 | 490 | 1226 | 31760 | 3100 |
| cal | 16 | isolated:global_first | 32/64 | 0/32 | 16/16 | 16/16 | 773 | 0 | 54088 | 0 |
| cal | 16 | isolated:goal_slice | 62/64 | 32/32 | 15/16 | 15/16 | 765 | 0 | 29600 | 6650 |
| cal | 16 | isolated:shared_relevance | 62/64 | 32/32 | 15/16 | 15/16 | 765 | 0 | 29600 | 6650 |
| cal | 16 | shared:global_first | 58/64 | 26/32 | 16/16 | 16/16 | 734 | 1976 | 44244 | 0 |
| cal | 16 | shared:goal_slice | 63/64 | 32/32 | 16/16 | 15/16 | 748 | 1050 | 28599 | 6069 |
| cal | 16 | shared:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 7216 |
| cal | 16 | dominant:global_first | 58/64 | 26/32 | 16/16 | 16/16 | 734 | 1976 | 44244 | 0 |
| cal | 16 | dominant:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 713 | 1165 | 28932 | 3231 |
| cal | 16 | dominant:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 3602 |
| cal | 16 | learned-141 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1533 | 34532 | 3592 |
| cal | 16 | learned-149 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1527 | 34547 | 3593 |
| cal | 32 | isolated:global_first | 46/64 | 14/32 | 16/16 | 16/16 | 1232 | 0 | 70085 | 0 |
| cal | 32 | isolated:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 768 | 0 | 29656 | 6654 |
| cal | 32 | isolated:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 768 | 0 | 29656 | 6654 |
| cal | 32 | shared:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 744 | 2004 | 44413 | 0 |
| cal | 32 | shared:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 749 | 1053 | 28617 | 6070 |
| cal | 32 | shared:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 7216 |
| cal | 32 | dominant:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 744 | 2004 | 44413 | 0 |
| cal | 32 | dominant:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 713 | 1165 | 28932 | 3231 |
| cal | 32 | dominant:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 3602 |
| cal | 32 | learned-141 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1533 | 34532 | 3592 |
| cal | 32 | learned-149 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1527 | 34547 | 3593 |
| cal | 64 | isolated:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 1361 | 0 | 72322 | 0 |
| cal | 64 | isolated:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 768 | 0 | 29656 | 6654 |
| cal | 64 | isolated:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 768 | 0 | 29656 | 6654 |
| cal | 64 | shared:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 744 | 2004 | 44413 | 0 |
| cal | 64 | shared:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 749 | 1053 | 28617 | 6070 |
| cal | 64 | shared:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 7216 |
| cal | 64 | dominant:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 744 | 2004 | 44413 | 0 |
| cal | 64 | dominant:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 713 | 1165 | 28932 | 3231 |
| cal | 64 | dominant:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 3602 |
| cal | 64 | learned-141 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1533 | 34532 | 3592 |
| cal | 64 | learned-149 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1527 | 34547 | 3593 |
| cal | 128 | isolated:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 1361 | 0 | 72322 | 0 |
| cal | 128 | isolated:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 768 | 0 | 29656 | 6654 |
| cal | 128 | isolated:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 768 | 0 | 29656 | 6654 |
| cal | 128 | shared:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 744 | 2004 | 44413 | 0 |
| cal | 128 | shared:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 749 | 1053 | 28617 | 6070 |
| cal | 128 | shared:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 7216 |
| cal | 128 | dominant:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 744 | 2004 | 44413 | 0 |
| cal | 128 | dominant:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 713 | 1165 | 28932 | 3231 |
| cal | 128 | dominant:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 3602 |
| cal | 128 | learned-141 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1533 | 34532 | 3592 |
| cal | 128 | learned-149 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1527 | 34547 | 3593 |
| cal | 256 | isolated:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 1361 | 0 | 72322 | 0 |
| cal | 256 | isolated:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 768 | 0 | 29656 | 6654 |
| cal | 256 | isolated:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 768 | 0 | 29656 | 6654 |
| cal | 256 | shared:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 744 | 2004 | 44413 | 0 |
| cal | 256 | shared:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 749 | 1053 | 28617 | 6070 |
| cal | 256 | shared:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 7216 |
| cal | 256 | dominant:global_first | 64/64 | 32/32 | 16/16 | 16/16 | 744 | 2004 | 44413 | 0 |
| cal | 256 | dominant:goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 713 | 1165 | 28932 | 3231 |
| cal | 256 | dominant:shared_relevance | 64/64 | 32/32 | 16/16 | 16/16 | 714 | 1422 | 30755 | 3602 |
| cal | 256 | learned-141 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1533 | 34532 | 3592 |
| cal | 256 | learned-149 | 64/64 | 32/32 | 16/16 | 16/16 | 596 | 1527 | 34547 | 3593 |
| test_depth | 8 | isolated:global_first | 3/96 | 0/48 | 1/24 | 2/24 | 761 | 0 | 109220 | 0 |
| test_depth | 8 | isolated:goal_slice | 2/96 | 0/48 | 1/24 | 1/24 | 762 | 0 | 99797 | 11530 |
| test_depth | 8 | isolated:shared_relevance | 2/96 | 0/48 | 1/24 | 1/24 | 762 | 0 | 99797 | 11530 |
| test_depth | 8 | shared:global_first | 3/96 | 0/48 | 1/24 | 2/24 | 761 | 2189 | 107164 | 0 |
| test_depth | 8 | shared:goal_slice | 2/96 | 0/48 | 1/24 | 1/24 | 761 | 1248 | 97646 | 11054 |
| test_depth | 8 | shared:shared_relevance | 2/96 | 0/48 | 1/24 | 1/24 | 761 | 1988 | 102504 | 11800 |
| test_depth | 8 | dominant:global_first | 3/96 | 0/48 | 1/24 | 2/24 | 761 | 2189 | 107164 | 0 |
| test_depth | 8 | dominant:goal_slice | 2/96 | 0/48 | 1/24 | 1/24 | 761 | 1274 | 98020 | 4504 |
| test_depth | 8 | dominant:shared_relevance | 2/96 | 0/48 | 1/24 | 1/24 | 761 | 1988 | 102504 | 4639 |
| test_depth | 8 | learned-141 | 3/96 | 0/48 | 1/24 | 2/24 | 761 | 2108 | 104234 | 4678 |
| test_depth | 8 | learned-149 | 3/96 | 0/48 | 1/24 | 2/24 | 761 | 2107 | 104145 | 4698 |
| test_depth | 16 | isolated:global_first | 41/96 | 0/48 | 19/24 | 22/24 | 1387 | 0 | 189192 | 0 |
| test_depth | 16 | isolated:goal_slice | 2/96 | 0/48 | 1/24 | 1/24 | 1514 | 0 | 161773 | 19663 |
| test_depth | 16 | isolated:shared_relevance | 2/96 | 0/48 | 1/24 | 1/24 | 1514 | 0 | 161773 | 19663 |
| test_depth | 16 | shared:global_first | 41/96 | 0/48 | 19/24 | 22/24 | 1387 | 3905 | 179264 | 0 |
| test_depth | 16 | shared:goal_slice | 2/96 | 0/48 | 1/24 | 1/24 | 1513 | 2099 | 158050 | 18457 |
| test_depth | 16 | shared:shared_relevance | 2/96 | 0/48 | 1/24 | 1/24 | 1513 | 3353 | 170651 | 21139 |
| test_depth | 16 | dominant:global_first | 41/96 | 0/48 | 19/24 | 22/24 | 1387 | 3905 | 179264 | 0 |
| test_depth | 16 | dominant:goal_slice | 2/96 | 0/48 | 1/24 | 1/24 | 1513 | 2170 | 161020 | 8150 |
| test_depth | 16 | dominant:shared_relevance | 2/96 | 0/48 | 1/24 | 1/24 | 1513 | 3353 | 170651 | 8816 |
| test_depth | 16 | learned-141 | 41/96 | 0/48 | 19/24 | 22/24 | 1387 | 3875 | 170349 | 8083 |
| test_depth | 16 | learned-149 | 41/96 | 0/48 | 19/24 | 22/24 | 1387 | 3875 | 169294 | 8082 |
| test_depth | 32 | isolated:global_first | 48/96 | 0/48 | 24/24 | 24/24 | 2179 | 0 | 286702 | 0 |
| test_depth | 32 | isolated:goal_slice | 94/96 | 48/48 | 23/24 | 23/24 | 2460 | 0 | 209368 | 25501 |
| test_depth | 32 | isolated:shared_relevance | 94/96 | 48/48 | 23/24 | 23/24 | 2460 | 0 | 209368 | 25501 |
| test_depth | 32 | shared:global_first | 85/96 | 37/48 | 24/24 | 24/24 | 2096 | 5944 | 236994 | 0 |
| test_depth | 32 | shared:goal_slice | 94/96 | 48/48 | 23/24 | 23/24 | 2412 | 3184 | 202787 | 23773 |
| test_depth | 32 | shared:shared_relevance | 95/96 | 48/48 | 24/24 | 23/24 | 2284 | 5254 | 213214 | 28076 |
| test_depth | 32 | dominant:global_first | 85/96 | 37/48 | 24/24 | 24/24 | 2096 | 5944 | 236994 | 0 |
| test_depth | 32 | dominant:goal_slice | 95/96 | 48/48 | 24/24 | 23/24 | 2300 | 3672 | 204048 | 11005 |
| test_depth | 32 | dominant:shared_relevance | 95/96 | 48/48 | 24/24 | 23/24 | 2284 | 5254 | 213214 | 12334 |
| test_depth | 32 | learned-141 | 96/96 | 48/48 | 24/24 | 24/24 | 1845 | 5191 | 201545 | 10625 |
| test_depth | 32 | learned-149 | 96/96 | 48/48 | 24/24 | 24/24 | 1845 | 5191 | 200490 | 10624 |
| test_depth | 64 | isolated:global_first | 48/96 | 0/48 | 24/24 | 24/24 | 3715 | 0 | 395037 | 0 |
| test_depth | 64 | isolated:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2474 | 0 | 209879 | 25540 |
| test_depth | 64 | isolated:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2474 | 0 | 209879 | 25540 |
| test_depth | 64 | shared:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2119 | 6013 | 237846 | 0 |
| test_depth | 64 | shared:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2423 | 3213 | 203187 | 23802 |
| test_depth | 64 | shared:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2285 | 5257 | 213250 | 28077 |
| test_depth | 64 | dominant:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2119 | 6013 | 237846 | 0 |
| test_depth | 64 | dominant:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2301 | 3675 | 204084 | 11006 |
| test_depth | 64 | dominant:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2285 | 5257 | 213250 | 12335 |
| test_depth | 64 | learned-141 | 96/96 | 48/48 | 24/24 | 24/24 | 1845 | 5191 | 201545 | 10625 |
| test_depth | 64 | learned-149 | 96/96 | 48/48 | 24/24 | 24/24 | 1845 | 5191 | 200490 | 10624 |
| test_depth | 128 | isolated:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 4575 | 0 | 424368 | 0 |
| test_depth | 128 | isolated:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2474 | 0 | 209879 | 25540 |
| test_depth | 128 | isolated:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2474 | 0 | 209879 | 25540 |
| test_depth | 128 | shared:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2119 | 6013 | 237846 | 0 |
| test_depth | 128 | shared:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2423 | 3213 | 203187 | 23802 |
| test_depth | 128 | shared:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2285 | 5257 | 213250 | 28077 |
| test_depth | 128 | dominant:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2119 | 6013 | 237846 | 0 |
| test_depth | 128 | dominant:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2301 | 3675 | 204084 | 11006 |
| test_depth | 128 | dominant:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2285 | 5257 | 213250 | 12335 |
| test_depth | 128 | learned-141 | 96/96 | 48/48 | 24/24 | 24/24 | 1845 | 5191 | 201545 | 10625 |
| test_depth | 128 | learned-149 | 96/96 | 48/48 | 24/24 | 24/24 | 1845 | 5191 | 200490 | 10624 |
| test_depth | 256 | isolated:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 4575 | 0 | 424368 | 0 |
| test_depth | 256 | isolated:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2474 | 0 | 209879 | 25540 |
| test_depth | 256 | isolated:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2474 | 0 | 209879 | 25540 |
| test_depth | 256 | shared:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2119 | 6013 | 237846 | 0 |
| test_depth | 256 | shared:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2423 | 3213 | 203187 | 23802 |
| test_depth | 256 | shared:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2285 | 5257 | 213250 | 28077 |
| test_depth | 256 | dominant:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2119 | 6013 | 237846 | 0 |
| test_depth | 256 | dominant:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2301 | 3675 | 204084 | 11006 |
| test_depth | 256 | dominant:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 2285 | 5257 | 213250 | 12335 |
| test_depth | 256 | learned-141 | 96/96 | 48/48 | 24/24 | 24/24 | 1845 | 5191 | 201545 | 10625 |
| test_depth | 256 | learned-149 | 96/96 | 48/48 | 24/24 | 24/24 | 1845 | 5191 | 200490 | 10624 |
| test_bridge | 8 | isolated:global_first | 9/96 | 0/48 | 0/24 | 9/24 | 761 | 0 | 58429 | 0 |
| test_bridge | 8 | isolated:goal_slice | 2/96 | 2/48 | 0/24 | 0/24 | 768 | 0 | 44028 | 9144 |
| test_bridge | 8 | isolated:shared_relevance | 2/96 | 2/48 | 0/24 | 0/24 | 768 | 0 | 44028 | 9144 |
| test_bridge | 8 | shared:global_first | 12/96 | 0/48 | 3/24 | 9/24 | 759 | 1967 | 55430 | 0 |
| test_bridge | 8 | shared:goal_slice | 2/96 | 2/48 | 0/24 | 0/24 | 768 | 1083 | 42895 | 8407 |
| test_bridge | 8 | shared:shared_relevance | 2/96 | 2/48 | 0/24 | 0/24 | 768 | 1424 | 44653 | 9140 |
| test_bridge | 8 | dominant:global_first | 12/96 | 0/48 | 3/24 | 9/24 | 759 | 1967 | 55430 | 0 |
| test_bridge | 8 | dominant:goal_slice | 2/96 | 2/48 | 0/24 | 0/24 | 768 | 1077 | 43559 | 3966 |
| test_bridge | 8 | dominant:shared_relevance | 2/96 | 2/48 | 0/24 | 0/24 | 768 | 1424 | 44653 | 4193 |
| test_bridge | 8 | learned-141 | 4/96 | 2/48 | 0/24 | 2/24 | 765 | 1634 | 50464 | 4583 |
| test_bridge | 8 | learned-149 | 6/96 | 2/48 | 1/24 | 3/24 | 765 | 1638 | 50404 | 4578 |
| test_bridge | 16 | isolated:global_first | 27/96 | 0/48 | 3/24 | 24/24 | 1369 | 0 | 95137 | 0 |
| test_bridge | 16 | isolated:goal_slice | 73/96 | 48/48 | 6/24 | 19/24 | 1191 | 0 | 55559 | 11623 |
| test_bridge | 16 | isolated:shared_relevance | 73/96 | 48/48 | 6/24 | 19/24 | 1191 | 0 | 55559 | 11623 |
| test_bridge | 16 | shared:global_first | 82/96 | 34/48 | 24/24 | 24/24 | 1180 | 3031 | 73706 | 0 |
| test_bridge | 16 | shared:goal_slice | 90/96 | 48/48 | 23/24 | 19/24 | 1115 | 1497 | 52710 | 10155 |
| test_bridge | 16 | shared:shared_relevance | 93/96 | 48/48 | 23/24 | 22/24 | 1112 | 2020 | 54186 | 11146 |
| test_bridge | 16 | dominant:global_first | 82/96 | 34/48 | 24/24 | 24/24 | 1180 | 3031 | 73706 | 0 |
| test_bridge | 16 | dominant:goal_slice | 92/96 | 48/48 | 23/24 | 21/24 | 1111 | 1665 | 53334 | 4955 |
| test_bridge | 16 | dominant:shared_relevance | 93/96 | 48/48 | 23/24 | 22/24 | 1112 | 2020 | 54186 | 5343 |
| test_bridge | 16 | learned-141 | 93/96 | 48/48 | 23/24 | 22/24 | 1077 | 2254 | 58820 | 5629 |
| test_bridge | 16 | learned-149 | 93/96 | 48/48 | 23/24 | 22/24 | 1064 | 2248 | 58449 | 5590 |
| test_bridge | 32 | isolated:global_first | 66/96 | 19/48 | 23/24 | 24/24 | 2120 | 0 | 123327 | 0 |
| test_bridge | 32 | isolated:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1275 | 0 | 57277 | 11832 |
| test_bridge | 32 | isolated:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1275 | 0 | 57277 | 11832 |
| test_bridge | 32 | shared:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 1210 | 3117 | 74435 | 0 |
| test_bridge | 32 | shared:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1124 | 1518 | 52998 | 10173 |
| test_bridge | 32 | shared:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1116 | 2028 | 54308 | 11152 |
| test_bridge | 32 | dominant:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 1210 | 3117 | 74435 | 0 |
| test_bridge | 32 | dominant:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1117 | 1677 | 53524 | 4962 |
| test_bridge | 32 | dominant:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1116 | 2028 | 54308 | 5347 |
| test_bridge | 32 | learned-141 | 96/96 | 48/48 | 24/24 | 24/24 | 1080 | 2261 | 58921 | 5632 |
| test_bridge | 32 | learned-149 | 96/96 | 48/48 | 24/24 | 24/24 | 1067 | 2255 | 58550 | 5593 |
| test_bridge | 64 | isolated:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2410 | 0 | 129922 | 0 |
| test_bridge | 64 | isolated:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1275 | 0 | 57277 | 11832 |
| test_bridge | 64 | isolated:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1275 | 0 | 57277 | 11832 |
| test_bridge | 64 | shared:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 1210 | 3117 | 74435 | 0 |
| test_bridge | 64 | shared:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1124 | 1518 | 52998 | 10173 |
| test_bridge | 64 | shared:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1116 | 2028 | 54308 | 11152 |
| test_bridge | 64 | dominant:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 1210 | 3117 | 74435 | 0 |
| test_bridge | 64 | dominant:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1117 | 1677 | 53524 | 4962 |
| test_bridge | 64 | dominant:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1116 | 2028 | 54308 | 5347 |
| test_bridge | 64 | learned-141 | 96/96 | 48/48 | 24/24 | 24/24 | 1080 | 2261 | 58921 | 5632 |
| test_bridge | 64 | learned-149 | 96/96 | 48/48 | 24/24 | 24/24 | 1067 | 2255 | 58550 | 5593 |
| test_bridge | 128 | isolated:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2410 | 0 | 129922 | 0 |
| test_bridge | 128 | isolated:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1275 | 0 | 57277 | 11832 |
| test_bridge | 128 | isolated:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1275 | 0 | 57277 | 11832 |
| test_bridge | 128 | shared:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 1210 | 3117 | 74435 | 0 |
| test_bridge | 128 | shared:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1124 | 1518 | 52998 | 10173 |
| test_bridge | 128 | shared:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1116 | 2028 | 54308 | 11152 |
| test_bridge | 128 | dominant:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 1210 | 3117 | 74435 | 0 |
| test_bridge | 128 | dominant:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1117 | 1677 | 53524 | 4962 |
| test_bridge | 128 | dominant:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1116 | 2028 | 54308 | 5347 |
| test_bridge | 128 | learned-141 | 96/96 | 48/48 | 24/24 | 24/24 | 1080 | 2261 | 58921 | 5632 |
| test_bridge | 128 | learned-149 | 96/96 | 48/48 | 24/24 | 24/24 | 1067 | 2255 | 58550 | 5593 |
| test_bridge | 256 | isolated:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2410 | 0 | 129922 | 0 |
| test_bridge | 256 | isolated:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1275 | 0 | 57277 | 11832 |
| test_bridge | 256 | isolated:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1275 | 0 | 57277 | 11832 |
| test_bridge | 256 | shared:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 1210 | 3117 | 74435 | 0 |
| test_bridge | 256 | shared:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1124 | 1518 | 52998 | 10173 |
| test_bridge | 256 | shared:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1116 | 2028 | 54308 | 11152 |
| test_bridge | 256 | dominant:global_first | 96/96 | 48/48 | 24/24 | 24/24 | 1210 | 3117 | 74435 | 0 |
| test_bridge | 256 | dominant:goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1117 | 1677 | 53524 | 4962 |
| test_bridge | 256 | dominant:shared_relevance | 96/96 | 48/48 | 24/24 | 24/24 | 1116 | 2028 | 54308 | 5347 |
| test_bridge | 256 | learned-141 | 96/96 | 48/48 | 24/24 | 24/24 | 1080 | 2261 | 58921 | 5632 |
| test_bridge | 256 | learned-149 | 96/96 | 48/48 | 24/24 | 24/24 | 1067 | 2255 | 58550 | 5593 |

- Seed 141: 737 parameters, final loss 0.015093667726767691, checkpoint SHA256 f40bc9e681df42da007feefedb7f9c78dd81e4e747a4dc1106ae25c4ce8086a5.
- Seed 149: 737 parameters, final loss 0.014822580790343253, checkpoint SHA256 3e98c244e0dc15ef8e2840dc7cb560af7ddfe3784d7c8b934de13df16790a37a.

[Artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45339224815/artifacts/14895043072), ZIP SHA256 dcf40ac63f21865f61bada24925f7aa858926bc21ee04d3cdcb920f7f748c0dd. The abstract RH interface is a synthetic application contract; no conclusion about the classical RH proof follows. Next: distinguish unrestricted semantic entailment, source-allowlist derivation, and actual submitted-proof dependency. Test matched advertisements with equal semantic truth and distinct provenance evidence.
