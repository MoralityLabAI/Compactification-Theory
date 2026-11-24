# CG-08 first complete results

[Run 45338431845](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45338431845), job 138255954147, source head 4bbead646bc0de2e123e5f99c99cee52f8703ca7. The unchanged MathCan RHMainLogic source SHA256 was ba2645a310f81497dfd35aa0e9e19c31b6e1cf5d261eb489e66415743d5153a2. Native reference replay checked 512 certificates and 640 declarations; actor replay checked 256 distinct certificates and 320 declarations. Both 577-parameter models trained on 256 trajectories, 2593 action sets and 34195 candidate rows.

Registered gate **FAIL**. Combined held-out 16/32-firing complete counts: lex 81, random 88, global-first 189, goal-slice 260, learned-131 237, learned-137 236. Both learned seeds required 266. At 32 firings on depth test, learned policies completed 48/96 (all 24 independent-with-gap and 24 refuted; 0/48 source-supported), while goal-slice completed 89/96 (48 supported, 22 gap, 19 refuted). At 128 firings every policy completed every test case. All accepted certificates passed the independent Python and native checkers; no invalid certificates. The generated Horn advertisements are counterfactual applications of the abstract original interface, not allegations about classical RH.

| Split | Budget | Policy | Complete | Source-supported | Independent + gap | Refuted | Firings | Rule scans |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cal | 8 | lex | 5/64 | 0/32 | 2/16 | 3/16 | 499 | 33103 |
| cal | 8 | random | 4/64 | 0/32 | 1/16 | 3/16 | 504 | 33980 |
| cal | 8 | global_first | 24/64 | 0/32 | 11/16 | 13/16 | 432 | 32072 |
| cal | 8 | goal_slice | 19/64 | 10/32 | 5/16 | 4/16 | 480 | 25548 |
| cal | 8 | learned-131 | 32/64 | 8/32 | 11/16 | 13/16 | 427 | 29971 |
| cal | 8 | learned-137 | 32/64 | 8/32 | 11/16 | 13/16 | 427 | 30115 |
| cal | 16 | lex | 15/64 | 4/32 | 5/16 | 6/16 | 937 | 55667 |
| cal | 16 | random | 18/64 | 5/32 | 7/16 | 6/16 | 944 | 57003 |
| cal | 16 | global_first | 34/64 | 2/32 | 16/16 | 16/16 | 700 | 47049 |
| cal | 16 | goal_slice | 63/64 | 32/32 | 15/16 | 16/16 | 675 | 30500 |
| cal | 16 | learned-131 | 46/64 | 14/32 | 16/16 | 16/16 | 607 | 38151 |
| cal | 16 | learned-137 | 44/64 | 12/32 | 16/16 | 16/16 | 627 | 39029 |
| cal | 32 | lex | 53/64 | 21/32 | 16/16 | 16/16 | 1427 | 77338 |
| cal | 32 | random | 53/64 | 24/32 | 14/16 | 15/16 | 1398 | 78539 |
| cal | 32 | global_first | 49/64 | 17/32 | 16/16 | 16/16 | 1099 | 60410 |
| cal | 32 | goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 676 | 30521 |
| cal | 32 | learned-131 | 64/64 | 32/32 | 16/16 | 16/16 | 701 | 39789 |
| cal | 32 | learned-137 | 64/64 | 32/32 | 16/16 | 16/16 | 723 | 40707 |
| cal | 64 | lex | 64/64 | 32/32 | 16/16 | 16/16 | 1565 | 82413 |
| cal | 64 | random | 64/64 | 32/32 | 16/16 | 16/16 | 1472 | 81302 |
| cal | 64 | global_first | 64/64 | 32/32 | 16/16 | 16/16 | 1243 | 63553 |
| cal | 64 | goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 676 | 30521 |
| cal | 64 | learned-131 | 64/64 | 32/32 | 16/16 | 16/16 | 701 | 39789 |
| cal | 64 | learned-137 | 64/64 | 32/32 | 16/16 | 16/16 | 723 | 40707 |
| cal | 128 | lex | 64/64 | 32/32 | 16/16 | 16/16 | 1565 | 82413 |
| cal | 128 | random | 64/64 | 32/32 | 16/16 | 16/16 | 1472 | 81302 |
| cal | 128 | global_first | 64/64 | 32/32 | 16/16 | 16/16 | 1243 | 63553 |
| cal | 128 | goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 676 | 30521 |
| cal | 128 | learned-131 | 64/64 | 32/32 | 16/16 | 16/16 | 701 | 39789 |
| cal | 128 | learned-137 | 64/64 | 32/32 | 16/16 | 16/16 | 723 | 40707 |
| cal | 256 | lex | 64/64 | 32/32 | 16/16 | 16/16 | 1565 | 82413 |
| cal | 256 | random | 64/64 | 32/32 | 16/16 | 16/16 | 1472 | 81302 |
| cal | 256 | global_first | 64/64 | 32/32 | 16/16 | 16/16 | 1243 | 63553 |
| cal | 256 | goal_slice | 64/64 | 32/32 | 16/16 | 16/16 | 676 | 30521 |
| cal | 256 | learned-131 | 64/64 | 32/32 | 16/16 | 16/16 | 701 | 39789 |
| cal | 256 | learned-137 | 64/64 | 32/32 | 16/16 | 16/16 | 723 | 40707 |
| test_depth | 8 | lex | 2/96 | 0/48 | 1/24 | 1/24 | 767 | 110284 |
| test_depth | 8 | random | 0/96 | 0/48 | 0/24 | 0/24 | 768 | 110462 |
| test_depth | 8 | global_first | 8/96 | 0/48 | 2/24 | 6/24 | 758 | 109784 |
| test_depth | 8 | goal_slice | 2/96 | 0/48 | 1/24 | 1/24 | 762 | 100210 |
| test_depth | 8 | learned-131 | 8/96 | 0/48 | 2/24 | 6/24 | 758 | 109784 |
| test_depth | 8 | learned-137 | 8/96 | 0/48 | 2/24 | 6/24 | 758 | 109784 |
| test_depth | 16 | lex | 2/96 | 0/48 | 1/24 | 1/24 | 1519 | 201995 |
| test_depth | 16 | random | 0/96 | 0/48 | 0/24 | 0/24 | 1536 | 206765 |
| test_depth | 16 | global_first | 37/96 | 0/48 | 18/24 | 19/24 | 1349 | 186016 |
| test_depth | 16 | goal_slice | 2/96 | 0/48 | 1/24 | 1/24 | 1514 | 161350 |
| test_depth | 16 | learned-131 | 37/96 | 0/48 | 18/24 | 19/24 | 1349 | 186016 |
| test_depth | 16 | learned-137 | 37/96 | 0/48 | 18/24 | 19/24 | 1349 | 186016 |
| test_depth | 32 | lex | 10/96 | 2/48 | 4/24 | 4/24 | 2987 | 358705 |
| test_depth | 32 | random | 3/96 | 0/48 | 1/24 | 2/24 | 3065 | 390247 |
| test_depth | 32 | global_first | 48/96 | 0/48 | 24/24 | 24/24 | 2148 | 285251 |
| test_depth | 32 | goal_slice | 89/96 | 48/48 | 22/24 | 19/24 | 2430 | 206577 |
| test_depth | 32 | learned-131 | 48/96 | 0/48 | 24/24 | 24/24 | 2148 | 272496 |
| test_depth | 32 | learned-137 | 48/96 | 0/48 | 24/24 | 24/24 | 2148 | 272273 |
| test_depth | 64 | lex | 61/96 | 17/48 | 22/24 | 22/24 | 4906 | 521002 |
| test_depth | 64 | random | 69/96 | 29/48 | 22/24 | 18/24 | 5326 | 605520 |
| test_depth | 64 | global_first | 56/96 | 8/48 | 24/24 | 24/24 | 3616 | 390114 |
| test_depth | 64 | goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2444 | 207059 |
| test_depth | 64 | learned-131 | 96/96 | 48/48 | 24/24 | 24/24 | 3140 | 307608 |
| test_depth | 64 | learned-137 | 96/96 | 48/48 | 24/24 | 24/24 | 3140 | 307385 |
| test_depth | 128 | lex | 96/96 | 48/48 | 24/24 | 24/24 | 5484 | 556194 |
| test_depth | 128 | random | 96/96 | 48/48 | 24/24 | 24/24 | 5567 | 623448 |
| test_depth | 128 | global_first | 96/96 | 48/48 | 24/24 | 24/24 | 4305 | 413140 |
| test_depth | 128 | goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2444 | 207059 |
| test_depth | 128 | learned-131 | 96/96 | 48/48 | 24/24 | 24/24 | 3140 | 307608 |
| test_depth | 128 | learned-137 | 96/96 | 48/48 | 24/24 | 24/24 | 3140 | 307385 |
| test_depth | 256 | lex | 96/96 | 48/48 | 24/24 | 24/24 | 5484 | 556194 |
| test_depth | 256 | random | 96/96 | 48/48 | 24/24 | 24/24 | 5567 | 623448 |
| test_depth | 256 | global_first | 96/96 | 48/48 | 24/24 | 24/24 | 4305 | 413140 |
| test_depth | 256 | goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 2444 | 207059 |
| test_depth | 256 | learned-131 | 96/96 | 48/48 | 24/24 | 24/24 | 3140 | 307608 |
| test_depth | 256 | learned-137 | 96/96 | 48/48 | 24/24 | 24/24 | 3140 | 307385 |
| test_bridge | 8 | lex | 0/96 | 0/48 | 0/24 | 0/24 | 768 | 51483 |
| test_bridge | 8 | random | 0/96 | 0/48 | 0/24 | 0/24 | 768 | 52524 |
| test_bridge | 8 | global_first | 5/96 | 0/48 | 0/24 | 5/24 | 755 | 54949 |
| test_bridge | 8 | goal_slice | 18/96 | 16/48 | 0/24 | 2/24 | 760 | 39918 |
| test_bridge | 8 | learned-131 | 12/96 | 10/48 | 0/24 | 2/24 | 759 | 48603 |
| test_bridge | 8 | learned-137 | 9/96 | 7/48 | 0/24 | 2/24 | 761 | 48807 |
| test_bridge | 16 | lex | 9/96 | 4/48 | 2/24 | 3/24 | 1503 | 89190 |
| test_bridge | 16 | random | 12/96 | 6/48 | 3/24 | 3/24 | 1511 | 89777 |
| test_bridge | 16 | global_first | 31/96 | 4/48 | 3/24 | 24/24 | 1381 | 89280 |
| test_bridge | 16 | goal_slice | 73/96 | 48/48 | 7/24 | 18/24 | 1150 | 50077 |
| test_bridge | 16 | learned-131 | 56/96 | 31/48 | 7/24 | 18/24 | 1293 | 65487 |
| test_bridge | 16 | learned-137 | 55/96 | 30/48 | 7/24 | 18/24 | 1337 | 66578 |
| test_bridge | 32 | lex | 60/96 | 27/48 | 15/24 | 18/24 | 2615 | 139298 |
| test_bridge | 32 | random | 73/96 | 38/48 | 19/24 | 16/24 | 2426 | 126980 |
| test_bridge | 32 | global_first | 73/96 | 27/48 | 22/24 | 24/24 | 2177 | 116315 |
| test_bridge | 32 | goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1232 | 51721 |
| test_bridge | 32 | learned-131 | 96/96 | 48/48 | 24/24 | 24/24 | 1485 | 69084 |
| test_bridge | 32 | learned-137 | 96/96 | 48/48 | 24/24 | 24/24 | 1538 | 70463 |
| test_bridge | 64 | lex | 96/96 | 48/48 | 24/24 | 24/24 | 2878 | 148522 |
| test_bridge | 64 | random | 96/96 | 48/48 | 24/24 | 24/24 | 2512 | 129694 |
| test_bridge | 64 | global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2370 | 119793 |
| test_bridge | 64 | goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1232 | 51721 |
| test_bridge | 64 | learned-131 | 96/96 | 48/48 | 24/24 | 24/24 | 1485 | 69084 |
| test_bridge | 64 | learned-137 | 96/96 | 48/48 | 24/24 | 24/24 | 1538 | 70463 |
| test_bridge | 128 | lex | 96/96 | 48/48 | 24/24 | 24/24 | 2878 | 148522 |
| test_bridge | 128 | random | 96/96 | 48/48 | 24/24 | 24/24 | 2512 | 129694 |
| test_bridge | 128 | global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2370 | 119793 |
| test_bridge | 128 | goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1232 | 51721 |
| test_bridge | 128 | learned-131 | 96/96 | 48/48 | 24/24 | 24/24 | 1485 | 69084 |
| test_bridge | 128 | learned-137 | 96/96 | 48/48 | 24/24 | 24/24 | 1538 | 70463 |
| test_bridge | 256 | lex | 96/96 | 48/48 | 24/24 | 24/24 | 2878 | 148522 |
| test_bridge | 256 | random | 96/96 | 48/48 | 24/24 | 24/24 | 2512 | 129694 |
| test_bridge | 256 | global_first | 96/96 | 48/48 | 24/24 | 24/24 | 2370 | 119793 |
| test_bridge | 256 | goal_slice | 96/96 | 48/48 | 24/24 | 24/24 | 1232 | 51721 |
| test_bridge | 256 | learned-131 | 96/96 | 48/48 | 24/24 | 24/24 | 1485 | 69084 |
| test_bridge | 256 | learned-137 | 96/96 | 48/48 | 24/24 | 24/24 | 1538 | 70463 |

- Seed 131: 577 parameters, final loss 0.047064169469039616, checkpoint SHA256 b0a7d7caf4827d85a92066c41124f599299a5050522577490daa76b8554a1014.
- Seed 137: 577 parameters, final loss 0.05405526735433718, checkpoint SHA256 e2efdcd5d7008cf4563e4f9457422d2b29189ad9d8c8b50e4d3e33a5229fdec9.

[Artifact 14895496144](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45338431845/artifacts/14895496144), ZIP SHA256 74f59c1afb629ee5192e10cbcf965afd62c76f279bd4790029c88a28abf60299. The protocol uses isolated context proof searches, so a globally established fact may be reproved in several local contexts. Next experiment separates proof-sharing savings from the effect of learned action ranking. A proof of H in Γ entails E → H under arbitrary E; an absent named final bridge is not a countermodel to that semantic obligation when H is independently established. Any provenance-specific objection must specify the restricted derivation contract.
