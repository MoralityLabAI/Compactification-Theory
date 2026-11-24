# CG-10 first complete results

[Corrected run 45376085409](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45376085409), job 138382644654, head 3859c2e13753be53484bdfa8b28de864766800b5. [Initial run 45340063996](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45340063996) stopped before training because #print axioms within the namespace printed relative dependency names; the expected dependency indices matched, but fully qualified comparison failed. Commit 3859c2e moves the audit outside the namespace. No corpus, model, seed, budget or gate change. Initial failure artifact 14894844790 SHA256 f837ff5bfd0c664241309e81210f6dcf8e4596476ec111d5a78ea86c1dba28c9.

Native source audits 512; reference certificates 512; distinct actor certificates 256. Both 641-parameter models trained on 256 trajectories, 2322 action groups and 6273 candidate rows. Registered learning gate **FAIL**: provenance-first 278, allowed-first 278, alternating 278, goal-slice 313, learned-151 and learned-157 both 296. Threshold 319. At 16 work units on depth, both learned arms certify the allowed axis in all 96 cases but complete only 9 joint classifications; goal slice completes 25. At 32 units all three certify all 96 deeper cases; the learned arms consume more work. At 128 all policies certify every test case. No invalid accepted axes.

The implemented teacher labels the union of its canonical allowed-derivation plan and available shortest-path/clean-closure provenance actions, rather than optimizing an exact joint shortest completion plan. Treat the run as canonical action-set imitation; no claim of optimal teacher scheduling. Native selected axiom audit and separate allowed implication proofs check different mathematical obligations.

| Split | Budget | Policy | Complete | Provenance | Allowed | Clean | Repairable | Required | Work | Invalid |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cal | 8 | provenance_first | 24/64 | 46 | 25 | 6 | 5 | 13 | 486 | 0 |
| cal | 8 | allowed_first | 24/64 | 24 | 64 | 6 | 5 | 13 | 486 | 0 |
| cal | 8 | alternating | 24/64 | 26 | 38 | 6 | 5 | 13 | 486 | 0 |
| cal | 8 | goal_slice | 32/64 | 46 | 33 | 10 | 9 | 13 | 477 | 0 |
| cal | 8 | learned-151 | 28/64 | 28 | 64 | 10 | 6 | 12 | 487 | 0 |
| cal | 8 | learned-157 | 28/64 | 28 | 64 | 10 | 6 | 12 | 487 | 0 |
| cal | 16 | provenance_first | 60/64 | 64 | 60 | 29 | 15 | 16 | 676 | 0 |
| cal | 16 | allowed_first | 60/64 | 60 | 64 | 29 | 15 | 16 | 676 | 0 |
| cal | 16 | alternating | 60/64 | 60 | 64 | 29 | 15 | 16 | 676 | 0 |
| cal | 16 | goal_slice | 64/64 | 64 | 64 | 32 | 16 | 16 | 612 | 0 |
| cal | 16 | learned-151 | 64/64 | 64 | 64 | 32 | 16 | 16 | 624 | 0 |
| cal | 16 | learned-157 | 64/64 | 64 | 64 | 32 | 16 | 16 | 624 | 0 |
| cal | 32 | provenance_first | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 32 | allowed_first | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 32 | alternating | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 32 | goal_slice | 64/64 | 64 | 64 | 32 | 16 | 16 | 612 | 0 |
| cal | 32 | learned-151 | 64/64 | 64 | 64 | 32 | 16 | 16 | 624 | 0 |
| cal | 32 | learned-157 | 64/64 | 64 | 64 | 32 | 16 | 16 | 624 | 0 |
| cal | 64 | provenance_first | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 64 | allowed_first | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 64 | alternating | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 64 | goal_slice | 64/64 | 64 | 64 | 32 | 16 | 16 | 612 | 0 |
| cal | 64 | learned-151 | 64/64 | 64 | 64 | 32 | 16 | 16 | 624 | 0 |
| cal | 64 | learned-157 | 64/64 | 64 | 64 | 32 | 16 | 16 | 624 | 0 |
| cal | 128 | provenance_first | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 128 | allowed_first | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 128 | alternating | 64/64 | 64 | 64 | 32 | 16 | 16 | 683 | 0 |
| cal | 128 | goal_slice | 64/64 | 64 | 64 | 32 | 16 | 16 | 612 | 0 |
| cal | 128 | learned-151 | 64/64 | 64 | 64 | 32 | 16 | 16 | 624 | 0 |
| cal | 128 | learned-157 | 64/64 | 64 | 64 | 32 | 16 | 16 | 624 | 0 |
| test_depth | 8 | provenance_first | 0/96 | 20 | 5 | 0 | 0 | 0 | 768 | 0 |
| test_depth | 8 | allowed_first | 0/96 | 0 | 36 | 0 | 0 | 0 | 768 | 0 |
| test_depth | 8 | alternating | 0/96 | 10 | 14 | 0 | 0 | 0 | 768 | 0 |
| test_depth | 8 | goal_slice | 0/96 | 20 | 5 | 0 | 0 | 0 | 768 | 0 |
| test_depth | 8 | learned-151 | 0/96 | 0 | 54 | 0 | 0 | 0 | 768 | 0 |
| test_depth | 8 | learned-157 | 0/96 | 0 | 54 | 0 | 0 | 0 | 768 | 0 |
| test_depth | 16 | provenance_first | 19/96 | 50 | 23 | 0 | 2 | 17 | 1475 | 0 |
| test_depth | 16 | allowed_first | 19/96 | 19 | 88 | 0 | 2 | 17 | 1475 | 0 |
| test_depth | 16 | alternating | 19/96 | 27 | 40 | 0 | 2 | 17 | 1475 | 0 |
| test_depth | 16 | goal_slice | 25/96 | 50 | 29 | 0 | 8 | 17 | 1453 | 0 |
| test_depth | 16 | learned-151 | 9/96 | 9 | 96 | 0 | 0 | 9 | 1520 | 0 |
| test_depth | 16 | learned-157 | 9/96 | 9 | 96 | 0 | 0 | 9 | 1520 | 0 |
| test_depth | 32 | provenance_first | 74/96 | 96 | 74 | 26 | 24 | 24 | 2304 | 0 |
| test_depth | 32 | allowed_first | 74/96 | 74 | 96 | 26 | 24 | 24 | 2304 | 0 |
| test_depth | 32 | alternating | 74/96 | 74 | 91 | 26 | 24 | 24 | 2304 | 0 |
| test_depth | 32 | goal_slice | 96/96 | 96 | 96 | 48 | 24 | 24 | 2144 | 0 |
| test_depth | 32 | learned-151 | 96/96 | 96 | 96 | 48 | 24 | 24 | 2380 | 0 |
| test_depth | 32 | learned-157 | 96/96 | 96 | 96 | 48 | 24 | 24 | 2374 | 0 |
| test_depth | 64 | provenance_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 2393 | 0 |
| test_depth | 64 | allowed_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 2393 | 0 |
| test_depth | 64 | alternating | 96/96 | 96 | 96 | 48 | 24 | 24 | 2393 | 0 |
| test_depth | 64 | goal_slice | 96/96 | 96 | 96 | 48 | 24 | 24 | 2144 | 0 |
| test_depth | 64 | learned-151 | 96/96 | 96 | 96 | 48 | 24 | 24 | 2380 | 0 |
| test_depth | 64 | learned-157 | 96/96 | 96 | 96 | 48 | 24 | 24 | 2374 | 0 |
| test_depth | 128 | provenance_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 2393 | 0 |
| test_depth | 128 | allowed_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 2393 | 0 |
| test_depth | 128 | alternating | 96/96 | 96 | 96 | 48 | 24 | 24 | 2393 | 0 |
| test_depth | 128 | goal_slice | 96/96 | 96 | 96 | 48 | 24 | 24 | 2144 | 0 |
| test_depth | 128 | learned-151 | 96/96 | 96 | 96 | 48 | 24 | 24 | 2380 | 0 |
| test_depth | 128 | learned-157 | 96/96 | 96 | 96 | 48 | 24 | 24 | 2374 | 0 |
| test_clutter | 8 | provenance_first | 30/96 | 80 | 31 | 3 | 8 | 19 | 717 | 0 |
| test_clutter | 8 | allowed_first | 30/96 | 30 | 92 | 3 | 8 | 19 | 717 | 0 |
| test_clutter | 8 | alternating | 30/96 | 38 | 60 | 3 | 8 | 19 | 717 | 0 |
| test_clutter | 8 | goal_slice | 51/96 | 80 | 52 | 14 | 18 | 19 | 701 | 0 |
| test_clutter | 8 | learned-151 | 33/96 | 33 | 96 | 14 | 5 | 14 | 738 | 0 |
| test_clutter | 8 | learned-157 | 36/96 | 36 | 96 | 14 | 6 | 16 | 740 | 0 |
| test_clutter | 16 | provenance_first | 89/96 | 96 | 89 | 41 | 24 | 24 | 985 | 0 |
| test_clutter | 16 | allowed_first | 89/96 | 89 | 96 | 41 | 24 | 24 | 985 | 0 |
| test_clutter | 16 | alternating | 89/96 | 89 | 93 | 41 | 24 | 24 | 985 | 0 |
| test_clutter | 16 | goal_slice | 96/96 | 96 | 96 | 48 | 24 | 24 | 862 | 0 |
| test_clutter | 16 | learned-151 | 95/96 | 95 | 96 | 48 | 24 | 23 | 957 | 0 |
| test_clutter | 16 | learned-157 | 95/96 | 95 | 96 | 48 | 24 | 23 | 954 | 0 |
| test_clutter | 32 | provenance_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 32 | allowed_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 32 | alternating | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 32 | goal_slice | 96/96 | 96 | 96 | 48 | 24 | 24 | 862 | 0 |
| test_clutter | 32 | learned-151 | 96/96 | 96 | 96 | 48 | 24 | 24 | 958 | 0 |
| test_clutter | 32 | learned-157 | 96/96 | 96 | 96 | 48 | 24 | 24 | 955 | 0 |
| test_clutter | 64 | provenance_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 64 | allowed_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 64 | alternating | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 64 | goal_slice | 96/96 | 96 | 96 | 48 | 24 | 24 | 862 | 0 |
| test_clutter | 64 | learned-151 | 96/96 | 96 | 96 | 48 | 24 | 24 | 958 | 0 |
| test_clutter | 64 | learned-157 | 96/96 | 96 | 96 | 48 | 24 | 24 | 955 | 0 |
| test_clutter | 128 | provenance_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 128 | allowed_first | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 128 | alternating | 96/96 | 96 | 96 | 48 | 24 | 24 | 1003 | 0 |
| test_clutter | 128 | goal_slice | 96/96 | 96 | 96 | 48 | 24 | 24 | 862 | 0 |
| test_clutter | 128 | learned-151 | 96/96 | 96 | 96 | 48 | 24 | 24 | 958 | 0 |
| test_clutter | 128 | learned-157 | 96/96 | 96 | 96 | 48 | 24 | 24 | 955 | 0 |

- Seed 151: 641 parameters, final loss 0.07662964735582874, checkpoint SHA256 4e3803249c27ed8c92007ec4aca8232186191293b7489f5c0e5602a1acfdb587.
- Seed 157: 641 parameters, final loss 0.0752453858385215, checkpoint SHA256 1522d1e504ab89f7922fa35ee5c8c7ca58ca357ecd4cec5ecef36ffecb359f94.

[Artifact 14911646154](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45376085409/artifacts/14911646154), ZIP SHA256 b2ca0b3e81e1ca3dbdfa47436a404432f3b09e9fa9a244ac0a02c886d3cfe5f6. Next CG-11 replaces generated dependency metadata with native elaborated dependency graphs and explicitly tests deceptive surface syntax.
