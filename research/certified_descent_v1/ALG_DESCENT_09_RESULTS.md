# ALG-DESCENT-09

# ALG-DESCENT-09 / BPD-RISK-03: frozen basis controls and binding contract

## Scope and outcome
24 fresh OOD roots (420000–420011, 421000–421011), three frozen V8 checkpoints, 16 independent calibration roots and 154,091 OOD prediction cases. Synthetic polynomial routing only; no factoring, DLP or cryptographic security inference. Protocol 1ba3e19; original run aborted on an indexed-child identity assertion, amendment fa78cb4, guarded code c57ad39. The guarded replay is development evidence on spent roots, not a confirmatory test. Calibration selection receipts were byte-identical across runs. Original and guarded run logs are preserved in the replay package.

## Predictive and routing comparison
All 1,896 guarded routes closed; one explicit binding rejection, 563 total rejections. Root-average loss/work:
| Arm | Loss | Work |
|---|---:|---:|
| bucket | .364447 | 997.083 |
| linear | .309056 | 996.944 |
| MLP | .329714 | 1010.306 |
| one-pass (409 parameters) | .360205 | 999.819 |
| dense three-pass (409 parameters) | .329081 | 1009.500 |
| learned dictionary deletion | .429579 | 1007.292 |
| SVD deletion | .450385 | 1008.722 |
| Haar 0 deletion | .327630 | 1004.250 |
| Haar 1 deletion | .387925 | 1004.694 |
| Haar 2 deletion | .387285 | 999.500 |
| Haar 3 deletion | .436142 | 1004.625 |

Three-pass minus one-pass loss = -.031124 (18/24 root wins; descriptive bootstrap [-.048083,-.016687]). Three-pass minus logistic loss = +.020025 (9/24 wins; [-.023874,+.064716]); work +12.556 ([1.083,28.569]). Prediction and routing superiority gates fail. No compute-matched claim: same parameters/training schedule, different recurrent FLOPs.

## Dictionary specificity and physical nulls
12 calibration-selected atoms per dictionary, one selected per checkpoint, six dictionaries per checkpoint. Learned deletion damage .100498 (23/24 roots worse; bootstrap [.069363,.133748]). SVD damage .121304. Learned minus SVD -.020806 (bootstrap [-.053307,.014248]); specificity gate fails. Learned deletion changes first proposal 19/72, complete trace 66/72, work 66/72. Four Haar damage values -.001451, .058844, .058204, .107061. Equal selection budgets do not guarantee equal perturbation magnitudes.

2,048 random physical rank-one edits per checkpoint, 16 selected on calibration-only minimax log-ratio of parameter Frobenius norm, three clean write RMS measurements, final-logit directional derivative RMS. Calibration passes [16,16,16], OOD passes [6,2,0], all five ratios within [.8,1.2]. None achieves required eight valid in both cohorts. Maximum OOD absolute ratio deviation 1.492. All-sham median damage .082542; adequate-subset medians .143875 (seed 19) and .294840 (seed 41), none (67). Do not pool inadequate nulls as mechanistic evidence. Central-difference derivative check 768 cases, relative errors 5.05e-5 to 6.85e-5.

## Binding-contract defect
Exact convolution of the 101,271 eligible candidates found 100 product-valid but misbound candidate lists. 11,721 product-invalid eligible candidates; 13,017 masked. A corrupted certificate can pass the polynomial product equation while supplying children different from the indexed subproblems. Guarded route charges attempted product and rejects unless each payload equals its indexed child key. The observed misbound attempt occurred at root 421002, learned arm, seed 41. Product-validity prediction labels remain unchanged in this replay; full admissibility is different. In this synthetic witness bank, trusted child polynomials are already known and interval partitions are guaranteed, so cheap key comparisons can screen the corruption directly. Redesign the risk task around independent/content-addressed children or costly verified relations rather than this artificial corruption path. Prior results are restricted to their original contract and cohorts; no retroactive claims.

## Reproduction and audit
From the replay package root:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python v9/guarded.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python v9/audit.py

Frozen V8 checkpoint included; scripts reproduce with PyTorch on CPU. V9 receipt retains every prediction and route. Independent coefficient convolution and iterative DFS with independent checkpoint inference reproduced all 3,792 metric fields (max error zero) and 13,272 routing fields; no discrepancies or budget violations. Shares generator, priors and symbolic scoring functions. All artifact hashes are in replay package. The audit source is also committed. No native adversarial BPD, MMS integration, Lean verification or general proof-search claim.

See [DataGoblin PR #89](https://github.com/MoralityLabAI/DataGoblin/pull/89).