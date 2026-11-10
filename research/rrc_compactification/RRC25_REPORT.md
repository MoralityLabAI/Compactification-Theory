# RRC-25: latent delayed composite branches, exact algebra and matched CPU ranking

**Executed 2026-11-10; one training seed, fresh split, deterministic exact replay.** The experiment extends RRC-24's delayed factor with two hidden transformed series. The first residual identifies the delay and amplitude but leaves sixteen operator pairs unresolved; the second delayed onset is also shared by all of those pairs. The algorithm is still deliberately supplied with the public operator vocabulary and a first-fault **header decoder**. It does not learn a general algorithm for formal-series factorization.

## Verified held-out program recovery

Every entry below is a **future-correct prediction through degree 256**, after exact degree-192 verification of both public demonstrations. Every method receives the same public four-base scan, and strict methods may fully check at most four candidate suffix pairs.

| Source family (32 cases each) | GRU 4 checks | FF 4 checks | Shuffled GRU 4 | Fixed 4 | Exact preview 4 | Complete factorized, ≤64 |
|---|---:|---:|---:|---:|---:|---:|
| Signed, training-family | 19 | 20 | 10 | 12 | 32 | 32 |
| Sparse signed, training-family | 20 | 19 | 6 | 8 | 32 | 32 |
| Partition products, training-family | **32** | 30 | 7 | 8 | 32 | 32 |
| Rational reciprocals, unseen-family | 5 | **13** | 7 | 7 | **32** | 32 |
| Convolution squares, unseen-family | 22 | 19 | 9 | 12 | **32** | 32 |
| Late-onset inputs, unseen-family | 3 | 4 | 3 | 4 | 4 | **11** |
| **Total /192** | **101** | **105** | 42 | 51 | **164** | **171** |

The recurrent student directly improves the four-check fixed ordering by **50** future-correct programs and is strong on product-series inputs. Its parameter-matched feedforward counterpart achieves **four more** future-correct predictions in aggregate. The exact short-template preview correctly identifies a usable suffix on all 160 early-informative cases using at most four full checks; it consumes sixteen additional short candidate-pair executions per compatible header and remains the stronger computational baseline. The full solver accepts all 192 **observed** positive prefixes, even though it only predicts 171 future continuations correctly. RRC-25 therefore fails the claimed recurrent-superiority gate.

The recurrent head has **1,000 FP32 parameters**, with twelve live recurrent state values. The MLP has **995 FP32 parameters**, a 0.5% smaller model. At inference the recurrent model performs approximately 6,672 dense MACs for nine steps, versus 968 for the MLP. Increasing the recurrent ranker's full verification budget to sixteen accepts all 192 observed-prefix cases and obtains **172/192 future correct**; the one extra future success versus the complete solver comes from a different guess among finitely indistinguishable suffix templates, rather than stronger evidence about the hidden generating program.

## What the finite-degree ambiguity audit found

All five early-informative source families have **one** exact suffix pair matching each observed degree-192 two-demo prefix (160/160 unique). The late-onset source has **28/32** observationally ambiguous prefixes, with exact numbers of fitting suffix templates distributed as follows:

| Fitting U/V templates per late-onset case | Number of cases |
|---:|---:|
| 1 | 4 |
| 2 | 2 |
| 4 | 11 |
| 8 | 4 |
| 16 | 11 |

The full factorized solver accepts **32/32** observed late-onset prefixes and forecasts **11/32** correctly. It selects the first valid template in its fixed grammar order. The 12 constructed out-of-grammar observed-output mutations are rejected by the complete exact solver. The six base/future-mutant pairs are identical in everything available at inference, differ at hidden degree 224, and force six false future acceptances for the complete solver. Every measured policy's proposals and traces agree within each pair.

## Measured resources

| Method over 192 positive cases | Mean full template checks | Mean nominal coefficient-work proxy | Dense model MACs per case |
|---|---:|---:|---:|
| Recurrent, 4 | 2.755 | 114,411 | 6,672 |
| Feedforward, 4 | 2.839 | 109,449 | 968 |
| Fixed first 4 | 3.380 | 99,532 | 0 |
| Exact short preview, 4 | 1.219 | 227,070 | 0 |
| Complete factorized | 7.542 | 234,738 | 0 |

The work proxy counts exact four-base header decoding, model-feature extraction, sixteen short pair previews when used, and all full candidate operations, with RRC-24's nominal triangular reciprocal cost. It is not real CPU instruction count or wall latency. Header decoding is supplied by hand to **every** policy and typically runs fewer than eight base operations because a mismatching first demonstration can reject a base early. Learned models never preview template executions during inference. Exact preview does, and its higher work proxy is an essential part of its resource footprint. The full solver can make as many as 64 candidate checks if several headers survive, and within a single resolved header it evaluates at most sixteen template pairs.

A *separate* warm CPU timing sample of 24 cases, two shuffled passes, Python 3.13.5 (AMD EPYC 9V74), records medians:

| Arm | Median warm ms/case | Correct futures over 48 timed executions |
|---|---:|---:|
| Recurrent 4 | 4.1618 | 20 |
| Feedforward 4 | 2.4646 | 28 |
| Fixed 4 | 2.8738 | 20 |
| Exact preview 4 | 3.2655 | 40 |
| Complete factorized | 4.7785 | 40 |

This CPU sample is small and contains four cases from each source family. It includes public header decomposition, neural ordering where applicable, and paid observed-prefix execution, while excluding data generation, historical training and future grading. The RRC-25 physical JSON records total and p90 as well. The timing is machine-specific and never a CI correctness threshold.

## Training, provenance and decisions

Training uses 720 cases and 144 development cases from three source families, 400 Adam updates at batch 96 per arm, learning rate .005, CPU FP32. The public teacher supplies each training source's exact header and sixteen-way operator-pair label; no test-only source family is used for optimization. Teacher-forced development top-four label inclusion is 84.03% for recurrent and 75.69% for MLP. The shuffled-label recurrent achieves 19.44%. These development diagnostics do not certify a program. All checked candidates are re-executed against both observed integer coefficient series.

The local tests and source-bound replay cover the first-fault nonidentification witness, full coefficient derivation, independent literal-template ambiguity count, header verification, model/interpreter isolation, optional PyTorch/portable parity, held-out generator boundaries, collision invariance, and all measured scores. Independent retraining must match the three saved checkpoint JSON files byte-for-byte. RRC-24 and RRC-23 regressions remain part of the CI gate.

**Decision:** Two unknown delayed transforms create a real suffix-identification problem beyond RRC-24's first-residual shortcut. A tiny recurrent model learns useful conditional ordering from supplied algebraic residual windows, with a steep decline on unfamiliar rational sources. Exact preview search is the stronger and more accurate finite solver, and the matched feedforward model slightly exceeds recurrence on the main four-check score. More substantial progress would require an operator language whose exact decomposition is expensive and a neural policy that beats a *compute-matched* symbolic preview under fresh family-disjoint tests.
