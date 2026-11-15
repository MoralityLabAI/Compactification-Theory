# NOR-06: input-conditioned entry into the trained GRU's nonlinear basin

Research branch from the merged NOR-01–05 campaign. Status: **FP64 six-family discovery and exact-rational signed-family capture completed**. The signed capture passed [GitHub Actions run 43695479529](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43695479529). The result concerns a trained RRC-25 zero-input continuation, with native suffix-template accuracy left for a separate experiment.

## Established premises

NOR-04 proved that the genuinely nonlinear 12-step map \(G=F^{12}\) of the source-locked 1,000-parameter trained GRU contracts uniformly on a Euclidean ball \(B(c,R)\), with \(R=10^{-11}\) and \(K\le0.928242792263\). NOR-05 established that the unique fixed point \(p\) of this 12-step map is stationary under the original one-step \(F\). The question is whether states generated from actual public RRC-25 feature sequences enter this ball.

## Executed FP64 discovery

The [survey implementation](nor06_basin_survey.py) uses the original public nine-by-eight feature rows, native GRU weights, the source checkpoint SHA, and a native sixteen-logit parity gate. A mutation test alters hidden program metadata and future labels to check that input features remain identical. For six source families, the seeds are \(\mathrm{compound25.SEED}+903900+1009i\); the first generated source case with a public header is chosen for family index \(i\).

The [executed CI run 43695018639](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43695018639) passed all five survey tests. First measured FP64 entries into the radius-\(10^{-11}\) ball are:

| Family | Extra zero-input step at first measured entry | Initial center distance |
| --- | ---: | ---: |
| signed | 1306 | 0.33444909 |
| sparse | 1340 | 0.19052422 |
| product | 1380 | 2.69976558 |
| rational | 1420 | 4.53192847 |
| squared | none through 4800 | 5.51493307 |
| late_source | 1340 | 0.26220061 |

The squared family's smallest measured distance is approximately 5.0209. This is a finite FP64 observation. The signed family now has an additional independent **exact-rational finite-entry certificate**, described below.

## Exact rational block shadowing

The [NOR-06B verifier](nor06_capture.py) lifts the original public feature floats and model checkpoint JSON decimals to exact rationals, then encloses the first nine GRU steps with rigorously bounded analytic sigmoid and tanh intervals. Let \(c_0\) be a rational reference state for that hidden output and let \(\rho_0\) be a proven Euclidean state-error radius.

Each subsequent 12-step block builds a rational box enclosing the Euclidean ball \(B(c_k,\rho_k)\) intersected with the globally invariant hidden unit cube. At every intermediate step, it computes interval enclosures of the true GRU Jacobian. Their ordered matrix product supplies a uniform Frobenius norm upper bound \(K_k\) for the block derivative. A separate exact-rational evaluation of the nominal center supplies a new rational center \(c_{k+1}\) and a rigorous local residual \(e_k\).

The validated recurrence is
\[
\boxed{\rho_{k+1}=\operatorname{roundUp}(e_k+K_k\rho_k).}
\]
The mean value theorem proves inductively that the actual trained zero-input state after \(12k\) steps from the public input lies within \(\rho_k\) of \(c_k\). A terminal exact-rational check of
\[
\boxed{\|c_k-c\|_2+\rho_k\le10^{-11}}
\]
**does certify entry** into NOR-04's invariant ball for the signed-family case, as established by the completed 109-block proof.

This verifier initially targets signed-family input, selected because the frozen FP64 survey measured entry at step 1306. The squared family is a short negative control. If the mathematical enclosure becomes too broad, the script emits an explicit inconclusive receipt with its source hashes and final error radius, preserving the distinction between failed certification and actual instability.

The companion [MathCan NOR-06 theorem](https://github.com/patrickdugan/MathCan/tree/research/nor06-certified-basin-capture-20261115/research/nor06) shows how a finite capture witness implies all-depth nonlinear convergence and a finite-history \(L_p\) all-time metric bound using OpenAI Math #094 plus the existing NOR-01 stitching result.

## Executed rational source capture

[Independent 109-block CI run 43695479529](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43695479529) passed both stages. The first ran exact-rational interval control tests. The second completed a full 12-step-block proof for signed and a twelve-block negative-control trial for squared, retaining the full source-bound JSON artifacts.

The signed-family source features are frozen by SHA-256 `2ba283d8a67e2aeb4d7c6e8920a11d93723df88a921342171036b267902466da`, with the original checkpoint SHA-256 `54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b`. At **109 complete blocks**, corresponding to 1,308 additional zero-input transitions after the public nine-input prefix, the verifier proves:

\[
\begin{aligned}
\|c_{109}-c_\star\|_2
&\le\frac{957509766970587967489}{10^{32}}
\approx9.57509766970588\times10^{-12},\\
\|h_{1308}-c_{109}\|_2
&\le\frac{9579}{1562500000000000000000000000000}
=6.13056\times10^{-27},\\
R-\|c_{109}-c_\star\|_2-\rho_{109}
&\ge\frac{8498046605882283891}{2\times10^{31}}
\approx4.24902330294114\times10^{-13}>0.
\end{aligned}
\]

Here \(c_\star\) is the exact rational center of the already certified NOR-04 ball of radius \(R=10^{-11}\). Every inequality is an outward-rounded rational upper bound, with the final test decided by exact fractional arithmetic. The full 109-block trace has SHA-256 `58a698820790ea017b285ce9bf6e086f5042d7b2b20233cbbd9f4b008b1954bd`.

NOR-05 then gives, for the original signed-family hidden state \(h_9\), every \(m\ge0\) and \(0\le j<12\),

\[
\left\|F^{1308+12m+j}(h_9)-p\right\|_2
\le
L_{\mathrm{cube}}^j(0.928242792263)^m
\left(9.591632\times10^{-12}\right),
\]

where the numerical constant is a rounded-up display of the exact rational bound \(\|h_{1308}-c_\star\|+s/(1-K)\). The fixed point \(p\) is the one-step, nonlinear zero-input attractor already certified in NOR-04/05. This closes the source-entry premise of the [MathCan NOR-06 finite-capture theorem](https://github.com/patrickdugan/MathCan/pull/54) **for one naturally sourced algebraic RRC-25 case**.

The independent squared-family control checked twelve blocks, retained a center-distance upper estimate near 5.0252055 and an enclosure radius near \(6.132611\times10^{-26}\), and correctly issued `not_established` for entry into the first ball. This finite failure is compatible with the NOR-07 second-attractor hypothesis.

The [frozen audit summary](results/nor06_certified_capture_summary.json) preserves exact fractions, checkpoint and source fingerprints, both trace hashes, the successful workflow run, and explicit certified/inconclusive statuses. [NOR-06C](nor06_receipt_audit.py) checks its arithmetic against the already merged NOR-04/05 receipts without repeating the expensive 109-block computation.

## Reproduce

Run from the repository root with Python 3.12 and its standard library:

    python -m unittest research.nor01_numerical_orbit_sketch.test_nor06_basin_survey -v
    python research/nor01_numerical_orbit_sketch/nor06_basin_survey.py --check --max-steps 4800 --out /tmp/nor06_discovery.json
    python -m unittest research.nor01_numerical_orbit_sketch.test_nor06_capture -v
    python research/nor01_numerical_orbit_sketch/nor06_capture.py --family signed --max-blocks 144 --check --out /tmp/nor06_signed_capture.json

The [survey CI](../../.github/workflows/nor06-basin-discovery.yml), [rational-capture CI](../../.github/workflows/nor06-rational-capture.yml), and [frozen-receipt auditor](../../.github/workflows/nor06-capture-receipt.yml) keep discovery, certified entry, and reproducibility checks separate. Neither workflow changes the checkpoint or claims native proof-ranking improvements.
