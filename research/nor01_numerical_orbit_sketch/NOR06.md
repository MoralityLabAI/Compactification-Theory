# NOR-06: input-conditioned entry into the trained GRU's nonlinear basin

Research branch from the merged NOR-01–05 campaign. Status: **FP64 discovery completed; independent rational capture test underway**. The result concerns a trained RRC-25 zero-input continuation, with native suffix-template accuracy left for a separate experiment.

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

The squared family's smallest measured distance is approximately 5.0209. This is a finite FP64 observation. No input-conditioned entry has yet been certified with interval arithmetic.

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
would certify entry into NOR-04's invariant ball.

This verifier initially targets signed-family input, selected because the frozen FP64 survey measured entry at step 1306. The squared family is a short negative control. If the mathematical enclosure becomes too broad, the script emits an explicit inconclusive receipt with its source hashes and final error radius, preserving the distinction between failed certification and actual instability.

The companion [MathCan NOR-06 theorem](https://github.com/patrickdugan/MathCan/tree/research/nor06-certified-basin-capture-20261115/research/nor06) shows how a finite capture witness implies all-depth nonlinear convergence and a finite-history \(L_p\) all-time metric bound using OpenAI Math #094 plus the existing NOR-01 stitching result.

## Reproduce

Run from the repository root with Python 3.12 and its standard library:

    python -m unittest research.nor01_numerical_orbit_sketch.test_nor06_basin_survey -v
    python research/nor01_numerical_orbit_sketch/nor06_basin_survey.py --check --max-steps 4800 --out /tmp/nor06_discovery.json
    python -m unittest research.nor01_numerical_orbit_sketch.test_nor06_capture -v
    python research/nor01_numerical_orbit_sketch/nor06_capture.py --family signed --max-blocks 144 --check --out /tmp/nor06_signed_capture.json

The [survey CI](../../.github/workflows/nor06-basin-discovery.yml) and [rational-capture CI](../../.github/workflows/nor06-rational-capture.yml) use separate artifact names and report their evidence categories independently. Neither workflow changes the checkpoint or claims native proof-ranking improvements.
