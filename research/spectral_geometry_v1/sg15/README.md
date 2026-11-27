# SG-15: exact dual-face recovery with rational certificates

Finite source-bound spectral routing, additive utility, d<=32, T<=16, integer task data. For adjusted utility z_i=u_ij-eta_i, the maximum relaxed value for label j is the maximum of zero (unused) and z_l+sum_{i<l} max(0,z_i)-price_lj for its last visit l. Every optimal last-use alternative defines a Boolean box: the last visit and positive earlier adjusted utilities are required; zero earlier utilities are optional; all later visits and negative earlier utilities are excluded. Exact face recovery branches over the per-label boxes. Per-row minimum/maximum coverage propagates impossible alternatives; once a box is selected for each label, optional slots can be filled independently per row. Node exhaustion is reported as unknown unless the finite stack is empty. The solver never uses the absence of an exact face as an integer optimality certificate.

For a common denominator den, integer multipliers eta represent eta/den. The dual bound is computed in scaled Python integers. A feasible schedule with objective equal to the floor of this bound is globally optimal, whether or not it attains the rational bound itself. The separate last-use formula independently verifies the bound and the exact schedule scorer validates cardinalities and objective. The LP in sg15_lp.py is only a numerical proposal generator: eta free, p_ij>=max(0,u_ij-eta_i), w_j>=0 and w_j>=u_lj-eta_l+sum_{i<l}p_ij-price_lj, minimize sum eta_i*k_i+sum w_j. SciPy HiGHS status and floating objective are not trusted as certificates. Rational reconstruction and a bounded-denominator fallback both produce valid candidate vectors; they need not be dual optimal. The LP is optional for certificate checking and not part of the core DataGoblin dependency set.

## Reproduction

Install DataGoblin and optional SciPy/NumPy. Put the SG-07 and SG-11 gym directories on PYTHONPATH. Run `python -m pytest -q -s training/rlt_gym/sg15_spectral_face/test_spectral_face.py` with that gym also on PYTHONPATH. Copy sg15_exposed.json from the supplied archive to the working directory, then run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg15_spectral_face/sg15.py`. This writes full results and per-root progress. The exposed JSON contains the three prior unresolved inputs only; it is not training or independent evaluation data. The raw run is supplied separately, not committed.

Four targeted tests passed: 4096 exhaustive label-mask comparisons, 128 complete tiny face-feasibility tasks with arbitrary integer/rational multipliers, 48 tiny LP-proposal tasks checked against exhaustive integer schedules, cap and malformed-input controls, and the fractional triangle-cover example (relaxed 28.5, integer 28). Full repository pytest and Ruff not run. No Lean proof or Rust compilation.

## Executed campaign

36 fresh exposed development roots, 12 persistent/12 rotating/12 mixed, d=32,T=12,k=7/8. Single randomized arm order per root. All arms start with SG-14 dual rounding. Block arm runs unchanged SG-12 at 2048 events and independently replays each transcript. Face arms try root-vector multipliers, four further alternating coordinate sweeps, or LP-proposed rational multipliers, respectively, round a feasible schedule and perform bounded face search (4096 nodes). Each result is verified by integer arithmetic. CPU includes setup, proposal, rounding, face recovery and verification for face arms; block timing excludes post-hoc independent transcript replay. Different work units/caps, not a matched-time experiment. No model inference or GPU. All cross-arm intervals intersect, and on each of 36 roots the intersection is exact. No exhaustive high-dimensional oracle. The experiment was not preregistered as confirmation.

| arm | certified /36 | mean gap | mean CPU ms | median CPU ms | exact faces | infeasible faces | face nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocks |31|0.666667|26.397395|16.390043|n/a|n/a|n/a|
| root_face |0|12.472222|3.634946|3.386683|0|36|36|
| sweep_face |19|4.416667|11.956252|8.386220|19|17|47|
| lp_face |36|0|29.997488|22.354504|34|2|61|

No capped unknown face results. LP arm maximum five nodes, 28/36 already certified after rounding before face search, 36/36 afterward. LP faster than block search on 13/36 roots. Two fractional upper bounds certify via integer floor: root 1520003 objective 342 upper 685/2; root 1520010 objective 287 upper 575/2. No claim of LP global optimality based on numerical solver status. For exact-face certificates primal/dual equality proves optimality of both. Two fractional examples establish strict integrality gaps for the proposed bounds only; the triangle-cover fixture establishes a genuine relaxation integrality gap using a fractional feasible solution of cost 3/2. The three previously unresolved fixtures are certified at 261, -31 and -89, in one, one and two face nodes. Previous intervals [258,262], [-35,-29], [-89,-87].

Recommendation: try a few cheap coordinate sweeps with face recovery, then choose the LP fallback or block search under explicit measured budget; test a staged policy on a new cohort. LP solves all 36 but is slower in mean CPU. SciPy 1.17.0 and NumPy 2.3.5, Python 3.13.5, one host, one timing repetition. LP has a two-second solver time limit, which is not a hard end-to-end deadline. Optional dependency and future larger-dimensional validation remain outstanding.

Source blob hashes: spectral_face 61b920e489b1a2a1edf11978843410159e11a67c, sg15_lp 659d43de4b2c0ef1f9eb1f8647a5835ce7e224d1, sg15 8e26616539022fd264afc552c6663b91540dce0e, tests 8dace9d5a66ae0cee33ad62fee152db8e7bba7ff. Raw results SHA256 65d0757dc96e127410427025eeae6bddb7033d32ad7e59d07ac586f97798df62.

## Implementation custody

[DataGoblin campaign and tests](https://github.com/MoralityLabAI/DataGoblin/tree/67d4a4f9d9f986aa1c4a06019344e9a5d8c3e8c6/training/rlt_gym/sg15_spectral_face) · [Exact integer certificate verifier](https://github.com/MoralityLabAI/DataGoblin/blob/67d4a4f9d9f986aa1c4a06019344e9a5d8c3e8c6/src/datagoblin/training_gym/spectral_face.py).

Complete archive `sg15_cpu_campaign.zip`, SHA256 `3dc7c1e27ec113667a7f967854fe7d9b3d3892229fff316c1acf0af2ccc44f70`. Includes executed source, verifier dependencies, full receipts, exposed fixtures, test log and SHA256 manifest. All four committed Python files match the executed Git blob hashes; archive integrity check passed. Generated timing data are supplied separately.
