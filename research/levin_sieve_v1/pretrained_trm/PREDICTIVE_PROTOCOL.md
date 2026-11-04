# LSV-07 frozen predictive-region protocol

**Registered before the externally pretrained checkpoint run.** The checkpoint, source revision, 64 test puzzle seeds, architecture, and upstream license are inherited unchanged from `source_lock.json`.

**Endpoint.** Number of 5,184 cell outputs whose label is certified unchanged after three recurrent passes, for every perturbation in the declared stage-one box. Measure certified-correct outputs using the held-out Sudoku solution *after* constructing the certificate. Record fully certified and correct puzzles separately.

**Intervention sets.** (1) Every hidden coordinate may be damaged independently within an ℓ∞ radius. (2) Every hidden channel of the first missing Sudoku cell may be damaged, with other inputs unchanged. Radii are `0, 1e-8, 1e-6, 1e-4, 0.02, 0.10, 0.50`. The first missing cell is selected from the input puzzle only. No fine-tuning, checkpoint edits, search over seeds, or adaptive changes to these endpoints are allowed after viewing results.

**Checks.** Compare certified classes to an independent undamaged FP64 implementation with identical stored weights. After certificates have been calculated, apply independent uniform-noise draws using seeds 34 and 89 for radii `0, 0.02, 0.10, 0.50`; any certified label change is an invalidated computation requiring audit, with raw outputs retained. Validate the interval primitives on random points and a synthetic constant block with a proven positive invariant region.

**Evidence labels.** Theorems PG-01 and PG-02 are analytic real-arithmetic results. The implementation of the interval operators in Torch FP64 carries **unverified transcendental/sqrt/dot-product error assumptions**, so the checkpoint's certified fractions must be labelled **conditional numerical certificates**. Their validity for a CPU FP32 or GPU kernel is an open gate. Even a positive synthetic self-map test says nothing about invariance of the actual trained recurrent checkpoint.

**Cost.** Record certificate runtime separately from inference. A certificate that costs orders of magnitude more than fixed four-pass inference may be informative mathematically and economically unhelpful for deployment.
