# PEF-06 three-paper selection | fixed before finite oracles

Date: 2026-11-15. Independent mathematics, connected to PEF-05
train-only signed-memory margins and causal intervention controls.

Exactly **three distinct OpenAI mathematics preprints** motivate the
new model and experimental controls:

1. *Sharp binary-information contraction on the discrete cube*,
   September 24, 2026, OpenAI/math result 119.
   https://github.com/openai/math/blob/main/preprints/Sharp-binary-information-contraction-on-the-discrete-cube-September-24-2026/main.pdf

2. *Average sensitivity of polynomial threshold functions*,
   September 25, 2026, OpenAI/math result 127.
   https://github.com/openai/math/blob/main/preprints/Average-Sensitivity-of-Polynomial-Threshold-Functions-September-25-2026/main.pdf

3. *A superquadratic separation between sensitivity and block sensitivity*,
   September 25, 2026, OpenAI/math result 132.
   https://github.com/openai/math/blob/main/preprints/A-superquadratic-separation-between-sensitivity-and-block-sensitivity-September-25-2026/paper.pdf

The papers establish general results about noisy Boolean summaries,
polynomial threshold average sensitivity and sensitivity versus block
sensitivity, respectively. PEF-06 derives its own much narrower,
self-contained finite statements. No preprint-level hard theorem is
assumed as an axiom or claimed to have been reproved.

## Fixed mathematical object and null

Let \(m\ge1\) and \(x\in\{0,1\}^{2m}\). Define the coalition latch

    F_m(x) = XOR_{j=0}^{m-1} (x_(2j) AND x_(2j+1)).

The distinguished basepoint is all zero. The comparisons are:
- individual-bit sensitivity and disjoint coalitional block sensitivity;
- exact independent BSC noise response at all zero;
- uniform-source average single-bit sensitivity, uniform-source noisy
  disagreement and exact binary-output joint law;
- full local Möbius expansion of the basepoint's noise-response
  polynomial, checked independently by subset enumeration;
- data-processing/Courtade–Kumar bound as a *numeric sanity check*
  on an exact finite law, without asserting equality.

A frozen LoopedTransformers **posthoc** companion compares
old-input single versus two-input natural interventions in the
previously archived PEF-05 models. It retains all twelve frozen
arm/seed checkpoints and never treats the reused source universe
as a new heldout generalization claim.

No training, GPU or spending is needed for PEF-06 finite oracles.
