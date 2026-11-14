# FC-12 pre-test amendment A: local hypothesis applications

**Date:** November 14, 2026. Applied before any FC-12 held-out theorem selector
or held-out model scores existed. The original preregistration remains
immutable in commit `d6fae0ddb9ce60036cff09e5ec93820ed3879af8`.

The first train-only source was kernel checked in
[CI #43545821464](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43545821464):
27 original MathCan research theorems, 51 real compiler goal states, and
train-state SHA256
`403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0`.

The [initial TRAIN-ONLY action-probe CI #43546204691](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43546204691)
examined **961** concrete tactic candidates on those 51 source states and
correctly **failed** the preregistered mandatory-action-coverage requirement:
`exact_local` achieved **0** real goal-progress successes. The other
three mandatory parameterized action families had genuine accepted progress:
`rw_local` 13, `rcases_local` 1, and `induction_list` 30.

The failure stemmed from candidate generation. It selected the first four
local declarations, usually ordinary `Nat`, `Type`, `List` or function
parameters. It also lacked locally applied hypotheses such as the original
MathCan Intervention theorem's `h word valid` and equality symmetry.
The successful original proof uses a local hypothesis application, whose
arguments are themselves present in the compiler-printed tactic context.

This pre-test amendment extends **only** the registered
`exact_local` parameterization:

- Prioritize hypothesis-like declarations over type/term parameters;
  preserve deterministic original source-context order.
- Permit `exact h x y` and `exact (h x y).symm`, where **all three
  identifiers** come from the same actual Lean state. This includes
  applying locally known equivalences to a locally named action list
  and its well-formedness certificate.
- Continue enforcing **at most four concrete candidates per family**
  and disallow any arbitrary tactic injection or source reference term.
- Leave train/test source names, Lean version, model architecture,
  optimizer, seeds, action count, kernel proof gate and native symbolic
  baseline unchanged.
- Re-run the whole original-source train-action matrix and require at
  least one actual Lean-progressing `exact_local`, `rw_local`,
  `rcases_local` and `induction_list` example before freezing
  the train-only source revision and creating the new held-out selector.

This revision is a **pre-test methodological change**, permitted by the
registration's train-only action-coverage checkpoint. The initial negative
coverage result stays available in CI history and is documented here.
It does not create a held-out improvement claim.
