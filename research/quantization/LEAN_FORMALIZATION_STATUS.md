Common gate continuation (2026-11-16): 23 new replayed roots establish the
uniform 2^(k+1)*D^k transport bound and its exact/robust decoded spatial capture.
Read COMMON_GATE_BOUND_REPORT.md. This checkpoint has 181 research roots.

# Lean spatial formalization status

Hamming/Taylor continuation (2026-11-16): 63 additional proved and replayed
lemmas now connect computed amplitude words, concrete polynomial Taylor matrices,
fixed-width correction frames and scheduled decoded spatial capture. Read
HAMMING_TAYLOR_FORMALIZATION_REPORT.md for the current 158-root research status.
Remainder lists and communication statuses below describe earlier checkpoints.


Decoder continuation (2026-11-16): 39 additional lemmas prove exact/robust
sentinel decoding and its composition with clipped spatial jet capture.
See DECODER_FORMALIZATION_REPORT.md for the current 95-root research status,
trace fixtures and remaining obligations. Publication of the research branches
is now authorized; the earlier local-only statements below describe the prior run.


The spatial interface now has **42 new proved lemmas**, all replayed through
the pinned OxiLean checker. The main composition theorem is
`MathCan.SpatialClipped.safe_spatial_stream`. It derives exact vector jet
selection from explicit safe-cell inequalities and bounded stored jets.
The full analytic approximation theorem remains open in Lean.

## Proven spatial statement

Work over an arbitrary scalar type with Lean core's standard `Field`, linear
order, compatible `OrderedRing`, and decidable comparison instances. There
is no custom axiom, admitted proof, assumed address correctness, or assumed
capture correctness. A concrete real-number type and its instances are not
constructed in this stage.

Let K>0 and delta>0. For every coordinate record s, assume

    s.coarse <= n,  s.fine < K,
    c*K+i+delta <= s.position <= c*K+i+1-delta,

where c=s.coarse, i=s.fine, and natural numbers are cast into the scalar field.
The paper's scaled coordinate is u=m*x+1-theta. Coarse ramps at (j+1)*K
recover c exactly; fine ramps after subtracting c*K recover i exactly.
The padded edge block c=n and K=1 are included. All lists and jet dimensions
are arbitrary and finite, including the empty coordinate list.

The explicit triangular ReLU table computes parity at the recovered fine
indices. The normalized snake recurrence uses gate constant 2 and produces
rank/B, B=K^d. Clipping each rank register to [0,1] preserves this value.
The main theorem amplifies it algebraically by B and runs a clipped streaming
capture register for every jet coordinate. If stored jets lie in [-C,C],
C>=0, the output equals exactly the jet at the serpentine rank. Every capture
register stays in [-C,C] for arbitrary scalar inputs, even off the safe set.

Separate boundary-anchor lemmas show that clipping anchors into the domain
preserves local coordinates in [0,1] on safe cells. Thus the spatial algebra
does not require a smooth extension outside the cube.

## Formal modules

| Module | Lemmas | Result |
|---|---:|---|
| `SpatialScalar` | 23 | ReLU, clipping, ramp plateaus, coarse/fine addresses, boundary anchors, one-hot sums |
| `SpatialRouting` | 6 | Three-hinge hats, table reads, snake rank bound and algebraic capture |
| `SpatialNormalized` | 4 | Bounded-gate normalized recurrence, parity lookup, geometric address composition |
| `SpatialClipped` | 9 | Clipped rank and capture invariants; `safe_spatial_stream` |

The older 14 recurrence and integer spatial lemmas remain independently
accepted. There are now 56 research theorem roots across the two corpora.
The unchanged Hodge theorem remains separately accepted.

## Replay evidence

- Lean 4.32.0-rc1: all four modules compile and pass `leanchecker`.
- Preserved project Lean 4.35.0: all four modules compile and pass `leanchecker`;
  its Lean and leanchecker binary hashes remain unchanged.
- OxiLean v0.1.5: **3,091 verified, 0 unsupported, 0 rejected** records.
- Every one of the 42 named theorem roots has an explicit verified verdict,
  and every transitive dependency record verifies.
- Native/reference-parser comparison: all **3,408 exported constants**
  have matching names, kinds, universe parameters, types and proof/value terms.
  Only expression metadata and let nondep hints are normalized.
- Export SHA-256: `40a100d7032d992d85b15ef7c393ecf6a4492be444fefe3d7d39f047b2614e2b`.
- Axiom closure: only `propext`, `Quot.sound`, and `Classical.choice`.
  Each root's exact closure agrees with native `#print axioms`.
- Exporter source: `3de59f10bc4b4a0f2de698597aeb1246caa0df0a`.
- OxiLean source: `3adb983d2aefc6a596bdc8b63438f54feecd1f55`.

`pinned432_spatial/manifest.json` enumerates every new theorem. The acceptance
validator requires equality between that list, the source declarations, native
axiom reports, and verified root verdicts. Eight tamper controls exercise its
refusal of missing verdicts/roots, unsupported dependencies, changed terms or
statements, wrong pins/hashes, and an open dependency graph.

## Remaining proof obligations for the full theorem

1. Instantiate the scalar interface with a formal real field; define the
   C^(k,alpha) unit ball on the cube and prove its Taylor remainder estimate,
   including the integer-smoothness convention.
2. Derive `validSites` records from arbitrary safe shifted-grid inputs, using
   floor indices and the padded boundary blocks. Formalize the multidimensional
   coarse lexicographic address and bounded interpolation circuit that retrieves
   packed words and initial jets.
3. Connect the base-4 sentinel decoder, correction stream, amplitude table,
   and quantized multivariate jet transport to the stored jets used here.
   Existing integer recurrence lemmas supply part of this step.
4. Prove the signed-register-to-dense-ReLU compiler, all carry-wire costs,
   bounded weights and fractional-bit counts. The algebraic amplification by
   B in the selector must be replaced by the staged doubling circuit in that
   compiler proof; this scalar theorem does not certify a weight of B.
5. Prove square/multiplication approximation, clipped polynomial evaluation,
   shifted-grid safe-phase counting and median robustness, then combine the
   error and resource inequalities into the complete approximation theorem.

The finite Python circuit checks and the paper construction remain useful
evidence; they are separate from these Lean proof obligations. No full
analytic/network theorem is exported under a weakened or assumed statement.

## Local reproduction

In the original workspace, set `CAMPAIGN_WORK` to the WSL path of `work/` and run

```bash
bash "$CAMPAIGN_WORK/replay_spatial_scalar432.sh"
python3 "$CAMPAIGN_WORK/test_spatial_acceptance.py"
bash "$CAMPAIGN_WORK/check_spatial_new435.sh"
```

The shell pipeline builds changed modules in dependency order, exports every
manifest root, runs OxiLean with full verdicts, audits dependencies, compares
the native proof terms, and runs the acceptance validator. Reproduction after
restoring the bundle uses the same runtime/exporter paths documented in
`MathCan/research/oxilean_v015/README.md`; copy the included helpers to work/.
Toolchain binaries and compiled oleans are not included in the delivery.

All work remains local. No pushes or messages were sent.
