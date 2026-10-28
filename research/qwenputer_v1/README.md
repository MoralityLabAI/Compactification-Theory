# QPC-T01: evaluator-relative compactification of a compiled memory circuit

Version 1, 2026-10-28. Additive note based on MoralityLabAI/Compactification-Theory
main `85b69af6d412c5be67eced904729f5c70267e820`. An initial HTTP 403 was resolved
before publication. The existing capacity and IC-01 files remain unchanged.

## Exact abstraction obligations

Let X be a declared reachable neural-state domain, U an input alphabet, F its
transition, and phi:X->Z an abstraction to a finite symbolic state. Require:

    phi(F(x,u)) = delta(phi(x),u)
    E(x) = e(phi(x))

for every declared reachable pair (x,u), plus a verified initial-state relation.
The first condition gives trace agreement by induction. The second makes the
abstraction sufficient for the chosen evaluator. To reason about interventions,
also require phi(I_x(x))=I_z(phi(x)) for every allowed intervention pair. These
conditions quantify over a specified task/interface and do not identify every
internal mechanism of a pretrained model.

## Constructive memory example

For row-address S, define F(S,k,v)=S+k*(v-k^T*S). If k=e_i, row i becomes v and
all other rows are preserved. Reading e_i returns v. Exact rational arithmetic
therefore gives a fully specified symbolic RAM example. With n cells, word
alphabet A and a finite instruction interface, there are |A|^n memory states;
additional registers/control state must be counted separately. A bounded finite
trace checker is decidable. This count assumes a finite word alphabet, unlike an
unbounded rational-valued memory.

For orthogonal U, transform S to U*S and all keys/queries to U*k and U*q. Then
F(U*S,U*k,v)=U*F(S,k,v). A fixed address-space orthogonal transform preserves this
memory transition, with decoder phi_U(S')=phi(U^T*S'). This supplies a clean
coordinate-obfuscation test. Moving only S fails the contract. Extending the
transformation to the full hybrid network needs architecture-specific verification.

A loss of information is justified only relative to the allowed observations and
interventions. If future queries can read every independent cell, two memories
that differ in any cell are observationally distinguishable. A quotient that
merges them cannot preserve all those reads. A narrower evaluator can admit a
smaller quotient. Spectral dimension alone does not settle that choice.

A negative fixture uses memories (0,0) and (0,1): observing only the first cell
gives the same abstraction, while an evaluator reading the second distinguishes
them. This explains why transition closure alone is insufficient.

## Verification and handoff

EvalMath's `src/spiral_ln/qwenputer_contract.py` and
`paper/qwenputer_contract.md` contain the independent exact reference, negative
controls and source-hashed receipt. They verify 1,512 tiny memory updates and
1,512 corresponding rotations. Those are finite examples supporting the displayed
algebra; no model-wide theorem or Lean kernel check is asserted.

Next: pin the actual compiled circuit and decoder; enumerate instruction-level
contracts; compare neural and symbolic transitions; freeze the evaluator; add
precision/kernel adversaries; then evaluate SAE/VPD/eigenmode recovery using
percolation-interp's separate QPR-01 protocol. Keep the existing capacity and E10
work on its current owner path.

Source motivation: [Qwenputer](https://divagr.com/blog/qwenputer), checked
2026-10-28. The production-model results are author reports. No model weights
were loaded by this patch.
