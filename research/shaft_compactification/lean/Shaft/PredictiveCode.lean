import Shaft.DepthTransfer

/-!
SHAFT-COMP-07: behavioral predictive-state codes and cross-substrate
transfer, all abstract and independent of any model architecture.
No trained-model performance or unproved compression claim follows from
these purely conditional statements.
-/

universe u v w x y
namespace ShaftCompact

/-- A code is predictively adequate if it reconstructs every query response. -/
def PredictiveAdequate {State : Type u} {Query : Type v}
    {Answer : Type w} {Code : Type x}
    (behavior : State → Query → Answer)
    (encode : State → Code) (decode : Code → Query → Answer) : Prop :=
  ∀ s q, decode (encode s) q = behavior s q

/-- The full response signature is a canonical (possibly large) exact code. -/
theorem canonical_signature_adequate
    {State : Type u} {Query : Type v} {Answer : Type w}
    (behavior : State → Query → Answer) :
    PredictiveAdequate behavior (fun s => behavior s)
      (fun code q => code q) := by
  intro s q
  rfl

/-- An exact code separates any states distinguishable by some query. -/
theorem predictive_code_separates
    {State : Type u} {Query : Type v} {Answer : Type w}
    {Code : Type x} (behavior : State → Query → Answer)
    (encode : State → Code) (decode : Code → Query → Answer)
    (adequate : PredictiveAdequate behavior encode decode)
    (s t : State) (distinguishable : ∃ q, behavior s q ≠ behavior t q) :
    encode s ≠ encode t := by
  intro sameCode
  obtain ⟨q, different⟩ := distinguishable
  apply different
  calc
    behavior s q = decode (encode s) q := (adequate s q).symm
    _ = decode (encode t) q := congrArg (fun code => decode code q) sameCode
    _ = behavior t q := adequate t q

/-- If a decoder is a left inverse, the encoder is necessarily injective. -/
theorem left_inverse_encoder_injective
    {Goal : Type u} {Code : Type v}
    (encode : Goal → Code) (decode : Code → Goal)
    (exact : ∀ g, decode (encode g) = g) :
    Function.Injective encode := by
  intro a b same
  calc
    a = decode (encode a) := (exact a).symm
    _ = decode (encode b) := congrArg decode same
    _ = b := exact b

/-- A common exact semantic code equates responses across substrates. -/
theorem predictive_code_cross_substrate
    {Source : Type u} {Target : Type v} {Query : Type w}
    {Answer : Type x} {Code : Type y}
    (sourceBehavior : Source → Query → Answer)
    (targetBehavior : Target → Query → Answer)
    (sourceCode : Source → Code) (targetCode : Target → Code)
    (decode : Code → Query → Answer)
    (s : Source) (t : Target)
    (sourceAdequate : ∀ q, decode (sourceCode s) q = sourceBehavior s q)
    (targetAdequate : ∀ q, decode (targetCode t) q = targetBehavior t q)
    (sameCode : sourceCode s = targetCode t) :
    ∀ q, sourceBehavior s q = targetBehavior t q := by
  intro q
  calc
    sourceBehavior s q = decode (sourceCode s) q := (sourceAdequate q).symm
    _ = decode (targetCode t) q := congrArg (fun code => decode code q) sameCode
    _ = targetBehavior t q := targetAdequate q

/-- Approximate behavioral agreement follows from a shared code and
    independently bounded decoder defects on each substrate. -/
theorem approximate_code_cross_substrate
    {Source : Type u} {Target : Type v} {Query : Type w}
    {Answer : Type x} {Code : Type y}
    (sourceBehavior : Source → Query → Answer)
    (targetBehavior : Target → Query → Answer)
    (sourceCode : Source → Code) (targetCode : Target → Code)
    (decode : Code → Query → Answer) (distance : Answer → Answer → Nat)
    (triangle : ∀ a b c, distance a c ≤ distance a b + distance b c)
    (s : Source) (t : Target) (errorSource errorTarget : Nat)
    (sourceBound : ∀ q, distance (sourceBehavior s q) (decode (sourceCode s) q) ≤ errorSource)
    (targetBound : ∀ q, distance (decode (targetCode t) q) (targetBehavior t q) ≤ errorTarget)
    (sameCode : sourceCode s = targetCode t) :
    ∀ q, distance (sourceBehavior s q) (targetBehavior t q) ≤ errorSource + errorTarget := by
  intro q
  calc
    distance (sourceBehavior s q) (targetBehavior t q) ≤
        distance (sourceBehavior s q) (decode (sourceCode s) q) +
        distance (decode (sourceCode s) q) (targetBehavior t q) :=
      triangle _ _ _
    _ = distance (sourceBehavior s q) (decode (sourceCode s) q) +
        distance (decode (targetCode t) q) (targetBehavior t q) := by
      rw [sameCode]
    _ ≤ errorSource + errorTarget := Nat.add_le_add (sourceBound q) (targetBound q)

/-- A change of hidden coordinates preserves adequacy if observations commute. -/
theorem predictive_code_coordinate_change
    {Old : Type u} {New : Type v} {Query : Type w}
    {Answer : Type x} {Code : Type y}
    (transport : Old → New)
    (oldBehavior : Old → Query → Answer)
    (newBehavior : New → Query → Answer)
    (commutes : ∀ s q, newBehavior (transport s) q = oldBehavior s q)
    (encode : New → Code) (decode : Code → Query → Answer)
    (adequate : PredictiveAdequate newBehavior encode decode) :
    PredictiveAdequate oldBehavior (fun s => encode (transport s)) decode := by
  intro s q
  exact (adequate (transport s) q).trans (commutes s q)

#print axioms canonical_signature_adequate
#print axioms predictive_code_separates
#print axioms left_inverse_encoder_injective
#print axioms predictive_code_cross_substrate
#print axioms approximate_code_cross_substrate
#print axioms predictive_code_coordinate_change

end ShaftCompact
