import Std

/-! A finite counterexample to optimality of linear mathematical receipts.
The eighteen rows are the public conditional-obligation worker 0, objective 0,
first endpoint in DataGoblin ALEPH-04. Masks use bit x for local world x.
All eight worlds are equiprobable. A message is fixed before the query.
The integer risk unit is 6720 times total query weight. -/

namespace AlgernonAlgebra.NonlinearReceipt

structure Query where
  target : Nat
  side : Nat
  weight : Nat
  deriving Repr

def queries : List Query := [
  ⟨177,170,169⟩, ⟨177,204,169⟩, ⟨177,240,25⟩,
  ⟨39,170,169⟩, ⟨39,204,16⟩, ⟨39,240,1⟩,
  ⟨46,170,36⟩, ⟨46,204,64⟩, ⟨46,240,196⟩,
  ⟨184,170,4⟩, ⟨184,204,400⟩, ⟨184,240,9⟩,
  ⟨163,170,400⟩, ⟨163,204,64⟩, ⟨163,240,36⟩,
  ⟨58,170,49⟩, ⟨58,204,81⟩, ⟨58,240,400⟩ ]

/-- A Bayes Brier cell with n worlds and k positive labels contributes
k(n-k)/8n. Multiplication by 6720 leaves an integer for every possible n. -/
def cell (q : Query) (mask : Nat) (message verified : Bool) : Nat :=
  let worlds := (List.range 8).filter fun w =>
    mask.testBit w == message && q.side.testBit w == verified
  let n := worlds.length
  let k := (worlds.filter fun w => q.target.testBit w).length
  if n == 0 then 0 else k * (n-k) * (840 / n)

def risk (mask : Nat) : Nat :=
  (queries.map fun q => q.weight *
    (cell q mask false false + cell q mask false true +
     cell q mask true false + cell q mask true true)).sum

/-- The homogeneous GF(2) row with coefficient mask d. An affine constant
only renames the two messages, which preserves the conditional Bayes risk. -/
def linearMask (d : Nat) : Nat :=
  ((List.range 8).filter fun w =>
    ((w.testBit 0 && d.testBit 0) != (w.testBit 1 && d.testBit 1)) !=
      (w.testBit 2 && d.testBit 2)).foldl (fun m w => m + 2^w) 0

theorem actualRisk : risk 116 = 1834000 := by native_decide

theorem denominator : 6720 * (queries.map Query.weight).sum = 15375360 := by native_decide

/-- Check all 256 possible deterministic encoders, including both labelings. -/
theorem unrestrictedMinimum :
    (List.range 256).all (fun m => decide (1834000 ≤ risk m)) = true := by native_decide

theorem bestLinearBound :
    (List.range 8).all (fun d => decide (2205840 ≤ risk (linearMask d))) = true := by native_decide

theorem bestLinearAttained :
    (List.range 8).any (fun d => risk (linearMask d) == 2205840) = true := by native_decide

theorem strictNumeratorGap : 2205840 - 1834000 = 371840 := by decide

/-- A common optimal code at two endpoints stays optimal on nonnegative
integer mixtures. Dividing by their positive common normalizer preserves it. -/
theorem endpointMixture (r₀ r₁ s₀ s₁ a b : Nat) (h₀ : r₀ ≤ s₀) (h₁ : r₁ ≤ s₁) :
    a*r₀ + b*r₁ ≤ a*s₀ + b*s₁ :=
  Nat.add_le_add (Nat.mul_le_mul_left a h₀) (Nat.mul_le_mul_left b h₁)

end AlgernonAlgebra.NonlinearReceipt
