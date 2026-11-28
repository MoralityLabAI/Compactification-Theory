import Shaft.NoisyGauge

/-!
SHAFT-COMP-10: cluster-aware observational-margin guarantees and a finite
example showing why mean bit-error rates alone cannot certify alignment.
The actual trained-GRU performance remains empirical evidence, separate from
these universal conditional lemmas.
-/

universe u
namespace ShaftCompact

/-- Bounded errors per independent or dependent anchor block yield a
deterministic score margin. No independence assumption is used. -/
theorem cluster_corruption_certificate
    {Gauge : Type u} (correct : Gauge) (score : Gauge → Nat)
    (observedSeparation anchorCount maxErrors corrupted : Nat)
    (errorBudget : corrupted ≤ anchorCount * maxErrors)
    (correctUpper : score correct ≤ corrupted)
    (otherLower : ∀ g, g ≠ correct →
      observedSeparation ≤ score g + corrupted)
    (separation : 2 * (anchorCount * maxErrors) < observedSeparation) :
    ∀ g, g ≠ correct → score correct < score g := by
  have hBudget : 2 * corrupted ≤ 2 * (anchorCount * maxErrors) :=
    Nat.mul_le_mul_left 2 errorBudget
  have hGap : 2 * corrupted < observedSeparation :=
    Nat.lt_of_le_of_lt hBudget separation
  exact observed_distance_margin correct score observedSeparation
    corrupted correctUpper otherLower hGap

/-- Four-bit mismatch count, reduced entirely to standard Boolean operations. -/
def fourMismatch (expected observed : Fin 4 → Bool) : Nat :=
  (if expected 0 == observed 0 then 0 else 1) +
  (if expected 1 == observed 1 then 0 else 1) +
  (if expected 2 == observed 2 then 0 else 1) +
  (if expected 3 == observed 3 then 0 else 1)

def allZero : Fin 4 → Bool := fun _ => false

def twoOnes : Fin 4 → Bool :=
  fun i => i == (2 : Fin 4) || i == (3 : Fin 4)

/-- One flipped bit can create a tie between two hypotheses. -/
theorem one_error_tie :
    fourMismatch allZero (fun i => i == (2 : Fin 4)) = 1 ∧
    fourMismatch twoOnes (fun i => i == (2 : Fin 4)) = 1 := by
  decide

/-- A different one-bit error preserves a strict score advantage. -/
theorem one_error_unique :
    fourMismatch allZero (fun i => i == (0 : Fin 4)) = 1 ∧
    fourMismatch twoOnes (fun i => i == (0 : Fin 4)) = 3 := by
  decide

#print axioms cluster_corruption_certificate
#print axioms one_error_tie
#print axioms one_error_unique

end ShaftCompact
