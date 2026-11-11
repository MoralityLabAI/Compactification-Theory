import Shaft.DiagnosticSchedule

/-!
SHAFT-COMP-14: exact finite-dimensional directional-observation reference.

A fixed measurement map of rank one can assign different squared
sensitivities to different state contrasts. The scalar, nonnegative cost
theorems are exact. These proofs do not formalize the real-valued
time-varying neural Gramian or any empirical prediction outcome.
-/
namespace ShaftCompact

def readFirst (contrast : Fin 2 → Nat) : Nat := contrast 0
def squaredRead (contrast : Fin 2 → Nat) : Nat :=
  (readFirst contrast) * (readFirst contrast)

def observedAxis : Fin 2 → Nat := fun i => if i == 0 then 1 else 0
def hiddenAxis : Fin 2 → Nat := fun i => if i == 1 then 1 else 0

theorem same_rank_different_direction_energy :
    squaredRead observedAxis = 1 ∧ squaredRead hiddenAxis = 0 := by
  decide

def weightedStepCost (weight observation : Nat) : Nat :=
    (weight * observation) * (weight * observation)

def weightedTrajectoryCost (readout : Nat → Nat) (weight : Nat → Nat) :
    Nat → Nat
  | 0 => weightedStepCost (weight 0) (readout 0)
  | n+1 => weightedTrajectoryCost readout weight n +
      weightedStepCost (weight (n+1)) (readout (n+1))

/-- Every extra measurement adds nonnegative task-directed energy. -/
theorem task_energy_monotone (readout weight : Nat → Nat) (n : Nat) :
    weightedTrajectoryCost readout weight n ≤
      weightedTrajectoryCost readout weight (n+1) := by
  simp [weightedTrajectoryCost]
  exact Nat.le_add_right _ _

/-- Zero accumulated energy entails zero contribution at every
    observed time, so it remains a genuine blind direction. -/
theorem task_energy_zero_each
    (readout weight : Nat → Nat) (n : Nat)
    (h : weightedTrajectoryCost readout weight n = 0) :
    ∀ k, k ≤ n → weightedStepCost (weight k) (readout k) = 0 := by
  induction n with
  | zero =>
      intro k hk
      have heq : k = 0 := by omega
      subst k
      simpa [weightedTrajectoryCost] using h
  | succ n ih =>
      have hsum : weightedTrajectoryCost readout weight n +
          weightedStepCost (weight (n+1)) (readout (n+1)) = 0 := by
        simpa [weightedTrajectoryCost] using h
      have hprev : weightedTrajectoryCost readout weight n = 0 := by omega
      have hnext : weightedStepCost (weight (n+1)) (readout (n+1)) = 0 := by omega
      intro k hk
      by_cases hkprev : k ≤ n
      · exact ih hprev k hkprev
      · have heq : k = n+1 := by omega
        simpa [heq] using hnext

#print axioms same_rank_different_direction_energy
#print axioms task_energy_monotone
#print axioms task_energy_zero_each

end ShaftCompact
