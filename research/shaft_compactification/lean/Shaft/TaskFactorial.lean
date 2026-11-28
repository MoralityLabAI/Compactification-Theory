import Shaft.TaskObservability

/-!
SHAFT-COMP-15: finite two-monitor factorial reference.
A rank-one learned coordinate monitor and rank-one native coordinate
monitor respond differently to orthogonal contrasts in the same
two-dimensional state. These proofs concern exact finite arithmetic,
not an empirical ROC-AUC claim or a real-valued Gramian theorem.
-/
namespace ShaftCompact

def secondRead (contrast : Fin 2 → Nat) : Nat := contrast 1
def nativeSquaredRead (contrast : Fin 2 → Nat) : Nat :=
  (secondRead contrast) * (secondRead contrast)

/-- For the first-axis contrast, the learned monitor observes a unit
signal while the native second-axis monitor has zero signal. -/
theorem first_axis_readout_factorial :
    squaredRead observedAxis = 1 ∧
    nativeSquaredRead observedAxis = 0 := by
  decide

/-- For the second-axis contrast, the roles reverse. -/
theorem second_axis_readout_factorial :
    squaredRead hiddenAxis = 0 ∧
    nativeSquaredRead hiddenAxis = 1 := by
  decide

/-- Both monitors produce one scalar reading, but their directional
energies are sensitive to independent state coordinates. -/
theorem rank_one_monitor_directional_separation :
    (squaredRead observedAxis = 1 ∧ squaredRead hiddenAxis = 0) ∧
    (nativeSquaredRead observedAxis = 0 ∧
     nativeSquaredRead hiddenAxis = 1) := by
  exact ⟨same_rank_different_direction_energy,
         first_axis_readout_factorial.2, second_axis_readout_factorial.2⟩

#print axioms first_axis_readout_factorial
#print axioms second_axis_readout_factorial
#print axioms rank_one_monitor_directional_separation

end ShaftCompact
