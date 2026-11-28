import Shaft.PersistentLatch

/-!
SHAFT-COMP-21: exact transition bookkeeping for the persistent latch
and previous-token register. Neural capability is evaluated separately.
-/
namespace ShaftCompact

theorem repeat_previous_after_step
    (s : Bool × Fin 6) (q : Fin 6) : (repeatToggle s q).2 = q := by
  simp [repeatToggle]

theorem successor_previous_after_step
    (s : Bool × Fin 6) (q : Fin 6) : (successorToggle s q).2 = q := by
  simp [successorToggle]

theorem repeat_latch_after_step
    (s : Bool × Fin 6) (q : Fin 6) :
    (repeatToggle s q).1 = Bool.xor s.1 (s.2 == q) := by
  rfl

theorem successor_latch_after_step
    (s : Bool × Fin 6) (q : Fin 6) :
    (successorToggle s q).1 =
      Bool.xor s.1 (q.val == (s.2.val+1)%6) := by
  rfl

theorem output_before_update_independent_previous
    (goal : Nat) (latch : Bool) (p q input : Fin 6) :
    latchVisible goal (latch,p) input =
      latchVisible goal (latch,q) input := by
  rfl

theorem latch_only_is_not_congruent_repeat :
    ¬ CausalCongruence (fun q s => repeatToggle s q) Prod.fst := by
  intro h
  have same : stateOld.1 = stateOther.1 := by decide
  have next : (repeatToggle stateOld (0 : Fin 6)).1 =
      (repeatToggle stateOther (0 : Fin 6)).1 :=
    h 0 stateOld stateOther same
  have ne : (repeatToggle stateOld (0 : Fin 6)).1 ≠
      (repeatToggle stateOther (0 : Fin 6)).1 := by decide
  exact ne next

theorem exact_restoration_of_temporal_state
    (goal : Nat) (word : List (Fin 6)) (state : Bool × Fin 6) :
    latchVisible goal (run (fun q s => repeatToggle s q) word state) (0 : Fin 6) =
    latchVisible goal (run (fun q s => repeatToggle s q) word (id state)) (0 : Fin 6) := by
  rfl

#print axioms repeat_previous_after_step
#print axioms successor_previous_after_step
#print axioms repeat_latch_after_step
#print axioms successor_latch_after_step
#print axioms output_before_update_independent_previous
#print axioms latch_only_is_not_congruent_repeat
#print axioms exact_restoration_of_temporal_state
end ShaftCompact
