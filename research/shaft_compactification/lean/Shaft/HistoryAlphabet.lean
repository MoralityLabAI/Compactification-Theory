import Shaft.HistoryProbe

/-!
SHAFT-COMP-18: 4- and 6-query historical alias witnesses.
The one-query output of an indexed goal can agree after a rotation of the
previous token while the all-two-query signature distinguishes the goals.
This finite theorem is architecture-independent and is not a neural claim.
-/
namespace ShaftCompact

def bitFour (g : Nat) (previous query : Fin 4) : Bool :=
  decide ((g / 2 ^ ((previous.val + query.val) % 4)) % 2 = 1)

def bitSix (g : Nat) (previous query : Fin 6) : Bool :=
  decide ((g / 2 ^ ((previous.val + query.val) % 6)) % 2 = 1)

/-- Two different (goal,previous-input) states have the same complete
    one-step behavior at a 4-symbol vocabulary. -/
theorem four_immediate_alias :
    ∀ q : Fin 4, bitFour 1 0 q = bitFour 2 1 q := by
  decide

/-- The final bit after two queries differs because the previous input
    then becomes q1, while goal bits remain different. -/
theorem four_history_exposes_alias :
    bitFour 1 (0 : Fin 4) 0 ≠ bitFour 2 (0 : Fin 4) 0 := by
  decide

theorem six_immediate_alias :
    ∀ q : Fin 6, bitSix 1 0 q = bitSix 2 1 q := by
  decide

theorem six_history_exposes_alias :
    bitSix 1 (0 : Fin 6) 0 ≠ bitSix 2 (0 : Fin 6) 0 := by
  decide

/-- Two-step continuation refinement is strictly stronger in each
    supported vocabulary size; no reconstruction conclusion follows. -/
theorem history_strictly_refines_four :
    (∀ q : Fin 4, bitFour 1 0 q = bitFour 2 1 q) ∧
      bitFour 1 (0 : Fin 4) 0 ≠ bitFour 2 (0 : Fin 4) 0 := by
  exact ⟨four_immediate_alias, four_history_exposes_alias⟩

theorem history_strictly_refines_six :
    (∀ q : Fin 6, bitSix 1 0 q = bitSix 2 1 q) ∧
      bitSix 1 (0 : Fin 6) 0 ≠ bitSix 2 (0 : Fin 6) 0 := by
  exact ⟨six_immediate_alias, six_history_exposes_alias⟩

#print axioms four_immediate_alias
#print axioms four_history_exposes_alias
#print axioms six_immediate_alias
#print axioms six_history_exposes_alias
#print axioms history_strictly_refines_four
#print axioms history_strictly_refines_six
end ShaftCompact
