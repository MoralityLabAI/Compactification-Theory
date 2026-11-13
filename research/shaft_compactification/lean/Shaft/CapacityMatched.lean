import Shaft.HistoryAlphabet

/-!
SHAFT-COMP-19: exact information control for repeated immediate signatures.
A larger output vector formed by repeating the same observed bit contains
no finer distinctions, while a history-dependent bit can split its aliases.
The neural AUC result is an independent empirical claim.
-/
namespace ShaftCompact

/-- A two-coordinate state with its first bit directly observable. -/
def firstBit (state : Bool × Bool) : Bool := state.1

/-- A capacity-matched monitor repeats the first bit 36 times. -/
def repeatFirst36 (state : Bool × Bool) (_index : Fin 36) : Bool := state.1

/-- The 36-output repetition defines exactly the same observational
    equivalence relation as the original single-output monitor. -/
theorem repeat_36_code_equiv (s t : Bool × Bool) :
    (∀ i : Fin 36, repeatFirst36 s i = repeatFirst36 t i) ↔
      firstBit s = firstBit t := by
  constructor
  · intro h
    exact h 0
  · intro h i
    exact h

/-- Equal immediate outputs can coexist with a different second, delayed bit.
    Repetition of that immediate output leaves the two states aliased. -/
theorem repeated_immediate_can_miss_history :
    ∃ s t : Bool × Bool,
      (∀ i : Fin 36, repeatFirst36 s i = repeatFirst36 t i) ∧
      s.2 ≠ t.2 := by
  refine ⟨(false,false),(false,true),?_,?_⟩
  · intro i
    rfl
  · decide

/-- On the six-symbol rotated task, the concrete previous-input/goal
    witness remains aliased under every repeated immediate query. -/
theorem six_current_alias_survives_repetition :
    ∀ i : Fin 36,
       bitSix 1 0 ⟨i.val % 6, Nat.mod_lt _ (by decide)⟩ =
         bitSix 2 1 ⟨i.val % 6, Nat.mod_lt _ (by decide)⟩ := by
  intro i
  exact six_immediate_alias _

/-- The same two six-symbol task states differ under a two-step query. -/
theorem six_history_breaks_repeated_alias :
    (∀ i : Fin 36,
       bitSix 1 0 ⟨i.val % 6, Nat.mod_lt _ (by decide)⟩ =
         bitSix 2 1 ⟨i.val % 6, Nat.mod_lt _ (by decide)⟩) ∧
       bitSix 1 (0 : Fin 6) 0 ≠ bitSix 2 (0 : Fin 6) 0 := by
  exact ⟨six_current_alias_survives_repetition,
          six_history_exposes_alias⟩

#print axioms repeat_36_code_equiv
#print axioms repeated_immediate_can_miss_history
#print axioms six_current_alias_survives_repetition
#print axioms six_history_breaks_repeated_alias

end ShaftCompact
