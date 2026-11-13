import Shaft.CapacityMatched

/-!
SHAFT-COMP-20: exact finite latch state.
The previous-input register is invisible in present behavior, becomes
visible after two queries, and leaves an observable effect after a
common 32-token continuation. This is a Boolean finite-state result.
-/
namespace ShaftCompact

def latchVisible (goal : Nat) (state : Bool × Fin 6) (query : Fin 6) : Bool :=
  Bool.xor (Nat.testBit goal query.val) state.1

def repeatToggle (state : Bool × Fin 6) (q : Fin 6) : Bool × Fin 6 :=
  (Bool.xor state.1 (state.2 == q), q)

def successorToggle (state : Bool × Fin 6) (q : Fin 6) : Bool × Fin 6 :=
  (Bool.xor state.1 (q.val == (state.2.val + 1) % 6), q)

def lastAfterFirst (step : (Bool × Fin 6) → Fin 6 → (Bool × Fin 6))
    (goal : Nat) (state : Bool × Fin 6) (first second : Fin 6) : Bool :=
  latchVisible goal (step state first) second

def stateOld : Bool × Fin 6 := (false, 0)
def stateOther : Bool × Fin 6 := (false, 1)

theorem persistent_current_alias :
    ∀ q : Fin 6, latchVisible 0 stateOld q = latchVisible 0 stateOther q := by
  decide

theorem persistent_repeat_two_query_split :
    lastAfterFirst repeatToggle 0 stateOld 0 2 ≠
      lastAfterFirst repeatToggle 0 stateOther 0 2 := by
  decide

theorem persistent_successor_two_query_split :
    lastAfterFirst successorToggle 0 stateOld 1 2 ≠
      lastAfterFirst successorToggle 0 stateOther 1 2 := by
  decide

def common32 (step : (Bool × Fin 6) → Fin 6 → (Bool × Fin 6))
    (first : Fin 6) (state : Bool × Fin 6) : Bool × Fin 6 :=
  ([first] ++ List.replicate 31 (2 : Fin 6)).foldl step state

theorem repeat_persists_32 :
    latchVisible 0 (common32 repeatToggle 0 stateOld) 3 ≠
      latchVisible 0 (common32 repeatToggle 0 stateOther) 3 := by
  decide

theorem successor_persists_32 :
    latchVisible 0 (common32 successorToggle 1 stateOld) 3 ≠
      latchVisible 0 (common32 successorToggle 1 stateOther) 3 := by
  decide

#print axioms persistent_current_alias
#print axioms persistent_repeat_two_query_split
#print axioms persistent_successor_two_query_split
#print axioms repeat_persists_32
#print axioms successor_persists_32
end ShaftCompact
