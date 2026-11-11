import Shaft.TemporalSupervision

/-!
SHAFT-COMP-22: exact canonical-to-behavioral factor map for the six-symbol
persistent latch automata. The two canonical states differing by goal-bit
complementation and latch inversion encode the same observable state.
The empirical linear-probe results are separate from these exact theorems.
-/
namespace ShaftCompact

abbrev GoalBits6 := Fin 6 → Bool

structure FullLatchState where
  goal : GoalBits6
  latch : Bool
  previous : Fin 6

abbrev BehavioralLatchCode := GoalBits6 × Fin 6

def latchTrigger (successorFamily : Bool) (p q : Fin 6) : Bool :=
  if successorFamily then (q.val == (p.val + 1) % 6) else (p == q)

def latchOutput (s : FullLatchState) (q : Fin 6) : Bool :=
  Bool.xor (s.goal q) s.latch

def latchStep (successorFamily : Bool) (q : Fin 6)
    (s : FullLatchState) : FullLatchState :=
  { s with latch := Bool.xor s.latch (latchTrigger successorFamily s.previous q),
           previous := q }

def latchCode (s : FullLatchState) : BehavioralLatchCode :=
  (fun q => Bool.xor (s.goal q) s.latch, s.previous)

def reducedLatchOutput (code : BehavioralLatchCode) (q : Fin 6) : Bool :=
  code.1 q

def reducedLatchStep (successorFamily : Bool) (q : Fin 6)
    (c : BehavioralLatchCode) : BehavioralLatchCode :=
  (fun r => Bool.xor (c.1 r) (latchTrigger successorFamily c.2 q), q)

/-- Every current output factors exactly through the effective goal table. -/
theorem output_factors_through_latch_code
    (s : FullLatchState) (q : Fin 6) :
    latchOutput s q = reducedLatchOutput (latchCode s) q := by
  rfl

/-- The compact effective-goal/previous code commutes with each full
latch transition, for both registered families. -/
theorem latch_code_commutes_with_step
    (family : Bool) (q : Fin 6) (s : FullLatchState) :
    latchCode (latchStep family q s) =
      reducedLatchStep family q (latchCode s) := by
  apply Prod.ext
  · apply funext
    intro r
    change Bool.xor (s.goal r) (Bool.xor s.latch
      (latchTrigger family s.previous q)) =
      Bool.xor (Bool.xor (s.goal r) s.latch)
      (latchTrigger family s.previous q)
    cases hg : s.goal r <;> cases hl : s.latch <;>
      cases ht : latchTrigger family s.previous q <;> rfl
  · rfl

/-- This code is a causal congruence, so every future continuation has
well-defined behavior on its equivalence classes. -/
theorem effective_latch_code_causal (family : Bool) :
    CausalCongruence (fun q s => latchStep family q s) latchCode := by
  intro q s t same
  rw [latch_code_commutes_with_step family q s,
      latch_code_commutes_with_step family q t, same]

/-- Arbitrarily long common input words preserve equality of effective
behavioral codes. -/
theorem equal_latch_codes_have_equal_future_codes
    (family : Bool) (word : List (Fin 6))
    (s t : FullLatchState) (h : latchCode s = latchCode t) :
    latchCode (run (fun q s => latchStep family q s) word s) =
      latchCode (run (fun q s => latchStep family q s) word t) := by
  exact causal_code_all_words
    (fun q s => latchStep family q s) latchCode
    (effective_latch_code_causal family) word s t h

/-- The one-step and two-step output reconstructions require only the
behavioral code, including the previous query symbol. -/
def latchTwoFromCode (family : Bool) (c : BehavioralLatchCode)
    (a b : Fin 6) : Bool :=
  reducedLatchOutput (reducedLatchStep family a c) b

 theorem latch_two_output_factorization (family : Bool)
    (s : FullLatchState) (a b : Fin 6) :
    latchOutput (latchStep family a s) b =
      latchTwoFromCode family (latchCode s) a b := by
  have hc := latch_code_commutes_with_step family a s
  calc
    latchOutput (latchStep family a s) b =
      reducedLatchOutput (latchCode (latchStep family a s)) b :=
        output_factors_through_latch_code _ _
    _ = reducedLatchOutput (reducedLatchStep family a (latchCode s)) b := by rw [hc]
    _ = latchTwoFromCode family (latchCode s) a b := rfl

#print axioms output_factors_through_latch_code
#print axioms latch_code_commutes_with_step
#print axioms effective_latch_code_causal
#print axioms equal_latch_codes_have_equal_future_codes
#print axioms latch_two_output_factorization
end ShaftCompact
