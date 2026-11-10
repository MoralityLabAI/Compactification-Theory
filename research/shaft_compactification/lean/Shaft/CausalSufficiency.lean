import Shaft.ClusterNoise

/-!
SHAFT-COMP-11: causal completeness of a behavioral state quotient.

An exact decoder that preserves all future behavior requires transition
congruence and observable output factorization. The finite delayed-bit
counterexample shows how a present-day code can omit future-relevant state.
Everything here is a deterministic abstract state machine.
-/

universe u v w x

namespace ShaftCompact

/-- Transition congruence makes a compact code causally sufficient for
the transition system. -/
def CausalCongruence {Input : Type u} {State : Type v} {Code : Type w}
    (step : Input → State → State) (encode : State → Code) : Prop :=
  ∀ a s t, encode s = encode t →
    encode (step a s) = encode (step a t)

/-- Congruence at one step extends to arbitrary finite continuations. -/
theorem causal_code_all_words
    {Input : Type u} {State : Type v} {Code : Type w}
    (step : Input → State → State) (encode : State → Code)
    (congruent : CausalCongruence step encode)
    (word : List Input) (s t : State)
    (same : encode s = encode t) :
    encode (run step word s) = encode (run step word t) := by
  induction word generalizing s t with
  | nil =>
      simpa [run] using same
  | cons a rest ih =>
      simpa [run] using ih (step a s) (step a t) (congruent a s t same)

/-- A right inverse on reachable codes is sufficient for restoring
observable future behavior, provided the code is a transition congruence
and current output factors through the code. -/
theorem causal_reconstruction_sound
    {Input : Type u} {State : Type v} {Code : Type w}
    {Output : Type x}
    (step : Input → State → State) (encode : State → Code)
    (restore : Code → State) (observe : State → Output)
    (out : Code → Output)
    (congruent : CausalCongruence step encode)
    (factor : ∀ s, observe s = out (encode s))
    (section : ∀ s, encode (restore (encode s)) = encode s)
    (word : List Input) (s : State) :
    observe (run step word (restore (encode s))) =
      observe (run step word s) := by
  calc
    observe (run step word (restore (encode s))) =
        out (encode (run step word (restore (encode s)))) := factor _
    _ = out (encode (run step word s)) :=
      congrArg out (causal_code_all_words step encode congruent word
        (restore (encode s)) s (section s))
    _ = observe (run step word s) := (factor _).symm

/-- If two states have equal compact codes but disagree after one
continuation, one reconstruction map cannot preserve both outcomes. -/
theorem one_code_cannot_restore_both
    {Input : Type u} {State : Type v} {Code : Type w}
    {Output : Type x}
    (step : Input → State → State) (encode : State → Code)
    (restore : Code → State) (observe : State → Output)
    (s t : State) (word : List Input)
    (same : encode s = encode t)
    (different : observe (run step word s) ≠ observe (run step word t)) :
    ¬ (observe (run step word (restore (encode s))) =
          observe (run step word s) ∧
       observe (run step word (restore (encode t))) =
          observe (run step word t)) := by
  intro both
  apply different
  calc
    observe (run step word s) =
        observe (run step word (restore (encode s))) := both.1.symm
    _ = observe (run step word (restore (encode t))) := by rw [same]
    _ = observe (run step word t) := both.2

/-- A three-bit shift register carries one hidden bit that becomes visible
only after two activation transitions. -/
def delayedStep (s : Bool × Bool × Bool) (activate : Bool) :
    Bool × Bool × Bool :=
  if activate then (s.2.1, s.2.2, s.2.2) else s

def delayedCode (s : Bool × Bool × Bool) : Bool := s.1

def leftState : Bool × Bool × Bool := (false, false, false)
def rightState : Bool × Bool × Bool := (false, false, true)

theorem delayed_alias_at_zero_and_one :
    delayedCode leftState = delayedCode rightState ∧
    delayedCode (delayedStep leftState true) =
      delayedCode (delayedStep rightState true) := by
  decide

theorem delayed_alias_breaks_after_two :
    delayedCode (delayedStep (delayedStep leftState true) true) ≠
      delayedCode (delayedStep (delayedStep rightState true) true) := by
  decide

#print axioms causal_code_all_words
#print axioms causal_reconstruction_sound
#print axioms one_code_cannot_restore_both
#print axioms delayed_alias_at_zero_and_one
#print axioms delayed_alias_breaks_after_two

end ShaftCompact
