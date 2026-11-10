import Init

/-!
SHAFT-COMP-05: exact behavioral equivalence relative to an observation map.
All statements concern abstract deterministic transition systems.
This formalization gives no neural reconstruction or real-device mechanism.
-/

universe u v w x
namespace ShaftCompact

/-- Execute a finite word of inputs in order. -/
def run {Input : Type u} {State : Type v}
    (step : Input → State → State) : List Input → State → State
  | [], s => s
  | a :: rest, s => run step rest (step a s)

/-- States are equivalent when every future input word has the same observation. -/
def FutureEq {Input : Type u} {State : Type v} {Output : Type w}
    (step : Input → State → State) (observe : State → Output)
    (s t : State) : Prop :=
  ∀ word : List Input, observe (run step word s) = observe (run step word t)

theorem futureEq_refl {Input : Type u} {State : Type v} {Output : Type w}
    (step : Input → State → State) (observe : State → Output)
    (s : State) : FutureEq step observe s s := by
  intro word
  rfl

theorem futureEq_symm {Input : Type u} {State : Type v} {Output : Type w}
    (step : Input → State → State) (observe : State → Output)
    {s t : State} (h : FutureEq step observe s t) :
    FutureEq step observe t s := by
  intro word
  exact (h word).symm

theorem futureEq_trans {Input : Type u} {State : Type v} {Output : Type w}
    (step : Input → State → State) (observe : State → Output)
    {s t z : State} (hst : FutureEq step observe s t)
    (htz : FutureEq step observe t z) : FutureEq step observe s z := by
  intro word
  exact (hst word).trans (htz word)

/-- Future equivalence is stable under every primitive transition. -/
theorem futureEq_step {Input : Type u} {State : Type v} {Output : Type w}
    (step : Input → State → State) (observe : State → Output)
    {s t : State} (h : FutureEq step observe s t) (a : Input) :
    FutureEq step observe (step a s) (step a t) := by
  intro word
  exact h (a :: word)

/-- An exact behavior-preserving code distinguishes observably different states. -/
theorem adequate_code_separates {Input : Type u} {State : Type v}
    {Output : Type w} {Code : Type x}
    (step : Input → State → State) (observe : State → Output)
    (encode : State → Code)
    (adequate : ∀ s t, encode s = encode t → FutureEq step observe s t)
    (s t : State) (different : ¬ FutureEq step observe s t) :
    encode s ≠ encode t := by
  intro equalCode
  exact different (adequate s t equalCode)

/-- A tiny machine with one visible and one latent Boolean coordinate. -/
def toyStep (a : Bool) (s : Bool × Bool) : Bool × Bool :=
  if a then (s.2, s.2) else s

def toyObserve (s : Bool × Bool) : Bool := s.1

theorem toy_same_current_output :
    toyObserve (false, false) = toyObserve (false, true) := by
  rfl

theorem toy_distinguished_by_future :
    ¬ FutureEq toyStep toyObserve (false, false) (false, true) := by
  intro h
  have hh := h [true]
  simp [run, toyStep, toyObserve] at hh

/-- Complete erasure is a constant map on the toy latent state. -/
def toyWipe (_ : Bool × Bool) : Bool × Bool := (false, false)

theorem toy_wipe_removes_objective_information (s t : Bool × Bool) :
    FutureEq toyStep toyObserve (toyWipe s) (toyWipe t) := by
  intro word
  rfl

/-- A single surviving artifact cannot exactly identify two distinct goals. -/
theorem same_artifact_cannot_decode_distinct_goals
    {Goal : Type u} {Artifact : Type v}
    (decode : Artifact → Goal) (artifact : Goal → Artifact)
    (g₁ g₂ : Goal) (distinct : g₁ ≠ g₂)
    (same : artifact g₁ = artifact g₂) :
    ¬ (decode (artifact g₁) = g₁ ∧ decode (artifact g₂) = g₂) := by
  intro h
  apply distinct
  calc
    g₁ = decode (artifact g₁) := h.1.symm
    _ = decode (artifact g₂) := congrArg decode same
    _ = g₂ := h.2

#print axioms futureEq_step
#print axioms adequate_code_separates
#print axioms toy_distinguished_by_future
#print axioms toy_wipe_removes_objective_information
#print axioms same_artifact_cannot_decode_distinct_goals

end ShaftCompact
