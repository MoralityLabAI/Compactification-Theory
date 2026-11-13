import Shaft.TaskFactorial

/-!
SHAFT-COMP-16: exact finite witness for history-dependent task semantics.
An identifier for the goal and current query can be inadequate without
additional context about the previous query. No probabilistic or neural
generalization claim is formalized in this theorem.
-/
namespace ShaftCompact

/-- A one-bit analogue of lag_match: the output depends on the current
query, on a goal bit, and on whether the previous query matches. -/
def temporalMatchOutput (goal current previous : Bool) : Bool :=
  Bool.xor goal (current == previous)

/-- The same goal and query yield different native answers across histories. -/
theorem temporal_history_changes_reply :
    temporalMatchOutput false false false ≠
      temporalMatchOutput false false true := by
  decide

/-- A code storing only the goal cannot decode all possible histories. -/
theorem no_goal_only_temporal_decoder (decode : Bool → Bool → Bool) :
    ¬ (∀ goal current previous, decode goal current =
      temporalMatchOutput goal current previous) := by
  intro h
  have h0 := h false false false
  have h1 := h false false true
  exact temporal_history_changes_reply (h0.symm.trans h1)

/-- Retaining the preceding query as part of a state code is sufficient
to evaluate the next binary reply for this finite example. -/
theorem goal_and_history_determine_reply
    (goal current previous : Bool) :
    temporalMatchOutput goal current previous =
      Bool.xor goal (current == previous) := by
  rfl

#print axioms temporal_history_changes_reply
#print axioms no_goal_only_temporal_decoder
#print axioms goal_and_history_determine_reply

end ShaftCompact
