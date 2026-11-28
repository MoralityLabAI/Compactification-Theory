import Shaft.TaskFactorial

/-!
SHAFT-COMP-17: an immediate behavioral signature can alias different
temporal tasks, while two-step counterfactual outputs disambiguate them.
These are exact finite Boolean facts, not neural performance claims.
-/
namespace ShaftCompact

def temporalXor (a b : Bool) : Bool := decide (a != b)
def goalIdentity (q : Bool) : Bool := q
def goalComplement (q : Bool) : Bool := !q

def temporalOne (g : Bool → Bool) (prev q : Bool) : Bool :=
  g (temporalXor prev q)
def temporalTwo (g : Bool → Bool) (_prev q1 q2 : Bool) : Bool :=
  g (temporalXor q1 q2)

theorem same_immediate_signature :
    ∀ q : Bool, temporalOne goalIdentity false q =
      temporalOne goalComplement true q := by
  decide

theorem separating_two_step_query :
    temporalTwo goalIdentity false false false ≠
      temporalTwo goalComplement true false false := by
  decide

theorem two_step_independent_of_older_query
    (g : Bool → Bool) (a b q1 q2 : Bool) :
    temporalTwo g a q1 q2 = temporalTwo g b q1 q2 := by
  rfl

theorem temporal_alias_broken_by_two_step :
    (∀ q : Bool, temporalOne goalIdentity false q =
      temporalOne goalComplement true q) ∧
    temporalTwo goalIdentity false false false ≠
      temporalTwo goalComplement true false false := by
  exact ⟨same_immediate_signature, separating_two_step_query⟩

#print axioms same_immediate_signature
#print axioms separating_two_step_query
#print axioms two_step_independent_of_older_query
#print axioms temporal_alias_broken_by_two_step

end ShaftCompact
