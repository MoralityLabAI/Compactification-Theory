import Shaft.BehaviorOnlyIdentification

/-!
SHAFT-COMP-24: With the previous input and the one-query response available,
a single two-query response identifies one bit of an otherwise unknown latch
trigger. All rows follow by repeating this observation for p and a.
No neural or probabilistic claim is made by this finite theorem.
-/
namespace ShaftCompact

def xorBit (a b : Bool) : Bool := a != b

def nativeSecond (b : Fin 6 → Bool)
    (rule : Fin 6 → Fin 6 → Bool) (p a c : Fin 6) : Bool :=
  xorBit (b c) (rule p a)

theorem xorBit_cancel (b t : Bool) : xorBit (xorBit b t) b = t := by
  cases b <;> cases t <;> rfl

theorem identify_rule_bit (b : Fin 6 → Bool)
    (rule : Fin 6 → Fin 6 → Bool) (p a c : Fin 6) :
    xorBit (nativeSecond b rule p a c) (b c) = rule p a := by
  exact xorBit_cancel (b c) (rule p a)

theorem unique_rule_from_behavior (b : Fin 6 → Bool)
    (t1 t2 : Fin 6 → Fin 6 → Bool)
    (same : ∀ p a, nativeSecond b t1 p a 0 = nativeSecond b t2 p a 0) :
    t1 = t2 := by
  funext p a
  have h := congrArg (fun output => xorBit output (b 0)) (same p a)
  simpa only [identify_rule_bit] using h

#print axioms xorBit_cancel
#print axioms identify_rule_bit
#print axioms unique_rule_from_behavior
end ShaftCompact
