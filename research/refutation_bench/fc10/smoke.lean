import Lean

set_option autoImplicit false

theorem fc10_smoke_omega (a b : Nat) (hab : a ≤ b) :
    a + 4 ≤ b + 4 := by
  trace "FC10_BEGIN:smoke_omega"
  trace_state
  trace "FC10_END:smoke_omega"
  omega

theorem fc10_smoke_and (P Q : Prop) (hp : P) (hq : Q) :
    P ∧ Q := by
  trace "FC10_BEGIN:smoke_and"
  trace_state
  trace "FC10_END:smoke_and"
  constructor <;> assumption

theorem fc10_smoke_list (xs ys zs : List Nat) :
    xs ++ (ys ++ zs) = (xs ++ ys) ++ zs := by
  trace "FC10_BEGIN:smoke_list"
  trace_state
  trace "FC10_END:smoke_list"
  simp [List.append_assoc]

#print axioms fc10_smoke_omega
#print axioms fc10_smoke_and
#print axioms fc10_smoke_list
