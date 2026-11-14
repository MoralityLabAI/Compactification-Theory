import Lean

theorem fc11_native_prop (P : Prop) (hp : P) : P := by
  first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor

theorem fc11_native_imp (P : Prop) : P → P := by
  first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor

theorem fc11_native_arith (a b : Nat) (hab : a ≤ b) : a + 3 ≤ b + 3 := by
  first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | aesop
  | assumption
  | intro
  | constructor

#print axioms fc11_native_prop
#print axioms fc11_native_imp
#print axioms fc11_native_arith
