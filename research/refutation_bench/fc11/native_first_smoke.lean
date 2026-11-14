import Lean

theorem fc11_native_prop (P : Prop) (hp : P) : P := by
  first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip

theorem fc11_native_imp (P : Prop) : P → P := by
  first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip

theorem fc11_native_arith (a b : Nat) (hab : a ≤ b) : a + 3 ≤ b + 3 := by
  first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip
  all_goals first
  | omega
  | decide
  | rfl
  | simp_all
  | fail "aesop unavailable under source Lean4.35 Init imports"
  | assumption
  | intro
  | constructor <;> skip

#print axioms fc11_native_prop
#print axioms fc11_native_imp
#print axioms fc11_native_arith
