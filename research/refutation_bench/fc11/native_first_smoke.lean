import Lean

macro "fc11_ctor" : tactic => `(tactic| constructor)

theorem fc11_native_prop (P : Prop) (hp : P) : P := by
  first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro
  all_goals (first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro)
  all_goals (first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro)

theorem fc11_native_imp (P : Prop) : P → P := by
  first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro
  all_goals (first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro)
  all_goals (first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro)

theorem fc11_native_arith (a b : Nat) (hab : a ≤ b) : a + 3 ≤ b + 3 := by
  first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro
  all_goals (first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro)
  all_goals (first | omega | decide | rfl | simp_all | fail "aesop unavailable under MathCan Lean imports" | assumption | fc11_ctor | intro)

#print axioms fc11_native_prop
#print axioms fc11_native_imp
#print axioms fc11_native_arith
