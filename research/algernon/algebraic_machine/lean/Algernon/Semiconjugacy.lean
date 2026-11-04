import Init

/-!
Algernon machine-algebra laws, Lean 4 without Mathlib.
The semantic transition is an arbitrary function; no learned behavior is claimed.
The kernel checks these results when this file is compiled.
-/

universe u v w
namespace AlgernonAlgebra

/-- Repeated execution of a deterministic step. -/
def iterate {α : Type u} (f : α → α) : Nat → α → α
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- If a decoder commutes with one transition, it commutes with every iterate. -/
theorem semiconjugacy_iterate {H : Type u} {S : Type v}
    (F : H → H) (T : S → S) (π : H → S)
    (commutes : ∀ h, π (F h) = T (π h))
    (n : Nat) (h : H) :
    π (iterate F n h) = iterate T n (π h) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    calc
      π (iterate F (n + 1) h) = π (F (iterate F n h)) := rfl
      _ = T (π (iterate F n h)) := commutes _
      _ = T (iterate T n (π h)) := congrArg T ih
      _ = iterate T (n + 1) (π h) := rfl

/-- Semantic equality is a congruence under a commuting transition. -/
theorem quotient_congruence {H : Type u} {S : Type v}
    (F : H → H) (T : S → S) (π : H → S)
    (commutes : ∀ h, π (F h) = T (π h))
    (h₁ h₂ : H) (eq : π h₁ = π h₂) :
    π (F h₁) = π (F h₂) := by
  calc
    π (F h₁) = T (π h₁) := commutes h₁
    _ = T (π h₂) := congrArg T eq
    _ = π (F h₂) := (commutes h₂).symm

/-- A latent periodic return implies a semantic return at the same number of steps. -/
theorem latent_period_projects {H : Type u} {S : Type v}
    (F : H → H) (T : S → S) (π : H → S)
    (commutes : ∀ h, π (F h) = T (π h))
    (h : H) (m : Nat) (periodic : iterate F m h = h) :
    iterate T m (π h) = π h := by
  calc
    iterate T m (π h) = π (iterate F m h) :=
      (semiconjugacy_iterate F T π commutes m h).symm
    _ = π h := congrArg π periodic

/-- A finite instruction word is executed left-to-right. -/
def execute {Instr : Type w} {H : Type u}
    (act : Instr → H → H) : List Instr → H → H
  | [], h => h
  | a :: rest, h => execute act rest (act a h)

/-- Generator-level semiconjugacy extends to *every instruction word*. -/
theorem instruction_word_semiconjugacy {Instr : Type w} {H : Type u} {S : Type v}
    (F : Instr → H → H) (T : Instr → S → S) (π : H → S)
    (commutes : ∀ a h, π (F a h) = T a (π h))
    (word : List Instr) (h : H) :
    π (execute F word h) = execute T word (π h) := by
  induction word generalizing h with
  | nil => rfl
  | cons a rest ih =>
    calc
      π (execute F (a :: rest) h) = π (execute F rest (F a h)) := rfl
      _ = execute T rest (π (F a h)) := ih (F a h)
      _ = execute T rest (T a (π h)) := congrArg (execute T rest) (commutes a h)
      _ = execute T (a :: rest) (π h) := rfl

/-- The domain of the local algebra must be invariant to apply local lemmas repeatedly. -/
theorem invariant_iterate {H : Type u}
    (F : H → H) (Valid : H → Prop)
    (closed : ∀ h, Valid h → Valid (F h))
    (n : Nat) (h : H) (hv : Valid h) :
    Valid (iterate F n h) := by
  induction n with
  | zero => exact hv
  | succ n ih => exact closed _ ih

/-- Local correctness on a forward-invariant set implies correctness at every depth. -/
theorem invariant_semiconjugacy {H : Type u} {S : Type v}
    (F : H → H) (T : S → S) (π : H → S)
    (Valid : H → Prop)
    (closed : ∀ h, Valid h → Valid (F h))
    (commutes : ∀ h, Valid h → π (F h) = T (π h))
    (n : Nat) (h : H) (hv : Valid h) :
    π (iterate F n h) = iterate T n (π h) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    calc
      π (iterate F (n + 1) h) = π (F (iterate F n h)) := rfl
      _ = T (π (iterate F n h)) :=
        commutes _ (invariant_iterate F Valid closed n h hv)
      _ = T (iterate T n (π h)) := congrArg T ih
      _ = iterate T (n + 1) (π h) := rfl

#print axioms semiconjugacy_iterate
#print axioms instruction_word_semiconjugacy
#print axioms latent_period_projects
#print axioms invariant_semiconjugacy

end AlgernonAlgebra
