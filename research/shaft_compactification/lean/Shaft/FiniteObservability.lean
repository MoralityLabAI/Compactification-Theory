import Shaft.CausalSufficiency

/-!
SHAFT-COMP-12: abstract finite-horizon observability and exact six-bit delay.
The numerical matrix rank and Gramian eigenvalue results are checked by a
separate finite integer Python reference and a trained CPU GRU experiment.
This dependency-light Lean module proves exact horizon/refinement statements
and the equivalence of zero nonnegative cumulative observation cost with
zero individual costs. It does not claim to formalize real spectral theory.
-/

universe u v
namespace ShaftCompact

/-- Evolve an arbitrary time-dependent transition system along a fixed
    prescribed input schedule. step n is the n-th local transition. -/
def cocycle {State : Type u}
    (step : Nat → State → State) : Nat → State → State
  | 0, s => s
  | n + 1, s => step n (cocycle step n s)

/-- State s is invisible to the monitor through observation horizon L. -/
def InvisibleUntil {State : Type u} {Output : Type v}
    (step : Nat → State → State) (read : State → Output)
    (zero : Output) (L : Nat) (s : State) : Prop :=
  ∀ k, k ≤ L → read (cocycle step k s) = zero

/-- Additional future observations can only shrink the invisible set. -/
theorem invisible_horizon_mono {State : Type u} {Output : Type v}
    (step : Nat → State → State) (read : State → Output)
    (zero : Output) (short long : Nat) (s : State)
    (hshort : short ≤ long)
    (hinv : InvisibleUntil step read zero long s) :
    InvisibleUntil step read zero short s := by
  intro k hk
  exact hinv k (Nat.le_trans hk hshort)

/-- The new constraint at time L+1 is the only new constraint. -/
theorem invisible_horizon_succ {State : Type u} {Output : Type v}
    (step : Nat → State → State) (read : State → Output)
    (zero : Output) (L : Nat) (s : State) :
    InvisibleUntil step read zero (L + 1) s ↔
      InvisibleUntil step read zero L s ∧
        read (cocycle step (L + 1) s) = zero := by
  constructor
  · intro h
    constructor
    · exact invisible_horizon_mono step read zero L (L+1) s (by omega) h
    · exact h (L+1) (Nat.le_refl _)
  · rintro ⟨hprior, hnext⟩
    intro k hk
    by_cases hkprior : k ≤ L
    · exact hprior k hkprior
    · have heq : k = L + 1 := by omega
      simpa [heq] using hnext

/-- A newly exposed direction witnesses strict shrinkage at horizon L+1. -/
theorem invisible_strict_witness {State : Type u} {Output : Type v}
    (step : Nat → State → State) (read : State → Output)
    (zero : Output) (L : Nat) (s : State)
    (_initial : InvisibleUntil step read zero L s)
    (exposed : read (cocycle step (L+1) s) ≠ zero) :
    ¬ InvisibleUntil step read zero (L+1) s := by
  intro hall
  exact exposed (hall (L+1) (Nat.le_refl _))

/-- Finite-horizon cost accumulates nonnegative natural-number penalties.
    A squared norm over integer traces is a specialization. -/
def cumulativeCost {State : Type u}
    (cost : Nat → State → Nat) : Nat → State → Nat
  | 0, s => cost 0 s
  | n + 1, s => cumulativeCost cost n s + cost (n + 1) s

/-- Sum of nonnegative costs vanishes iff every measured cost vanishes. -/
theorem cumulativeCost_zero_iff {State : Type u}
    (cost : Nat → State → Nat) (L : Nat) (s : State) :
    cumulativeCost cost L s = 0 ↔
      ∀ k, k ≤ L → cost k s = 0 := by
  induction L with
  | zero =>
      constructor
      · intro h k hk
        have heq : k = 0 := by omega
        subst k
        simpa [cumulativeCost] using h
      · intro h
        simpa [cumulativeCost] using h 0 (Nat.le_refl 0)
  | succ L ih =>
      constructor
      · intro h k hk
        have hsum : cumulativeCost cost L s + cost (L+1) s = 0 := by
          simpa [cumulativeCost] using h
        have hl : cumulativeCost cost L s = 0 := by omega
        have hr : cost (L+1) s = 0 := by omega
        by_cases hkprior : k ≤ L
        · exact (ih.mp hl) k hkprior
        · have heq : k = L + 1 := by omega
          simpa [heq] using hr
      · intro hall
        have hl : cumulativeCost cost L s = 0 := ih.mpr (by
          intro k hk
          exact hall k (by omega))
        have hr : cost (L+1) s = 0 := hall (L+1) (Nat.le_refl _)
        simp [cumulativeCost, hl, hr]

/-- Concrete six-position shift register with one-bit observation. -/
def sixBitShift (s : Fin 6 → Bool) (i : Fin 6) : Bool :=
  if hi : i.val + 1 < 6 then s ⟨i.val + 1, hi⟩ else false

def sixWitness : Fin 6 → Bool := fun i => i == (5 : Fin 6)
def sixZero : Fin 6 → Bool := fun _ => false
def sixObserve (s : Fin 6 → Bool) : Bool := s 0

def sixIter : Nat → (Fin 6 → Bool) → (Fin 6 → Bool)
  | 0, s => s
  | n + 1, s => sixBitShift (sixIter n s)

/-- Five-step delayed difference is invisible at horizons 0 through 4. -/
theorem six_shift_invisible_through_four :
    ∀ k : Fin 5, sixObserve (sixIter k.val sixZero) =
      sixObserve (sixIter k.val sixWitness) := by
  decide

/-- At horizon five, the otherwise hidden bit becomes observable. -/
theorem six_shift_exposed_at_five :
    sixObserve (sixIter 5 sixZero) ≠
      sixObserve (sixIter 5 sixWitness) := by
  decide

#print axioms invisible_horizon_mono
#print axioms invisible_horizon_succ
#print axioms invisible_strict_witness
#print axioms cumulativeCost_zero_iff
#print axioms six_shift_invisible_through_four
#print axioms six_shift_exposed_at_five

end ShaftCompact
