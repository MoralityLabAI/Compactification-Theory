import Shaft.BehaviorOnlyIdentification

/-!
SHAFT-COMP-24: exact XOR-based trigger identification when previous-token
labels are observed, and formal counterexamples with ungrounded labels.
The neural outcomes are separate empirical measurements.
-/
namespace ShaftCompact

def nativeTriggerTwo (effective : Fin 6 → Bool)
    (trigger : Fin 6 → Fin 6 → Bool) (prev first second : Fin 6) : Bool :=
  Bool.xor (effective second) (trigger prev first)

/-- B1 xor B2 is the trigger bit for the grounded previous token. -/
theorem xor_recovers_trigger (b : Fin 6 → Bool)
    (t : Fin 6 → Fin 6 → Bool) (p a c : Fin 6) :
    Bool.xor (b c) (nativeTriggerTwo b t p a c) = t p a := by
  cases hb : b c <;> cases ht : t p a <;> simp [nativeTriggerTwo, hb, ht]

/-- All observed previous-token classes identify every trigger entry. -/
theorem observed_previous_identifies_law (b : Fin 6 → Bool)
    (t u : Fin 6 → Fin 6 → Bool)
    (same : ∀ p a c, nativeTriggerTwo b t p a c = nativeTriggerTwo b u p a c) :
    t = u := by
  funext p a
  calc
    t p a = Bool.xor (b 0) (nativeTriggerTwo b t p a 0) :=
      (xor_recovers_trigger b t p a 0).symm
    _ = Bool.xor (b 0) (nativeTriggerTwo b u p a 0) := by rw [same p a 0]
    _ = u p a := xor_recovers_trigger b u p a 0

def shiftPrev (p : Fin 6) : Fin 6 := ⟨(p.val + 1) % 6, by omega⟩

def permutedTrigger (t : Fin 6 → Fin 6 → Bool) (p a : Fin 6) : Bool :=
  t (shiftPrev p) a

/-- Simultaneous row and latent-label renaming leaves behavior invariant. -/
theorem latent_row_relabel_gauge (b : Fin 6 → Bool)
    (t : Fin 6 → Fin 6 → Bool) (p a c : Fin 6) :
    nativeTriggerTwo b (permutedTrigger t) p a c =
      nativeTriggerTwo b t (shiftPrev p) a c := rfl

def identityTrigger (p a : Fin 6) : Bool := p == a

def swapFinalTriggers (p a : Fin 6) : Bool :=
  if p == 4 then a == 5 else if p == 5 then a == 4 else p == a

/-- Missing two of six labeled classes leaves two different assignments. -/
theorem two_missing_previous_classes_ambiguous :
    (∀ p : Fin 6, p.val < 4 → ∀ a : Fin 6,
      identityTrigger p a = swapFinalTriggers p a) ∧
    (∃ p a : Fin 6, identityTrigger p a ≠ swapFinalTriggers p a) := by
  decide

#print axioms xor_recovers_trigger
#print axioms observed_previous_identifies_law
#print axioms latent_row_relabel_gauge
#print axioms two_missing_previous_classes_ambiguous

end ShaftCompact
