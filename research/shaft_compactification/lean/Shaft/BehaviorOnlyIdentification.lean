import Shaft.FactorizedLatch

/-!
SHAFT-COMP-23. Exact two-query behavior identifies the effective-table
and previous-token code without canonical goal/latch supervision. The
constructive inverse votes across six first-query rows; exactly one row
is complemented by the latch trigger. All lemmas concern finite behavior.
-/
namespace ShaftCompact

def countSix (f : Fin 6 → Bool) : Nat :=
  (if f 0 then 1 else 0) + (if f 1 then 1 else 0) +
  (if f 2 then 1 else 0) + (if f 3 then 1 else 0) +
  (if f 4 then 1 else 0) + (if f 5 then 1 else 0)

def majoritySix (f : Fin 6 → Bool) : Bool := decide (4 ≤ countSix f)

def firstComplementRow (f : Fin 6 → Bool) (b : Bool) : Fin 6 :=
  if f 0 != b then 0 else if f 1 != b then 1 else
  if f 2 != b then 2 else if f 3 != b then 3 else
  if f 4 != b then 4 else 5

def predecessorSix (q : Fin 6) : Fin 6 := ⟨(q.val+5)%6, by omega⟩

def previousFromColumn (family : Bool) (f : Fin 6 → Bool) : Fin 6 :=
  let bit := majoritySix f
  let row := firstComplementRow f bit
  if family then predecessorSix row else row

theorem majority_recovers_effective_bit :
    ∀ family : Bool, ∀ p : Fin 6, ∀ bit : Bool,
      majoritySix (fun a => Bool.xor bit (latchTrigger family p a)) = bit := by
  decide

theorem column_recovers_previous :
    ∀ family : Bool, ∀ p : Fin 6, ∀ bit : Bool,
      previousFromColumn family
        (fun a => Bool.xor bit (latchTrigger family p a)) = p := by
  decide

def codeFromTwo (family : Bool) (signature : Fin 6 → Fin 6 → Bool) :
    BehavioralLatchCode :=
  (fun q => majoritySix (fun a => signature a q),
   previousFromColumn family (fun a => signature a 0))

theorem codeFromTwo_exact (family : Bool) (code : BehavioralLatchCode) :
    codeFromTwo family (fun a b => latchTwoFromCode family code a b) = code := by
  apply Prod.ext
  · funext q
    change majoritySix
      (fun a => Bool.xor (code.1 q) (latchTrigger family code.2 a)) = code.1 q
    exact majority_recovers_effective_bit family code.2 (code.1 q)
  · change previousFromColumn family
      (fun a => Bool.xor (code.1 0) (latchTrigger family code.2 a)) = code.2
    exact column_recovers_previous family code.2 (code.1 0)

theorem twoQuery_injective (family : Bool) (a b : BehavioralLatchCode)
    (eqTwo : (fun q r => latchTwoFromCode family a q r) =
      (fun q r => latchTwoFromCode family b q r)) : a = b := by
  have ha := codeFromTwo_exact family a
  have hb := codeFromTwo_exact family b
  calc
    a = codeFromTwo family (fun q r => latchTwoFromCode family a q r) := ha.symm
    _ = codeFromTwo family (fun q r => latchTwoFromCode family b q r) := by rw [eqTwo]
    _ = b := hb

#print axioms majority_recovers_effective_bit
#print axioms column_recovers_previous
#print axioms codeFromTwo_exact
#print axioms twoQuery_injective
end ShaftCompact
