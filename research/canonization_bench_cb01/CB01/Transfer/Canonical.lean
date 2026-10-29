import CB01.Transfer.Statements
import CB01.Canonical

/-!
Canonical lemmas submitted before held-out tests are written.
-/

set_option autoImplicit false
universe u v w z

namespace CB01.Transfer.Canonical

theorem menu {H : Type u} {W : Type v} {V : Type w} {O : Type z}
    (law : H → W → O) (restrict : V → W) (x y : H) :
    MenuTarget law restrict x y := by
  intro hEq
  exact CB01.Canonical.precompose law restrict x y hEq

/-- Universal exact-code collision lemma, reusable outside the source contract. -/
theorem code_sound {H : Type u} {W : Type v} {C : Type w} {O : Type z}
    (law : H → W → O) (encode : H → C) (decode : C → W → O)
    (x y : H) (correct : CorrectCode law encode decode)
    (same : encode x = encode y) : TraceEq law x y := by
  intro word
  calc
    law x word = decode (encode x) word := (correct x word).symm
    _ = decode (encode y) word :=
        congrArg (fun c => decode c word) same
    _ = law y word := correct y word

theorem code {H : Type u} {W : Type v} {C : Type w} {O : Type z}
    (law : H → W → O) (encode : H → C) (decode : C → W → O)
    (x y : H) : CodeTarget law encode decode x y := by
  intro correct witness same
  cases witness with
  | intro word different =>
      exact different (code_sound law encode decode x y correct same word)

theorem step (M : CB01.Machine) (x y : Nat) (action : Bool) :
    StepTarget M x y action := by
  intro hEq
  exact ⟨CB01.futureEq_event M x y action hEq,
         CB01.futureEq_step M x y action hEq⟩

end CB01.Transfer.Canonical
