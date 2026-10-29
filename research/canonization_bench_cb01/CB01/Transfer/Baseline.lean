/-!
Three independent, source-specific proofs. Never imports Transfer.Canonical.
-/
import CB01.Transfer.Statements

set_option autoImplicit false
universe u v w z

namespace CB01.Transfer.Baseline

theorem menu {H : Type u} {W : Type v} {V : Type w} {O : Type z}
    (law : H → W → O) (restrict : V → W) (x y : H) :
    MenuTarget law restrict x y := by
  intro hEq word
  exact hEq (restrict word)

theorem code {H : Type u} {W : Type v} {C : Type w} {O : Type z}
    (law : H → W → O) (encode : H → C) (decode : C → W → O)
    (x y : H) : CodeTarget law encode decode x y := by
  intro correct witness same
  cases witness with
  | intro word different =>
      have left : law x word = decode (encode x) word :=
        (correct x word).symm
      have middle : decode (encode x) word = decode (encode y) word :=
        congrArg (fun c => decode c word) same
      have right : decode (encode y) word = law y word :=
        correct y word
      exact different (left.trans (middle.trans right))

theorem step (M : CB01.Machine) (x y : Nat) (action : Bool) :
    StepTarget M x y action := by
  intro hEq
  constructor
  · exact CB01.futureEq_event M x y action hEq
  · exact CB01.futureEq_step M x y action hEq

end CB01.Transfer.Baseline
