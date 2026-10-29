/-!
Frozen canonical refactor. Holdout tests must import this without changing it.
-/
import CB01.Transfer.Canonical

set_option autoImplicit false
universe u v w z

namespace CB01.Transfer.Candidate

theorem menu {H : Type u} {W : Type v} {V : Type w} {O : Type z}
    (law : H → W → O) (restrict : V → W) (x y : H) :
    MenuTarget law restrict x y := by
  exact Canonical.menu law restrict x y

theorem code {H : Type u} {W : Type v} {C : Type w} {O : Type z}
    (law : H → W → O) (encode : H → C) (decode : C → W → O)
    (x y : H) : CodeTarget law encode decode x y := by
  exact Canonical.code law encode decode x y

theorem step (M : CB01.Machine) (x y : Nat) (action : Bool) :
    StepTarget M x y action := by
  exact Canonical.step M x y action

end CB01.Transfer.Candidate
