/-!
The identical declared Prop targets force semantic type alignment between
baseline and candidate. Lean elaborates both proofs independently.
-/
import CB01.Transfer.Baseline
import CB01.Transfer.Candidate

set_option autoImplicit false
universe u v w z

namespace CB01.Transfer.Audit

theorem compare_menu {H : Type u} {W : Type v} {V : Type w} {O : Type z}
    (law : H → W → O) (scope : V → W) (x y : H) :
    MenuTarget law scope x y :=
  Baseline.menu law scope x y

theorem compare_menu_candidate {H : Type u} {W : Type v} {V : Type w} {O : Type z}
    (law : H → W → O) (scope : V → W) (x y : H) :
    MenuTarget law scope x y :=
  Candidate.menu law scope x y

theorem compare_code {H : Type u} {W : Type v} {C : Type w} {O : Type z}
    (law : H → W → O) (encode : H → C) (decode : C → W → O)
    (x y : H) : CodeTarget law encode decode x y :=
  Baseline.code law encode decode x y

theorem compare_code_candidate {H : Type u} {W : Type v} {C : Type w} {O : Type z}
    (law : H → W → O) (encode : H → C) (decode : C → W → O)
    (x y : H) : CodeTarget law encode decode x y :=
  Candidate.code law encode decode x y

theorem compare_step (M : CB01.Machine) (x y : Nat) (action : Bool) :
    StepTarget M x y action :=
  Baseline.step M x y action

theorem compare_step_candidate (M : CB01.Machine) (x y : Nat) (action : Bool) :
    StepTarget M x y action :=
  Candidate.step M x y action

end CB01.Transfer.Audit
