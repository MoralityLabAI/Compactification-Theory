import CB01.Core

/-!
CB02 statement-only contracts. These are deterministic/abstract specializations
of EvalMath IRC-02/T at pinned commit 2616092a...; they do not formalize its
rational stochastic posterior or conditional Bayesian update.
-/

set_option autoImplicit false
universe u v w z

namespace CB01.Transfer

/-- Full observable future-law equality under a shared continuation test. -/
def TraceEq {H : Type u} {W : Type v} {O : Type w}
    (law : H → W → O) (x y : H) : Prop :=
  CB01.SameFuture law x y

/-- An exact encoder/decoder reproduces each test's public law. -/
def CorrectCode {H : Type u} {W : Type v} {C : Type w} {O : Type z}
    (law : H → W → O) (encode : H → C) (decode : C → W → O) : Prop :=
  ∀ h word, decode (encode h) word = law h word

/-- IRC-02/MENU: restricting the common test menu preserves equality. -/
def MenuTarget {H : Type u} {W : Type v} {V : Type w} {O : Type z}
    (law : H → W → O) (restrict : V → W) (x y : H) : Prop :=
  TraceEq law x y →
    TraceEq (fun history word => law history (restrict word)) x y

/-- IRC-02/CODE: separated future laws require distinct exact codewords. -/
def CodeTarget {H : Type u} {W : Type v} {C : Type w} {O : Type z}
    (law : H → W → O) (encode : H → C) (decode : C → W → O)
    (x y : H) : Prop :=
  CorrectCode law encode decode →
    (∃ word, law x word ≠ law y word) →
    encode x ≠ encode y

/-- IRC-02/STEP: deterministic specialization of graded successor congruence. -/
def StepTarget (M : CB01.Machine) (x y : Nat) (action : Bool) : Prop :=
  CB01.FutureEq M x y →
    M.event x action = M.event y action ∧
      CB01.FutureEq M (M.step x action) (M.step y action)

end CB01.Transfer
