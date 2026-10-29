import CB01.Transfer.Canonical

/-!
CB02 held-out consumers. Authored after frozen development CI succeeded at
e698c55125107577155fc5342870e56153a0b71b. No baseline or candidate import.
The unchanged canonical library must solve these new client propositions.
-/
set_option autoImplicit false
universe u v w z t

namespace CB01.Transfer.Heldout

/-- HOLDOUT-01: restrict an adaptive test grammar within a fixed public stratum. -/
theorem public_stratum_scope {Q : Type u} {H : Type v} {W : Type w} {O : Type z}
    (law : Q → Nat → H → W → O) (q : Q) (deadline : Nat)
    (x y : H) (translate : Nat → W)
    (h : TraceEq (law q deadline) x y) :
    TraceEq (fun history (n : Nat) =>
      law q deadline history (translate n)) x y := by
  exact Canonical.menu (law q deadline) translate x y h

/-- HOLDOUT-02: an exact decoder also determines any summarized future law. -/
theorem summarized_law_from_code {H : Type u} {W : Type v} {C : Type w}
    {O : Type z} {P : Type t}
    (law : H → W → O) (encode : H → C) (decode : C → W → O)
    (summarize : O → P) (x y : H)
    (correct : CorrectCode law encode decode) (same : encode x = encode y) :
    TraceEq (fun history word => summarize (law history word)) x y := by
  have h : TraceEq law x y :=
    Canonical.code_sound law encode decode x y correct same
  exact CB01.Canonical.postcompose law summarize x y h

/-- HOLDOUT-03: jointly preserve two public feedback channels after menu restriction. -/
theorem two_channels_after_scope {H : Type u} {W : Type v} {V : Type w}
    {O : Type z} {P : Type t}
    (first : H → W → O) (second : H → W → P) (scope : V → W)
    (x y : H) (hFirst : TraceEq first x y)
    (hSecond : TraceEq second x y) :
    TraceEq (fun history word =>
      (first history (scope word), second history (scope word))) x y := by
  have a : TraceEq (fun history word => first history (scope word)) x y :=
    Canonical.menu first scope x y hFirst
  have b : TraceEq (fun history word => second history (scope word)) x y :=
    Canonical.menu second scope x y hSecond
  exact CB01.Canonical.pair
    (fun history word => first history (scope word))
    (fun history word => second history (scope word)) x y a b

/-- HOLDOUT-04: same exact code implies equal next event and successor trace. -/
theorem code_determines_timed_successor {C : Type u}
    (M : CB01.Machine) (encode : Nat → C)
    (decode : C → List Bool → List CB01.TimedEvent)
    (x y : Nat) (action : Bool)
    (correct : CorrectCode (CB01.trace M) encode decode)
    (same : encode x = encode y) :
    M.event x action = M.event y action ∧
      CB01.FutureEq M (M.step x action) (M.step y action) := by
  have h : CB01.FutureEq M x y :=
    Canonical.code_sound (CB01.trace M) encode decode x y correct same
  exact Canonical.step M x y action h

end CB01.Transfer.Heldout
