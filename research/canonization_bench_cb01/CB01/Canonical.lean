import CB01.Core
set_option autoImplicit false
universe u v w z

namespace CB01.Canonical

/-- One abstraction subsumes the two independent context proofs. -/
theorem precompose {H : Type u} {W : Type v} {V : Type w} {O : Type z}
    (observe : H → W → O) (change : V → W) (x y : H)
    (h : SameFuture observe x y) :
    SameFuture (fun history word => observe history (change word)) x y := by
  intro word
  exact h (change word)

/-- Observable post-processing preserves future indistinguishability. -/
theorem postcompose {H : Type u} {W : Type v} {O : Type w} {P : Type z}
    (observe : H → W → O) (summarize : O → P) (x y : H)
    (h : SameFuture observe x y) :
    SameFuture (fun history word => summarize (observe history word)) x y := by
  intro word
  exact congrArg summarize (h word)

/-- Independent output channels can be combined when both agree. -/
theorem pair {H : Type u} {W : Type v} {O : Type w} {P : Type z}
    (first : H → W → O) (second : H → W → P) (x y : H)
    (hFirst : SameFuture first x y) (hSecond : SameFuture second x y) :
    SameFuture (fun history word => (first history word, second history word)) x y := by
  intro word
  change (first x word, second x word) = (first y word, second y word)
  rw [hFirst word, hSecond word]

end CB01.Canonical
