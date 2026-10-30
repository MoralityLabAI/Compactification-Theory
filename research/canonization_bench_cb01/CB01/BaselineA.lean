import CB01.Core
set_option autoImplicit false
universe u v w

namespace CB01.BaselineA

/-- Independent proof A for a fixed future prefix. -/
theorem prefix_context {H : Type u} {A : Type v} {O : Type w}
    (observe : H → List A → O) (x y : H) (prefixWords : List A)
    (h : SameFuture observe x y) :
    SameFuture (fun history word => observe history (prefixWords ++ word)) x y := by
  intro word
  exact h (prefixWords ++ word)

end CB01.BaselineA
