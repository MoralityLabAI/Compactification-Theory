import CB01.Core
set_option autoImplicit false
universe u v w z

namespace CB01.BaselineB

/-- Independent proof B for an arbitrary action-word translation. -/
theorem translated_context {H : Type u} {A : Type v} {B : Type w}
    {O : Type z} (observe : H → List A → O) (x y : H)
    (translate : List B → List A) (h : SameFuture observe x y) :
    SameFuture (fun history word => observe history (translate word)) x y := by
  intro word
  exact h (translate word)

end CB01.BaselineB
