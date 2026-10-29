import CB01.Canonical
import CB01.BaselineA
import CB01.BaselineB
set_option autoImplicit false
universe u v w z

namespace CB01.Consumers

/-- Statement-identical to BaselineA.prefix_context, via the shared lemma. -/
theorem prefix_context {H : Type u} {A : Type v} {O : Type w}
    (observe : H → List A → O) (x y : H) (prefix : List A)
    (h : SameFuture observe x y) :
    SameFuture (fun history word => observe history (prefix ++ word)) x y := by
  exact Canonical.precompose observe (fun word : List A => prefix ++ word) x y h

/-- Statement-identical to BaselineB.translated_context, via the shared lemma. -/
theorem translated_context {H : Type u} {A : Type v} {B : Type w}
    {O : Type z} (observe : H → List A → O) (x y : H)
    (translate : List B → List A) (h : SameFuture observe x y) :
    SameFuture (fun history word => observe history (translate word)) x y := by
  exact Canonical.precompose observe translate x y h

end CB01.Consumers
