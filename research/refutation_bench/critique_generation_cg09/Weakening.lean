import Init

namespace CG09Weakening

/-- A proof in a weaker context transports to a stronger context. -/
theorem transport {World : Type} (base weak strong goal : World → Prop)
    (hsub : ∀ w, strong w → weak w)
    (h : ∀ w, base w → weak w → goal w) :
    ∀ w, base w → strong w → goal w :=
  fun w hb hs => h w hb (hsub w hs)

/-- A model of the stronger context also witnesses failure in the weaker one. -/
theorem countermodel_to_weaker {World : Type}
    (base weak strong goal : World → Prop)
    (hsub : ∀ w, strong w → weak w)
    (h : ∃ w, base w ∧ strong w ∧ ¬ goal w) :
    ∃ w, base w ∧ weak w ∧ ¬ goal w := by
  obtain ⟨w, hb, hs, hn⟩ := h
  exact ⟨w, hb, hsub w hs, hn⟩

/-- An independently proved global conclusion supplies the final local bridge. -/
theorem global_supplies_bridge {World : Type}
    (base localContext goal : World → Prop)
    (h : ∀ w, base w → goal w) :
    ∀ w, base w → localContext w → goal w :=
  fun w hb _ => h w hb

/-- The final semantic obligation cannot have a countermodel if the goal
    already follows globally. A missing named derivation is a different issue. -/
theorem global_excludes_final_countermodel {World : Type}
    (base localContext goal : World → Prop)
    (h : ∀ w, base w → goal w) :
    ¬ ∃ w, base w ∧ localContext w ∧ ¬ goal w := by
  intro hex
  obtain ⟨w, hb, _, hn⟩ := hex
  exact hn (h w hb)

#print axioms transport
#print axioms countermodel_to_weaker
#print axioms global_supplies_bridge
#print axioms global_excludes_final_countermodel

end CG09Weakening
