import Std

/- COG-07: the finite certificate's logical meaning.
   Candidate hypotheses and examples may have arbitrary types. The executable
   checker enumerates the finite hypothesis class and verifies `covers`.
   These lemmas establish the connection to consistency and decoder output. -/
namespace COG07

def Consistent {H E : Type} (allowed : H → Prop) (satisfies : H → E → Prop)
    (sample : List E) (h : H) : Prop :=
  allowed h ∧ ∀ e, e ∈ sample → satisfies h e

def Excluded {H E : Type} (satisfies : H → E → Prop)
    (sample : List E) (h : H) : Prop :=
  ∃ e, e ∈ sample ∧ ¬ satisfies h e

theorem cover_preserves_consistency {H E : Type}
    (allowed : H → Prop) (satisfies : H → E → Prop)
    (full small : List E)
    (subset : ∀ e, e ∈ small → e ∈ full)
    (covers : ∀ h, allowed h → Excluded satisfies full h → Excluded satisfies small h)
    (h : H) :
    Consistent allowed satisfies full h ↔ Consistent allowed satisfies small h := by
  constructor
  · intro hh
    exact ⟨hh.1, fun e he => hh.2 e (subset e he)⟩
  · intro hh
    refine ⟨hh.1, ?_⟩
    intro e he
    apply Classical.byContradiction
    intro bad
    obtain ⟨w, hw, failed⟩ := covers h hh.1 ⟨e, he, bad⟩
    exact failed (hh.2 w hw)

theorem cover_preserves_version_space {H E : Type}
    (allowed : H → Prop) (satisfies : H → E → Prop)
    (full small : List E)
    (subset : ∀ e, e ∈ small → e ∈ full)
    (covers : ∀ h, allowed h → Excluded satisfies full h → Excluded satisfies small h) :
    Consistent allowed satisfies full = Consistent allowed satisfies small := by
  funext h
  exact propext (cover_preserves_consistency allowed satisfies full small subset covers h)

theorem fixed_decoder_is_preserved {H E : Type}
    (allowed : H → Prop) (satisfies : H → E → Prop)
    (full small : List E)
    (subset : ∀ e, e ∈ small → e ∈ full)
    (covers : ∀ h, allowed h → Excluded satisfies full h → Excluded satisfies small h)
    (decoder : (H → Prop) → H) :
    decoder (Consistent allowed satisfies full) = decoder (Consistent allowed satisfies small) := by
  rw [cover_preserves_version_space allowed satisfies full small subset covers]

#print axioms cover_preserves_consistency
#print axioms cover_preserves_version_space
#print axioms fixed_decoder_is_preserved

end COG07
