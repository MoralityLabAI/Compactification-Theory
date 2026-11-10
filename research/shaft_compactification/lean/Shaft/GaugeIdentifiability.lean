import Shaft.PredictiveCode

/-!
SHAFT-COMP-08: behavioral-gauge identifiability from paired anchors.
This concerns finite task signatures and explicit bijections, not evidence
that an unknown neural model has a recoverable or dangerous objective.
-/

universe u v w
namespace ShaftCompact

/-- Relabeling hidden goals preserves the set of unlabeled outputs. -/
theorem unlabeled_signatures_invariant
    {Goal : Type u} {Output : Type v}
    (source target : Goal → Output) (relabel : Goal → Goal)
    (onto : Function.Surjective relabel)
    (matching : ∀ g, target (relabel g) = source g)
    (o : Output) :
    (∃ g, source g = o) ↔ (∃ g, target g = o) := by
  constructor
  · rintro ⟨g, hg⟩
    exact ⟨relabel g, (matching g).trans hg⟩
  · rintro ⟨g, hg⟩
    obtain ⟨g', heq⟩ := onto g
    exact ⟨g', ((matching g').symm.trans (by simpa [heq] using hg))⟩

/-- Once anchor patterns uniquely label target coordinates, matching is unique. -/
theorem anchor_alignment_unique
    {Query : Type u} {Pattern : Type v}
    (source target : Query → Pattern)
    (injective : Function.Injective target)
    (first second : Query → Query)
    (firstMatch : ∀ q, source q = target (first q))
    (secondMatch : ∀ q, source q = target (second q)) :
    first = second := by
  funext q
  apply injective
  calc
    target (first q) = source q := (firstMatch q).symm
    _ = target (second q) := secondMatch q

/-- The three anchors 0001,0010,0100 assign coordinate 0,1,2 a
    distinct one-hot column and coordinate 3 an all-zero column. -/
def onehotPattern (query : Fin 4) (anchor : Fin 3) : Bool :=
  decide (query.val = anchor.val)

theorem onehotPattern_injective :
    ∀ q r : Fin 4,
      (∀ a : Fin 3, onehotPattern q a = onehotPattern r a) → q = r := by
  decide

/-- The column of query zero has one true entry, whereas the complement
    of any onehotPattern column has at least two true entries. -/
theorem onehotPattern_not_complemented :
    ∀ r : Fin 4,
      ¬ (∀ a : Fin 3, onehotPattern (0 : Fin 4) a = !(onehotPattern r a)) := by
  decide

/-- All three paired anchors fix an arbitrary query map and global bit
    flip uniquely; the theorem is stronger than requiring a permutation. -/
theorem three_anchor_gauge_unique
    (mapping : Fin 4 → Fin 4) (flip : Bool)
    (matching : ∀ q a,
      onehotPattern q a =
        (if flip then !(onehotPattern (mapping q) a)
         else onehotPattern (mapping q) a)) :
    flip = false ∧ (∀ q, mapping q = q) := by
  cases flip with
  | false =>
    constructor
    · rfl
    · intro q
      apply onehotPattern_injective q (mapping q)
      intro a
      simpa using matching q a
  | true =>
    have h : ∀ a : Fin 3,
        onehotPattern (0 : Fin 4) a =
          !(onehotPattern (mapping 0) a) := by
      intro a
      simpa using matching 0 a
    exact False.elim ((onehotPattern_not_complemented (mapping 0)) h)

/-- With only anchors 0001 and 0010, query indices 2 and 3
    are identical to the observer. -/
def twoAnchorPattern (q : Fin 4) (a : Fin 2) : Bool :=
  decide (q.val = a.val)

theorem two_anchor_alias :
    (∀ a : Fin 2,
       twoAnchorPattern (2 : Fin 4) a = twoAnchorPattern (3 : Fin 4) a)
       ∧ (2 : Fin 4) ≠ (3 : Fin 4) := by
  decide

#print axioms unlabeled_signatures_invariant
#print axioms anchor_alignment_unique
#print axioms onehotPattern_injective
#print axioms onehotPattern_not_complemented
#print axioms three_anchor_gauge_unique
#print axioms two_anchor_alias

end ShaftCompact
