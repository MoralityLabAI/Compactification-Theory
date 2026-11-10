import Shaft.GaugeIdentifiability

/-!
SHAFT-COMP-09: deterministic robust-identification margin lemmas.

The premises explicitly bound scoring damage caused by missing observations
and corrupt bits. The theorems show sufficient conditions for a unique
minimum-score gauge. The probabilistic Chernoff/union bound is established
on paper separately and is not claimed to be checked by Lean.
-/

universe u
namespace ShaftCompact

/-- If every incorrect gauge has a clean disagreement margin greater
than all missing and twice all corrupted positions, the correct gauge
strictly wins under any score obeying the error-accounting premises. -/
theorem robust_gauge_unique
    {Gauge : Type u} (correct : Gauge) (score : Gauge → Nat)
    (minimumDistance erased corrupted : Nat)
    (correctUpper : score correct ≤ corrupted)
    (otherLower : ∀ g, g ≠ correct →
      minimumDistance ≤ score g + erased + corrupted)
    (margin : erased + 2 * corrupted < minimumDistance) :
    ∀ g, g ≠ correct → score correct < score g := by
  intro g different
  have h := otherLower g different
  have hc := correctUpper
  omega

/-- Using the minimum number of *observed* disagreement positions gives
a sharper sufficient condition than charging for every erasure. -/
theorem observed_distance_margin
    {Gauge : Type u} (correct : Gauge) (score : Gauge → Nat)
    (observedSeparation corrupted : Nat)
    (correctUpper : score correct ≤ corrupted)
    (otherLower : ∀ g, g ≠ correct →
      observedSeparation ≤ score g + corrupted)
    (margin : 2 * corrupted < observedSeparation) :
    ∀ g, g ≠ correct → score correct < score g := by
  intro g different
  have h := otherLower g different
  have hc := correctUpper
  omega

/-- Monotone information: a larger observed-separation margin retains
a uniqueness certificate that held at a smaller certified threshold. -/
theorem observed_distance_certificate_mono
    (small large corrupted : Nat)
    (hgap : 2 * corrupted < small)
    (hle : small ≤ large) :
    2 * corrupted < large :=
  Nat.lt_of_lt_of_le hgap hle

/-- A strict score winner is unique among all score minimizers. -/
theorem strict_winner_unique
    {Gauge : Type u} (correct : Gauge) (score : Gauge → Nat)
    (winner : ∀ g, g ≠ correct → score correct < score g)
    (candidate : Gauge)
    (minimum : ∀ g, score candidate ≤ score g) :
    candidate = correct := by
  by_contra h
  have strict := winner candidate h
  have atMost := minimum correct
  omega

/-- If every competing signature has at least d discrepancies and an
adversary can erase e and flip r observations, the bound d > e + 2r
is sufficient provided the mismatch-score accounting conditions hold. -/
theorem deterministic_missing_and_noisy_certificate
    {Gauge : Type u} (correct : Gauge) (score : Gauge → Nat)
    (d e r : Nat)
    (goodScore : score correct ≤ r)
    (badScore : ∀ g, g ≠ correct → d ≤ score g + e + r)
    (budget : e + 2 * r < d) (winner : Gauge)
    (winnerLeast : ∀ g, score winner ≤ score g) :
    winner = correct := by
  exact strict_winner_unique correct score
    (robust_gauge_unique correct score d e r goodScore badScore budget)
    winner winnerLeast

#print axioms robust_gauge_unique
#print axioms observed_distance_margin
#print axioms observed_distance_certificate_mono
#print axioms strict_winner_unique
#print axioms deterministic_missing_and_noisy_certificate

end ShaftCompact
