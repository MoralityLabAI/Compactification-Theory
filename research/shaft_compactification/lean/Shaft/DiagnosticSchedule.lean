import Shaft.FiniteObservability

/-!
SHAFT-COMP-13: diagnostic-observer equivalence can depend on which
continuations are permitted. This finite Boolean witness explains why a
rank measured on one diagnostic schedule need not control divergence on
an independently sampled future schedule.
-/
namespace ShaftCompact

/-- An inert diagnostic input preserves the entire state. An alternative
input makes the latent bit immediately visible at the first output. -/
def probeScheduleStep (a : Bool) (s : Bool × Bool) : Bool × Bool :=
  if a then (s.2,s.2) else s

def probeScheduleRead (s : Bool × Bool) : Bool := s.1
def probeStateA : Bool × Bool := (false,false)
def probeStateB : Bool × Bool := (false,true)

/-- Same present behavior and indistinguishable diagnostic history even
after eight inert steps. -/
theorem diagnostic_alias_at_eight :
    probeScheduleRead probeStateA = probeScheduleRead probeStateB ∧
    probeScheduleRead (run probeScheduleStep (List.replicate 8 false) probeStateA) =
      probeScheduleRead (run probeScheduleStep (List.replicate 8 false) probeStateB) := by
  decide

/-- A single different future input separates the two states. -/
theorem diagnostic_alias_future_split :
    probeScheduleRead (run probeScheduleStep [true] probeStateA) ≠
      probeScheduleRead (run probeScheduleStep [true] probeStateB) := by
  decide

/-- The monitored quotient for one restricted schedule cannot guarantee
the same observation under every unrestricted continuation. -/
theorem restricted_schedule_not_complete :
    ∃ a b : Bool × Bool,
      probeScheduleRead a = probeScheduleRead b ∧
      probeScheduleRead (run probeScheduleStep (List.replicate 8 false) a) =
        probeScheduleRead (run probeScheduleStep (List.replicate 8 false) b) ∧
      probeScheduleRead (run probeScheduleStep [true] a) ≠
        probeScheduleRead (run probeScheduleStep [true] b) := by
  exact ⟨probeStateA,probeStateB,diagnostic_alias_at_eight.1,
    diagnostic_alias_at_eight.2,diagnostic_alias_future_split⟩

#print axioms diagnostic_alias_at_eight
#print axioms diagnostic_alias_future_split
#print axioms restricted_schedule_not_complete
end ShaftCompact
