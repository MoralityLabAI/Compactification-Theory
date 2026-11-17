import CB01.Canonical
set_option autoImplicit false

namespace CB01.Heldout

/-- Held-out client: translate a numerical length into a future action script. -/
theorem repeat_true (M : Machine) (x y : Nat) (h : FutureEq M x y) :
    SameFuture (fun state (n : Nat) => trace M state (List.replicate n true)) x y := by
  exact Canonical.precompose (trace M)
    (fun n : Nat => List.replicate n true) x y h

/-- Held-out client: expose only the number of timed events. -/
theorem trace_length (M : Machine) (x y : Nat) (h : FutureEq M x y) :
    SameFuture (fun state word => (trace M state word).length) x y := by
  exact Canonical.postcompose (trace M)
    (fun events : List TimedEvent => events.length) x y h

/-- Held-out client: retain both full trace and a summarized measurement. -/
theorem combined_report (M : Machine) (x y : Nat) (h : FutureEq M x y) :
    SameFuture (fun state word => (trace M state word, (trace M state word).length)) x y := by
  exact Canonical.pair (trace M)
    (fun state word => (trace M state word).length) x y h (trace_length M x y h)

end CB01.Heldout
