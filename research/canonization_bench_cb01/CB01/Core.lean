/-!
CB01 common statement: two histories agree on every declared future experiment.
A future experiment may return a timed trace, a rational law, or any other output.
The small `Machine` below is deliberately deterministic; it is not IRC-02's
stochastic oracle, which remains owned by EvalMath.
-/
set_option autoImplicit false

universe u v w

namespace CB01

def SameFuture {H : Type u} {W : Type v} {O : Type w}
    (observe : H → W → O) (x y : H) : Prop :=
  ∀ word : W, observe x word = observe y word

theorem sameFuture_refl {H : Type u} {W : Type v} {O : Type w}
    (observe : H → W → O) (x : H) : SameFuture observe x x := by
  intro _
  rfl

theorem sameFuture_symm {H : Type u} {W : Type v} {O : Type w}
    (observe : H → W → O) (x y : H) (h : SameFuture observe x y) :
    SameFuture observe y x := by
  intro word
  exact (h word).symm

theorem sameFuture_trans {H : Type u} {W : Type v} {O : Type w}
    (observe : H → W → O) (x y z : H)
    (hxy : SameFuture observe x y) (hyz : SameFuture observe y z) :
    SameFuture observe x z := by
  intro word
  exact (hxy word).trans (hyz word)

structure TimedEvent where
  signal : Nat
  ticks : Nat
  deriving DecidableEq, Repr

structure Machine where
  event : Nat → Bool → TimedEvent
  step : Nat → Bool → Nat

def trace (M : Machine) : Nat → List Bool → List TimedEvent
  | _, [] => []
  | state, action :: suffix =>
      M.event state action :: trace M (M.step state action) suffix

def FutureEq (M : Machine) (x y : Nat) : Prop :=
  SameFuture (trace M) x y

theorem futureEq_event (M : Machine) (x y : Nat) (action : Bool)
    (h : FutureEq M x y) : M.event x action = M.event y action := by
  simpa [trace] using (h [action])

theorem futureEq_step (M : Machine) (x y : Nat) (action : Bool)
    (h : FutureEq M x y) :
    FutureEq M (M.step x action) (M.step y action) := by
  intro word
  have htail := congrArg List.tail (h (action :: word))
  simpa [trace] using htail

end CB01
