import Shaft.Observability

/-!
SHAFT-COMP-06: exact finite-horizon observational quotients,
substrate-coverage refinement, and abstract depth-error amplification.
The numerical bound uses Nat-valued errors. It makes no claim that a
neural network satisfies a particular Lipschitz or step-defect bound.
-/

universe u v w x y
namespace ShaftCompact

/-- Bounded-continuation behavioral equivalence at a finite horizon. -/
def HorizonEq {Input : Type u} {State : Type v} {Output : Type w}
    (step : Input → State → State) (observe : State → Output)
    (horizon : Nat) (s t : State) : Prop :=
  ∀ word : List Input, word.length ≤ horizon →
    observe (run step word s) = observe (run step word t)

/-- Observational equivalence at a longer horizon implies equivalence at any shorter horizon. -/
theorem horizonEq_mono {Input : Type u} {State : Type v} {Output : Type w}
    (step : Input → State → State) (observe : State → Output)
    (short long : Nat) (hle : short ≤ long) (s t : State)
    (h : HorizonEq step observe long s t) :
    HorizonEq step observe short s t := by
  intro word hw
  exact h word (Nat.le_trans hw hle)

/-- One transition consumes one unit of remaining horizon. -/
theorem horizonEq_step {Input : Type u} {State : Type v} {Output : Type w}
    (step : Input → State → State) (observe : State → Output)
    (horizon : Nat) (s t : State) (a : Input)
    (h : HorizonEq step observe (horizon + 1) s t) :
    HorizonEq step observe horizon (step a s) (step a t) := by
  intro word hw
  have hs : (a :: word).length ≤ horizon + 1 := by
    simpa using Nat.succ_le_succ hw
  simpa [run] using h (a :: word) hs

/-- Two objectives are equivalent under all allowed substrates and continuations. -/
def GoalEqOn {Substrate : Type u} {Goal : Type v} {Input : Type w}
    {State : Type x} {Output : Type y}
    (allowed : Substrate → Prop)
    (seedState : Substrate → Goal → State)
    (step : Substrate → Input → State → State)
    (observe : Substrate → State → Output)
    (horizon : Nat) (g h : Goal) : Prop :=
  ∀ b, allowed b → ∀ word : List Input, word.length ≤ horizon →
    observe b (run (step b) word (seedState b g)) =
      observe b (run (step b) word (seedState b h))

/-- More substrates yield a finer or equal observational equivalence. -/
theorem goalEqOn_more_substrates {Substrate : Type u} {Goal : Type v}
    {Input : Type w} {State : Type x} {Output : Type y}
    (oldAvailable newAvailable : Substrate → Prop)
    (included : ∀ b, oldAvailable b → newAvailable b)
    (seedState : Substrate → Goal → State)
    (step : Substrate → Input → State → State)
    (observe : Substrate → State → Output)
    (horizon : Nat) (g h : Goal)
    (newEq : GoalEqOn newAvailable seedState step observe horizon g h) :
    GoalEqOn oldAvailable seedState step observe horizon g h := by
  intro b hb word hw
  exact newEq b (included b hb) word hw

/-- An adequate finite-horizon objective code separates distinct behavior classes. -/
theorem finite_horizon_code_separates {Input : Type u} {State : Type v}
    {Output : Type w} {Code : Type x}
    (step : Input → State → State) (observe : State → Output)
    (horizon : Nat) (encode : State → Code)
    (adequate : ∀ s t, encode s = encode t → HorizonEq step observe horizon s t)
    (s t : State) (different : ¬ HorizonEq step observe horizon s t) :
    encode s ≠ encode t := by
  intro heq
  exact different (adequate s t heq)

/-- Symbolic bound for a nonnegative integer step defect and gain. -/
def depthBudget (gain defect : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => gain * depthBudget gain defect n + defect

/-- An abstract error satisfying the stated recurrence is bounded by its unrolling. -/
theorem depth_error_bound (gain defect : Nat) (error : Nat → Nat)
    (initial : error 0 = 0)
    (oneStep : ∀ n, error (n + 1) ≤ gain * error n + defect) :
    ∀ n, error n ≤ depthBudget gain defect n := by
  intro n
  induction n with
  | zero => simp [depthBudget, initial]
  | succ n ih =>
    calc
      error (n + 1) ≤ gain * error n + defect := oneStep n
      _ ≤ gain * depthBudget gain defect n + defect :=
        Nat.add_le_add_right (Nat.mul_le_mul_left gain ih) defect
      _ = depthBudget gain defect (n + 1) := rfl

/-- A unit gain gives the linear n·defect bound. -/
theorem depthBudget_unit_gain (defect n : Nat) :
    depthBudget 1 defect n = n * defect := by
  induction n with
  | zero => simp [depthBudget]
  | succ n ih =>
    simp [depthBudget, ih, Nat.succ_mul]

#print axioms horizonEq_mono
#print axioms horizonEq_step
#print axioms goalEqOn_more_substrates
#print axioms finite_horizon_code_separates
#print axioms depth_error_bound
#print axioms depthBudget_unit_gain

end ShaftCompact
