# RRC-10: few-shot recovery of a hidden polynomial multiplier

Let \(F(q),G(q)\in1+q\mathbb Z[[q]]\). An episode hides the coefficients of a *new* multiplier \(H(q)\) and provides two pairs \((F_i,G_i)\), where \(G_i=F_iH\) is observed only through degree 24. The student must construct an executable \(H\) and use it on an independent query series \(X\), optionally followed by the **publicly known** N/R/S wrapper.

Since \(F_0=1\), each coefficient of the multiplier is uniquely determined from a single paired demonstration by the triangular identity

\[
 h_0=1,\qquad h_n=g_n-\sum_{j=1}^n f_j h_{n-j}.
\]

The experiment trains two weights \((u,v)\) in the public causal feature basis
\[
 \widehat h_n=u g_n+v\sum_{j=1}^n f_j\widehat h_{n-j}.
\]
A rational least-squares fit on **training-only** demonstrations at degrees 1–24 identifies \((u,v)=(1,-1)\), and a fixed 1/8 integer-snapping threshold compiles these weights. No exact teacher target is accessible at inference. The causal feature builder, convolution program, degree order, verifier and post-composition wrappers are all **hand-written** and charged as shared compute. This test identifies the coefficients of an already provided operator graph. It does not establish that a neural net invented the graph or that unknown nonlinear operators can be learned from such data.

## Scope of exact prediction

For an unknown *polynomial* multiplier with support confined to degrees 0–24, inversion of one demo through degree 24 determines the whole polynomial. The compiled interpreter therefore computes \(XH\) correctly through degree 48, for any unit-constant integer query series and for every declared known wrapper, with exact integer arithmetic. The second demonstration is an independent consistency check. A disagreement rejects the proposed operator.

A multiplier \(\widetilde H=H+q^{29}\) gives **identical demonstrations through degree 24**. The same inferred operator and prediction must be issued on both episodes, while their future outputs differ. Each of eight planted pairs has exactly one correct future prediction in the frozen test, and this is an information bound for any deterministic prefix-only procedure.

For the local baseline, \(h_n\) is approximated using only \(g_n,f_n\) and fitted weights. This is parameter-matched and uses the same pairs but lacks the convolution state. The exact inversion comparator is a **stronger algorithmic reference** with the same supplied causal primitive; its perfect matching result explains the source of transfer. A no-demonstration arm tries zero weights and abstains under exact verification.

## Accounting

The compiled weight vector contains two signed integers stored in a bounded three-bit encoding each: **six syntax bits**, excluding the convolution kernel, interpreter, demonstration data, operator coefficients and checking. The full learned rational checkpoint and all source-series coefficients are logged separately. Running inversion and applying a multiplier each requires \(O(T^2)\) integer coefficient operations; complexity of growing integer operands is extra. The experiment is an arithmetic proof of in-DSL generalization, with finite synthetic evaluation and no claim of general formal-series rule discovery.
