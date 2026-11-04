# LSV-06: Output-goal stability from exact affine margins

**2026-11-04.** This is an exact mathematical readout certificate applied to already measured hidden-state perturbations. It does not prove that the recurrent transition itself recovers an undamaged latent state. The distinction matters: LSV-05 experimentally finds expanding hidden-state discrepancies with largely preserved Sudoku predictions.

For a frozen affine readout \(\ell_r(z)=w_r^\top z+b_r\), let \(r\) be the unique clean predicted label and define

\[
  m_r(z)=\min_{s\ne r}(\ell_r(z)-\ell_s(z)),\qquad
  K_r=\max_{s\ne r}\sum_j |w_{rj}-w_{sj}|.
\]

**Theorem GM-01 (goal-level margin certificate).** If \(m_r(z)>K_r d\), then every \(z'\) with \(\|z'-z\|_\infty\le d\) has the same unique top-1 prediction \(r\).

**Proof.** For any competitor \(s\),
\[
\ell_r(z')-\ell_s(z')
=(\ell_r(z)-\ell_s(z))+(w_r-w_s)^\top(z'-z)
\ge m_r(z)-\sum_j |w_{rj}-w_{sj}|\,\|z'-z\|_\infty
\ge m_r(z)-K_r d>0.
\]
Every competing class has lower score. QED.

The code calculates each \(K_r\), clean affine-logit margin and per-cell observed hidden-state discrepancy \(d\) from the **exact rational numbers represented by stored FP32 tensor values**, avoiding float rounding inside the certificate. It explicitly verifies that every certified prediction is unchanged under the paired altered-state calculation in exact arithmetic. The primitive is implemented in [`goal_margin.py`](goal_margin.py).

The radius \(d\) in the current experiment comes from a *recorded perturbed trajectory*. Consequently the certificate proves the readout remains stable in a ball containing the paired final hidden state. A fully predictive perturbation certificate additionally needs a uniform, pre-execution bound on \(d\) derived from the recurrent dynamics. The global Lipschitz bound in LSV-05 is approximately \(7.74\times10^{10}\) per block and yields an unusably large four-step bound. State magnitude and output correctness can remain stable while error vectors grow; the task-level margin addresses only the latter readout stage.

All models and tasks retain the [locked upstream source](source_lock.json) and [frozen evaluation protocol](PROTOCOL.md). Report certified label preservation alongside observed classification stability, and explicitly count the gap. The six noise streams and Sudoku answers are fixed before this analysis.
