# LSV-02: stale-information recovery with a certified slowdown

Date: 2026-11-04. Evidence: analytic proof and exact-rational finite tests; mathematical extension of [LSV-01](THEOREMS.md) under bounded delay. The result uses established ideas in asynchronous iterations and small-gain analysis. It is not presented as a new theorem in the literature or a Lean-verified result.

## Theorem D1: B-fair, D-stale repair

Let the weighted small-gain certificate of LSV-01 hold, with A>=0, w>0 and Aw<=rho*w for 0<=rho<1. Assume only one coordinate updates per step, and other coordinates remain fixed. The updating coordinate i may independently read any coordinate j from a state at time s_{t,j}, where t-D <= s_{t,j} <= t:

    |e_i(t+1)| <= sum_j A_ij |e_j(s_{t,j})| + eta_i.            (D1)

Specify the entire initial history e(-D),...,e(0), and define

    E(t)=max_i |e_i(t)|/w_i
    H_0=max_{-D<=s<=0} E(s)
    delta=max_i eta_i/w_i,   R=delta/(1-rho).

Each aligned block of B updates must visit every coordinate. Define

    L = B*(1+ceil(D/B)),

where ceil(0/B)=0. Then for all integers t>=0,

    E(t) <= R + rho^floor(t/L)*max(H_0-R,0).                  (D2)

Moreover, at times t=kL the entire last-D-step history satisfies the same bound with exponent k. If a goal tube tolerance tau>R is specified, the number of superblocks sufficient to guarantee entry is the least k satisfying

    rho^k * max(H_0-R,0) <= tau-R.                              (D3)

**Proof.** Set d=max(H_0-R,0), so the initial history is at most R+d. At any future update whose inputs remain at most R+d, the new component is bounded by rho*(R+d)+delta = R+rho*d. Untouched components remain bounded by R+d; hence by induction the entire trajectory is bounded by R+d. Because the first aligned B updates visit all components, at time B every component has last been updated with inputs at most R+d, giving E(B)<=R+rho*d. From time B onward, further updates still use inputs bounded by R+d, so their outputs remain at most R+rho*d, and the current-state bound is maintained. After L steps, L-B >= D, so every state in the current D-step history lies at or after time B and obeys E<=R+rho*d. The superblock endpoint L is a multiple of B, allowing the same aligned-fair argument to restart from L with d replaced by rho*d. Induct over superblocks. Between endpoints, the current state remains inside the starting bound. QED.

**Relation to no-delay case.** With D=0, L=B, so D2 specializes exactly to theorem T1. With a sliding-window B-fair schedule that visits all coordinates in *every* B consecutive updates, the same reasoning permits a shorter interval L=B+D without the rounding to an aligned block boundary.

## Sharpness example

For n=1, B=1, D=1, eta=0, choose A=(rho) with 0<rho<1 and the maximal-delay rule

    e(t+1)=rho*e(t-1),    e(-1)=e(0)=1.

Then e(2k)=rho^k, attaining the bound at every superblock boundary L=2. The reference test checks rho=1/2 and four superblocks exactly, including the final value 1/16.

## Implementation and limitations

[delayed.py](delayed.py) implements an exact rational audit of explicit histories, update indices, per-coordinate read ages, and local contracts. It checks the derived finite-time envelope with no floating-point tolerance. [test_delayed.py](test_delayed.py) covers zero-delay reduction, a sharp delayed scalar witness, a two-component network with changing delays, and malformed or invalid traces.

The bound depends on a uniform maximum information age D. Unboundedly stale messages, simultaneous conflicting writes, and changing goals require additional assumptions. The state domain must include the full history and every reachable intervention state for a global guarantee. The certificate's radius can exceed useful tolerance in an application. Sample traces certify only what was audited, so a neural deployment needs independently justified sensitivity bounds.