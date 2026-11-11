# OIC-29: finite code distance and the cost of preprogrammed sensing

**11 November 2026.** Mathematical status: exact finite constructive witnesses, verified by two independent formulations of the sensor codebook and by exhaustive fault-receipt enumeration. Neither the seven-phase adaptive minimum nor the 21-query fixed-schedule decision is resolved here.

## Observation contract

For n=7, a hidden world is (d,z0) in Z7². At time t, a predetermined sensor address m_t returns one binary bit

\[
 c_t(d,z_0)=[z_0+td\equiv m_t\pmod7].
\]

The target is d. The initial phase is unknown. At most one report bit may be inverted, with no erasure. The *fixed* schedule is specified before any observations. Distinguishing d is different from distinguishing all 49 (d,z0) pairs.

## OIC-T55: Cross-shift code-distance criterion

For a fixed schedule \(w=(m_t)_{t=0}^{Q-1}\), define the codebook \(C_d(w)=\{(c_t(d,z))_{t<Q}:z\in\mathbb Z_7\}\). Under up to e substituted report bits, every true world can be correctly identified by **shift** (regardless of initial phase) if and only if

\[
\boxed{\min_{d\ne d'}\min_{z,z'} d_H(c(d,z),c(d',z'))\ge2e+1.}
\]

*Proof.* When the minimum distance exceeds 2e, Hamming balls of radius e around codewords associated with different shifts are disjoint, so the decoder can label every received word. Conversely, if a cross-shift pair has distance at most 2e, divide their disagreement coordinates into two groups of size at most e and construct a received word by taking each codeword's symbol on the corresponding complementary group. That received word lies inside both balls. A single correct shift decision is then impossible. QED.

## OIC-T56: Two explicit seven-phase correction witnesses

The following fixed schedules obey the theorem's criterion with e=1. Each minimum is verified by enumerating all C(7,2)·7²=1,029 distinct-shift world pairs. A second verifier enumerates the 49 hidden worlds and every fault position including the zero-fault case. It independently checks that each feasible received word is compatible with one shift only.

**22 queries at three available sensor labels (0,1,2):**

~~~text
0,1,2,2,1,2,0,2,0,0,1,2,1,1,1,2,1,0,0,0,2,1
~~~

The minimum cross-shift distance is 3. The second check visits 49×23=1,127 true-world/fault paths, comprising 1,058 distinct received words.

**27 queries at two available sensor labels (0,1):**

~~~text
0,0,1,1,1,1,1,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,1,1,0,0,1
~~~

The minimum cross-shift distance is 3. The second check visits 49×28=1,372 paths, comprising 1,260 distinct received words.

These are constructive upper bounds for fixed schedules. Minimality of the address count at either query horizon is unproved; Q=21 fixed feasibility is open. OIC-28's Q=20 integer infeasibility remains a *solver-reported* finding with no independent UNSAT certificate.

## OIC-T57: Query count and total communication can rank policies differently

A dynamic fully addressed controller with Q queries, n possible positions and direct binary address coding transmits Q report bits and Q ceil(log2 n) sensor address bits. At n=7, the verified adaptive Q=20 policy therefore uses 20+60=80 online link bits before charging its decision-tree description, local belief state or one-time configuration.

A fixed policy can instead preprogram its query word. If it uses k distinct physically accessible marker labels, a simple literal encoding charges k ceil(log2 n) bits to install those labels and Q ceil(log2 k) bits for the schedule, in addition to the Q sensor reports. The example encodings (with no general compression) give:

| Policy | Queries | Distinct positions | Online bits if every address is transmitted | Marker+schedule configuration | First episode with preprogramming | Subsequent episode report bits |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive Q20 (OIC-28) | 20 | up to 7 | 80 | policy tree additionally charged | at least 80 plus tree | 80 |
| Fixed Q22 | 22 | 3 | 88 | 9+44=53 | 75 | 22 |
| Fixed Q27 | 27 | 2 | 108 | 6+27=33 | 60 | 27 |

The offline-configured alternatives trade additional time steps for a smaller live command channel. Their first-episode totals presume preprogramming once and charge those bits once; they do not include installation hardware, synchronization, policy generation or any extra cost for reprogramming. If addresses must be sent online, dynamic addressing costs apply.

The fixed schedules are independent of sensor reports and permit exact one-substitution correction via simple codebook lookup. The adaptive Q20 policy requires conditional queries and its separately archived 1,697-node decision tree.

## Scope and reproduction

- `oic29_fixed.py`: independent pure-Python reference over integers, full 1,029-pair distance spectrum and exhaustive receipt collision verifier.
- `test_oic29_fixed.py`: six tests, including phase-gauge invariance, fault cases and resource ledger.
- `results/oic29_fixed_reference.json`: deterministic machine-readable witness words and histograms.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic29_fixed.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic29_fixed.py' -v
python research/observational_inwardness_v1/oic29_fixed.py --check
~~~

This is finite-state active sensing. It gives no mathematical assertion about subjective experience, actual looped-transformer hidden dynamics or the intrinsic nature of spacetime.
