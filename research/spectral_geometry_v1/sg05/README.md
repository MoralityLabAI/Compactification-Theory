# SG-05: learned dual warm starts for certified spectral routing

Follow-up to [SG-04](../sg04/README.md). For state (t,S) and multiplier eta, each component independently chooses its remaining consumers, paying its last-use source price if not already in S. Add sum eta_i*k_i. The resulting bound is convex and piecewise linear. Holding other coordinates fixed, split each component's feasible masks into those excluding i and those including i (before charging eta_i); their best relaxed scores are F0,F1. Its contribution is F0+max(0,F1-F0-eta_i). Sort these breakpoints descending; for 0<k_i<d any eta in [breakpoint_(k_i+1),breakpoint_(k_i)] minimizes the coordinate. For k_i=0 choose the largest breakpoint, and k_i=d the smallest. Active labels have breakpoint u_ij; inactive labels enumerate last-use indices and choose all profitable earlier visits, respecting inclusion/exclusion. We choose the integer midpoint in the minimizing interval. Exact coordinate minimization need not globally minimize a nonsmooth convex function; no convergence-to-global-optimum claim.

Run node sg05.mjs > results-replay.json. The portable JS source was executed before commit. 48 training roots, 36 fresh same-size and 12 larger exposed development roots; train roots have d=8,T=6, larger d=9,T=7. Three planted utility regimes. Training: 12 sampled states/root, teacher is best bound among zero/cutoff starts with up to four alternating coordinate sweeps. Fit a 13-coefficient ridge-style linear predictor of multiplier residual relative to row cutoff. Shuffled control permutes residual targets within each training root and fits identical procedure. Predictions rounded to integers and clipped to [-100,100]. All policies retain the same minimum of six SG-04 fixed vectors plus cost-free top-k. Compare fixed, zero+one reverse sweep, cutoff+one reverse sweep, learned direct (no sweep), learned+one reverse sweep and shuffled+one reverse sweep. Learned and shuffled use same feature/fit budget; no tuning on evaluation roots. Sampled state bounds include root. Search uses same feasible incumbent and best-bound ordering; on-demand cached bounds differ only in multiplier policy. Caps count generated action transitions, not preprocessing or ranking; dual evaluations and coordinate updates are reported separately and have different costs. The model sees all offline utilities, no native recurrent inference. No confirmation, no hardware speed claim, no GPU or paid inference. All evaluation roots and every arm are archived, not hidden.

Finite checks: 64 small tasks, 6144 exact coordinate comparisons, 497664 alternative integer grid values, 3072 monotonic sweep checks; 4608 evaluation-state admissibility checks and 576 search certificate comparisons to independent exact prefix DP. All passed. Coordinate-grid checks are finite validation of the mathematical derivation, not a formal proof. Zero/full budget and zero lambda cases included. The certificate derives from the mathematical dual bound, never the learned score.

## Results
{
  "2048": {
    "fixed": {
      "roots": 48,
      "regret": 0.2916666666666667,
      "gap": 0.3125,
      "edges": 526.75,
      "dual": 339.5,
      "coordinates": 0,
      "optimal": 47,
      "certified": 47
    },
    "zero": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 255.25,
      "dual": 336.8333333333333,
      "coordinates": 192.97916666666666,
      "optimal": 48,
      "certified": 48
    },
    "cutoff": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 279.4583333333333,
      "dual": 360.8333333333333,
      "coordinates": 207.66666666666666,
      "optimal": 48,
      "certified": 48
    },
    "learned_direct": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 446.5,
      "dual": 361.5208333333333,
      "coordinates": 0,
      "optimal": 48,
      "certified": 48
    },
    "learned": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 244.04166666666666,
      "dual": 315.6666666666667,
      "coordinates": 179.625,
      "optimal": 48,
      "certified": 48
    },
    "shuffled": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 239.5,
      "dual": 326,
      "coordinates": 186.375,
      "optimal": 48,
      "certified": 48
    }
  },
  "8192": {
    "fixed": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 550.375,
      "dual": 345.75,
      "coordinates": 0,
      "optimal": 48,
      "certified": 48
    },
    "zero": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 255.25,
      "dual": 336.8333333333333,
      "coordinates": 192.97916666666666,
      "optimal": 48,
      "certified": 48
    },
    "cutoff": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 279.4583333333333,
      "dual": 360.8333333333333,
      "coordinates": 207.66666666666666,
      "optimal": 48,
      "certified": 48
    },
    "learned_direct": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 446.5,
      "dual": 361.5208333333333,
      "coordinates": 0,
      "optimal": 48,
      "certified": 48
    },
    "learned": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 244.04166666666666,
      "dual": 315.6666666666667,
      "coordinates": 179.625,
      "optimal": 48,
      "certified": 48
    },
    "shuffled": {
      "roots": 48,
      "regret": 0,
      "gap": 0,
      "edges": 239.5,
      "dual": 326,
      "coordinates": 186.375,
      "optimal": 48,
      "certified": 48
    }
  },
  "bounds": {
    "fixed": {
      "root_slack": 24.375,
      "state_slack": 4.72265625
    },
    "zero": {
      "root_slack": 4.708333333333333,
      "state_slack": 0.5247395833333334
    },
    "cutoff": {
      "root_slack": 9.75,
      "state_slack": 1.16796875
    },
    "learned_direct": {
      "root_slack": 11.854166666666666,
      "state_slack": 3.0221354166666665
    },
    "learned": {
      "root_slack": 2.7291666666666665,
      "state_slack": 0.2721354166666667
    },
    "shuffled": {
      "root_slack": 3.6041666666666665,
      "state_slack": 0.3684895833333333
    }
  }
}

All outcomes are development diagnostics. Compare learned against analytic warm starts and shuffled control; smaller search work may trade against extra dual arithmetic. No claim of generalization to neural models or cost savings in wall time.
