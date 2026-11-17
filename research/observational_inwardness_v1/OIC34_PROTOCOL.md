# OIC-34: Certified n=7 adaptive threshold and conflict-guided search

**Prospective, November 11, 2026.** OIC-33 proves a tight-pair marker
intersection cut that reduces both n=7 reference search sizes by about 75%,
but Q19 still reaches an explicit resource cap. The certified one-substitution
adaptive interval remains 15..20; the global fixed optimum is 21.

## Primary task

Resolve Q19 with either a complete verified policy or a checkable negative
search certificate. If Q19 is SAT, descend to Q18. If UNSAT, establish the
exact threshold as 20 because OIC-28 has already verified Q20 SAT.

## Search directions

- Build a conflict-directed memo of *losing candidate subsets* and prove
  monotone dominance (adding compatible candidate worlds to an information
  state cannot make a losing state winnable).
- Extend OIC-T70 by a sound two-step necessary address lookahead, explicitly
  separating branches that may receive different adversarial reports.
- Evaluate exact retrograde enumeration and quantified Boolean formulations
  on n<=4 and match the independent OIC-23 Bellman oracle first.
- Add deterministic resumable checkpoints for explorations over 50 million
  states, with source-hash and search-bound metadata and a replayable final
  proof trace. A partial checkpoint is not an UNSAT proof.

## Evidence and resource contract

Keep the original 49-world belief masks per fault layer (98 logical bits),
all 42 affine gauges, and mandatory-marker OIC-T70 checks. Report source
hash, search heuristic, explicit node and wall caps, peak memory, and whether
execution ended in SAT, UNSAT, CAP or TIMEOUT. Only SAT certificates pass an
independently written public-history and true-world fault-path verifier.

Compare adaptive online address/report bits against the fixed 21-query
three-marker block code under separate policy-description accounting. No
conclusion about a trained recurrent model is implied by this synthetic gym.
