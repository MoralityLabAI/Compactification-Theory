# SMC-26E: Source-alias repair before new test generation

**18 November 2026, pre-outcome engineering amendment.** The scientific SMC26 protocol was first committed at 6bbd2c6 before training B811/B919 or generating novel puzzle cohorts.

Original [GitHub Actions #44129876450](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44129876450) passed seven source-independent tests, original artifact checks, and the complete new-teacher training/calibration stage. B811 chose validation step 175 and had old-calibration accuracy 45.8984%; B919 chose step 150 and had 41.6016%. Old B501/B701 had 45.8984% and 50.5859%. All four selected temperature T=2 and passed the predeclared capability band.

The prediction job halted while loading original frozen B501/B701 SAE71 aliases, reporting a missing original B teacher SAE component. The original SMC-13A component names are newB_501_SAE_71 and newB_701_SAE_71. The SMC26 loader mistakenly searched for newB501_SAE_71 and newB701_SAE_71. A single correction now calls a tested prior_component_key function that returns the original names. The source-independent regression suite adds a test of these exact frozen aliases.

This halt preceded new SMC-26 test puzzle generation, physical intervention selection, any test-native forward pass, or access to new test labels. The new full workflow regenerates the same source-registered new-teacher seeds and SAE71 fits, freezes calibration first, and then uses the unchanged fresh Sudoku source seed bases 202636030000/202636040000. No scientific gate, action policy, bootstrap seed, or cutoff was changed. The failed run remains available for reproducibility.
