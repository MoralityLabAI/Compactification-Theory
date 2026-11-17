# SFT critique card: ACE-MechInterp Gym v0.1

Scope: only research/ace_mechinterp_gym/ in Compactification-Theory PR #28. No extra literature, broad repo inspection or model training.

Oracle: hidden-bit state (y,b), task h=y, actions identity and reveal(y,b)=(b,b). Exact quotient needs one bit with identity and two with reveal. Uniform same-code-pair violation probability is 1/4 for code C=y with both actions.

Inference: one independently acquired pair per evidence cluster. Unsafe null P(V_i=1 | complete earlier evidence)>=0.2; alternative q=0.05. E-factor (.05/.2)^V * (.95/.8)^(1-V). Robust lower envelope permits <=1 substitution plus arbitrary erasures. Four fixed child tests spend .01 each from a .05 parent budget. Policy-matched robust Bayes has the same clean-path likelihood ratio and decision threshold.

Dependence falsifier: 32 identical copies of one clean zero lead to 3/4 false certification probability when treated as independent; one source cluster blocks it. Signed acquisition and statistical independence are outside this synthetic gym.

Task: identify ONE consequential mathematical/protocol flaw or validate the conditional argument. Suggest the smallest falsifying test and minimal fix. <=180 words. Read this card only unless a specific flaw needs a targeted source line.
