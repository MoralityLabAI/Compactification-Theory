# RRC-06: raw-count neural intermediates, exact certificates

**Frozen before the final 500-step training and first held-out evaluation on 2026-11-09.** CPU-only pilot; one training seed and one roster. All computations are bounded to degree 128. Model-selection decisions may use the 120-case development set; in this first pilot, architecture and training duration are fixed without development-driven selection.

## Hypothesis and observation boundary

A 2,028-parameter weight-shared GRU that scans raw coefficient features twice will propose a complete Euler-exponent vector and a log-derivative vector, with potentially more exact intermediate coordinates than a 2,084-parameter full-prefix feedforward student. The zero-algebra-input property is an essential RRC-05-to-06 improvement. The exact symbolic solver remains the resource-aware ceiling in the declared grammar; its success is not a neural accomplishment. Hypothesis success requires a measured *direct* accepted held-out program improvement over the full-prefix feedforward control and a depth response under fixed parameters, as well as exact trace checks. Any failure is retained.

Input is strictly `f_0..f_64` and three generic features per index: `log1p(f_n)/16`, 8 times neighboring log difference, and public degree fraction `n/64`. The model never receives the hidden generator, period, multiplicity cap, Euler coefficients `a_n`, logarithmic-derivative coefficients `b_n`, a divisor graph, source identities, future coefficients, or outputs from the exact solver. Supervised training targets `a_n` and `asinh((b_n/n))/3` are calculated using exact algebra, solely for the training loss. The final model emits `[-1,0,1]` Euler class logits and a continuous log-derivative prediction for each of 65 indices. Inference rounds these to integer candidate `a,b` vectors. No exact algebraic correction fills missing predictions.

## Frozen roster and exposure

Generator is `rrc06.py`, base seed `2026110916`, fixed order and roster digests. The new test random seeds and new nonperiodic generator types differ from RRC-04/05. The qualitative gap family and the Rogers–Ramanujan controls have been used previously, and cannot be described as novel blind source families.

| Split | Count | Grammar source | Access |
| --- | ---: | --- | --- |
| Train | 720 unique prefixes | random periodic products, caps ∞ and 1, moduli 2/3/4/6/8/9/10/12 | training labels |
| Development | 120 | same caps/moduli, prefix-disjoint from training | diagnostic metrics |
| Unseen periods | 72 | random products from periods 5/7/11/13/17/19, caps ∞ and 1 | test only |
| Unseen cap | 36 | cap 2, training moduli | test only |
| Local gap | 40 | adjacent gaps 1–5, minimum 1–8 | test only |
| Nonperiodic | 15 | squarefree, powers of 3, factorials, oblong, odd prime powers, all three caps | test only |
| Out of grammar | 12 | random periods 23/29/31, all caps; verifier cap 20 | test only |
| Future collisions | 10 | five base/mutant pairs at 67/79/101/113/127 | test only |

All product training/development/test unique prefixes are deduplicated globally. The collision pairs deliberately share the entire observation through degree 64, with first future mismatch at the registered mutation degree. Incidental overlap of fixed local-gap/nonperiodic controls with train/test must be enumerated and disclosed.

## Models, costs, and frozen budget

- **Recurrent:** PyTorch one-layer GRU, input 3, hidden 23, one linear head with 4 outputs per degree; 2,028 FP32 parameters, 23 FP32 live hidden-state coordinates. Two full prefix scans at training. Frozen-depth ablation at 0/1/2/4/8 scans; number of trainable weights unchanged.
- **Feedforward:** flattened 195 raw features, tanh hidden 4, 260 outputs; 2,084 FP32 parameters. One-pass access to all coefficients, 2.76% more parameters than the recurrent comparator. Same training examples, steps, supervision, optimizer and feature values. No iterative execution.
- **Negative control:** matched recurrent architecture trained on case-shuffled target vectors. It does not receive correct coefficient-to-target supervision.
- **Optimization:** all models get 500 Adam updates, batch size 64, learning rate .003, CPU single thread and deterministic seeds. Loss = weighted ternary-class cross entropy + .2 MSE on transformed `b`. Model state is serialized as ordinary JSON tensors. Standard-library inference independently reproduces torch outputs within 5e-5 on three development examples per model.
- Account for bare FP32 weight bits, recurrent state, features, output vectors, repeated scans, training updates, and exact verifier costs separately. Class and numerical head predictions are jointly graded.

## Certificates and reported measures

Exact finite-series residuals use two independently defined algebraic equations:

`R^b_n = n f_n − Σ_{k=1}^n bhat_k f_{n-k}` and
`R^a_n = n f_n − Σ_{k=1}^n f_{n-k} Σ_{d|k} d ahat_d`.

A *verified intermediate* needs zero residual at every degree 1–64. An *accepted program* additionally requires the proposed Euler `a` to parse into a periodic multiplicity grammar of period ≤20, with at least two observations per residue, and independently replay all observed coefficients. A verifier may reject or accept a proposed trace, but never rewrite a proposed vector. The exact Euler inversion / bounded program synthesizer is run independently to score a ceiling. Accepted programs are then checked through degree 128, and late-defect false acceptances are disclosed.

Report per split and arm: 64-coordinate exact `a` and exact `b` match rates, first mismatch/residual, fully verified `a/b` traces, accepted executable programs, correct full-degree predictions, false acceptances, and depth ablation. Preserve per-case trace digests and a source/checkpoint-hashed replay receipt. Re-run RRC-01, RRC-02, RRC-03, RRC-04 and RRC-05 tests under the new branch; CI must install only Python standard library for RRC-06 verification (training is optional on CPU PyTorch).

**Claims boundary:** Passing a finite certificate does not establish an infinite identity. An identical prefix is insufficient to determine an adversarially modified future, and full program recovery may fail even when many intermediate coordinates are correct. Any benefit from repeated scans must be shown against the cost-matched model and the exact baseline.
