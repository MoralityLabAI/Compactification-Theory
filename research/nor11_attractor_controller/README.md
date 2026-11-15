# NOR-11: learned attractor controller for a trained algebraic GRU

**Protocol frozen 15 November 2026.** This is an operational test of NOR-04–10. A source-locked small recurrent model learns an additional head that predicts its own late hidden-state basin. The head can authorize a bounded hidden-state intervention, and the experiment measures both the resulting geometry and the original exact-verifier task.

The experiment trains **13 new auxiliary parameters**, with the original 1,000-parameter RRC-25 GRU and its 16-class readout frozen. It is a controlled proof-of-concept for compact recurrent reasoning, rather than a new pretrained transformer or a recurrent architecture retrained from scratch.

## Frozen source

- Original GRU: research/rrc_compactification/results/rrc25_recurrent.json, SHA-256 54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b.
- Input: the exact public nine-row, eight-feature RRC-25 demonstration tensor after the public header decoder. Only cases with **exactly one public header** are eligible, so every intervention arm has the same four-check verifier semantics.
- Target attractors: the two distinct, already certified real-analytic fixed points from NOR-04/05 and NOR-07, with rational center receipts pinned under research/nor01_numerical_orbit_sketch/results/.
- Auxiliary supervised target: apply 720 **zero-input** steps after the public nine-feature prefix to the original GRU in Python float64, then select the nearest attractor only if its distance is at most 0.05. This is an **FP64 diagnostic label** for the new source cases. Their individual basin entry is not independently certified by NOR-06/08.

## Data and leakage policy

All example generation and source seeds are predeclared in controller.py. Training has 32 cases per original train family (signed, sparse, product), development has 12 per train family, and held-out testing has 12 per all six test families, for **96/36/72 cases**. Train/dev/test seeds are unrelated to each other, to earlier NOR-06/08 source cohorts, and to the original checkpoint training-seed roster.

Training and development use **distinct public feature tensors** throughout; test tensors have zero overlap against either. The test cohort permits repeated tensors **within test** because the late-source generator can map distinct visible algebraic cases to identical early 9×8 model inputs. Both raw case-level task accuracy and feature-deduplicated basin-probe metrics are recorded, along with per-family effective feature counts and the largest alias group. Public visibility and input hashes are part of the CI receipt. These remain source-like synthetic algebraic examples, with no claims about unseen mathematical theorems from an external corpus.

The head is trained using only the twelve-dimensional hidden vector and the model-generated zero-input attractor label. The program, query future and 16-class target remain outside the training function. Development evaluates the original exact private-future grader to freeze a task-aware policy. Held-out future correctness is read **only during final scoring**. Mutating a source's hidden program and future labels leaves its public features, hidden state and auxiliary-head inputs unchanged.

## Controller and competitors

The 13-parameter auxiliary head is a full-batch class-balanced logistic regressor on standardized frozen hidden vectors, trained for 240 Adam steps with learning rate 0.035 and L2 coefficient 0.015. All standardization statistics come from train only.

The fixed escape controller computes the probability of the second attractor. When that probability is at least 0.65, it intervenes after the public input:

\[
h_{\rm changed}=(1-\alpha)h+\alpha c_1,\qquad\alpha=0.5,
\]

where \(c_1\) is the known first attracting center and the state is clipped coordinatewise to the GRU's hidden invariant cube. It then ranks the original sixteen candidate templates using the **unchanged original readout matrix**. Moving toward the first basin is an exploratory control policy, selected before seeing test outcomes. The intervention itself does not certify membership in the first ball.

Development selects among no action and six threshold/alpha combinations: threshold in {0.6,0.8}, alpha in {0.125,0.25,0.5}. Utility is

\[
\mathrm{futureCorrect}-0.5\,\mathrm{falseAccept}
-0.05\,\mathrm{interventions}.
\]

The selected policy is frozen before testing. Abstention at alpha zero is permitted.

Held-out competitors on the exact same 72 cases are: the original GRU and ranking head; development-selected basin-aware policy; fixed threshold/alpha escape; unconditional alpha-0.5 shift to the first anchor; a basin head trained on **permuted labels** with the same intervention; and a deterministic random-direction shift of comparable preclipping norm. Every arm has the original **four paid full-template verification checks**, with independent public prefix verification and withheld-future scoring.

The held-out records separate model accuracy (first/second reference-mode classification, balanced accuracy, AUC, Brier score), geometric escape (mode switches in a 720-step diagnostic continuation after intervention), and native task metrics (accepted, future_correct, false_accept, paid checks, per-family breakdown). A changed attractor mode only counts as a successful proof when the original exact verifier and future scorer pass.

## What would count as success?

The primary experimental question is whether the development-selected controller raises **held-out future-correct programs** at the same four-check budget while maintaining an acceptable false-accept rate. Secondary evidence includes generalization of the auxiliary head beyond train families, whether interventions reliably change a late mode, and whether conditioning beats unconditional or random-direction controls.

An auxiliary head can classify basins well and still reduce the native task score. That would establish measurable hidden asymptotic geometry with negative transfer to proof-template ranking.

No learned basin classifier is promoted to a NOR-10 **certified** basin label by this empirical test. A new source-specific interval capture remains necessary for a formal guarantee.

## Reproduce

Standard-library Python 3.12, from repository root:

    python -m unittest research.nor11_attractor_controller.test_controller -v
    python research/nor11_attractor_controller/controller.py --check --train-per-family 32 --dev-per-family 12 --test-per-family 12 --out /tmp/nor11_controller.json

The [NOR-11 CI workflow](../../.github/workflows/nor11-basin-controller.yml) runs unit controls, trains the auxiliary head, evaluates every arm, and archives the source-locked result as a GitHub Actions artifact. CPU-only execution changes neither the frozen checkpoint nor upstream training code.
