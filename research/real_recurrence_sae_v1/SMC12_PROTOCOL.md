# SMC-12: Source-locked angular cones of self-reinforcing SAE decoder atoms

November 15, 2026. Companion: [OLM-12 in MathCan](https://github.com/patrickdugan/MathCan/pull/84). Continues the [same-fit SMC-11 source-locked experiment](SMC11_RESULTS.md).

## Research question and source boundary

SMC-11D proved that 48 selected highest-positive decoder atoms preserve their feature identity along their own positive intervention rays, at any finite ray amplitude, on a pretrained 44,001-parameter four-step recurrent Sudoku model. SMC-12 certifies how far the decoder direction can move before that self-selection relation can fail.

The teacher checkpoint is unchanged: upstream source f0406195fece4d0db29ced28f3a69d1993d95b1e, checkpoint SHA-256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1, and roster SHA-256 eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.

The experiment uses the original three **weights-only frozen SAE checkpoints**, trained once with seeds 71,89,113, each a 48→96→48 top-four ReLU autoencoder with 300 optimizer updates and train-only normalization. It reads the **same fitted tensor bytes** that SMC-11C used for exact feature identity and validated downstream causal comparisons, so this stage does not retrain anything.

Source [SMC-11C full run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43721281348) and its [full snapshot artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43721281348/artifacts/14164917026) are checked before proceeding. Original ZIP SHA-256 is 1b62914ba7fecd91614eab36c1f799e2fc0d6d4e3279d0e1ac03a0ee5ee2a2fb; the source joint-full.json SHA-256 is 7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087. All three SAE tensor files have independently recorded SHA-256 values in the source experiment and the new verifier.

## Exact mathematical certificate

For a frozen highest-positive SAE neuron t, use the physical feature-score normal v_i = W_i / scale. Its all-positive-amplitude self-selection cone is

\[
C_t=\{d:v_t\cdot d\ge0,\ (v_t-v_j)\cdot d\ge0
\text{ for all }j\ne t\}.
\]

The physical direction is the **same FP64-normalized dyadic vector** chosen by SMC-09/11 after original FP32 decoder×train-scale multiplication. The exact rational verifier computes 96 score-gap slopes per case and

\[
q_*^2=\min_{n\ne0}\frac{(n\cdot d)^2}{\|n\|^2\|d\|^2}.
\]

A positive minimum certifies the direction lies strictly inside every cone halfspace. The sharp relative raw decoder error tolerance is the square root of q_*²; a unit-direction chord tolerance delta is certified by the exact rational guard q_*² greater than delta² minus delta⁴/4, for 0 < delta² < 2. An integer-square-root bracket supplies a rational lower and upper bound for the raw threshold.

The verifier constructs the exact shortest perturbation to the limiting score hyperplane for every one of the 48 sources, then checks that a slightly larger perturbation gives a negative cone slope. This is a **sharp proof witness**, rather than a sample of random directions.

The joint physical raw decoder and encoder score-normal guard also tests whether q_*² exceeds [rho+eta(1+rho)]², where rho and eta bound relative norm errors in the physical decoder and each score-constraint normal. This condition preserves the cone direction after such errors **if the perturbed encoder is also known to select t strictly at the starting state**.

## Entire dictionary and initialization controls

An independent SMC-12B experiment tests all 96 dictionary atoms for each of the three fitted SAEs and for their **original random initializations**, with the exact same train-normalization scale. Original initial decoder hashes are pinned to the earlier SMC-07 training receipt.

[OLM-12E](https://github.com/patrickdugan/MathCan/pull/84) gives a separate algebraic characterization: in ideal exact-real scaling, define the encoder-decoder cross-talk matrix K = W D. Atom t self-reinforces iff K_tt is nonnegative and dominates its entire column. That criterion is independent of the positive normalization scale. The experiment recomputes both this exact K criterion and the actual source-rounded FP64 decoder-ray cone, and fails on any classification disagreement.

A dictionary cone classification is conditional on the neuron first being the strictly highest positive feature at some source state. It does not show that every self-reinforcing neuron activates in data, or that its feature has a semantic meaning. This three-seed control does not establish a general population effect.

## Scientific scope

The guaranteed all-positive-amplitude self-selection property concerns only the **frozen affine SAE feature identity**, not the pretrained teacher's downstream logits. Downstream causal effect rankings were separately certified by SMC-09–11 for selected finite radii and small input uncertainty budgets. The full top-four support can change while its highest positive feature remains fixed.

The source test set contains the same 16 previously inspected Sudoku puzzles as SMC-07–11. This experiment does not provide independent generalization, new teacher checkpoints, hardware FP32/FP64 roundoff certification, source-model accuracy improvement, or feature-semantic interpretation.

## Reproduce

The source uses standard-library rational arithmetic plus PyTorch 2.7.1 CPU. Run the source tests and frozen replay:

    python -m unittest discover -s research/real_recurrence_sae_v1 -p 'test_smc12_cones.py' -v
    python -m unittest discover -s research/real_recurrence_sae_v1 -p 'test_smc12_dictionary_controls.py' -v
    python research/real_recurrence_sae_v1/smc12_cones.py --zip /path/to/smc11c-same-fit-verified-full.zip --out runs/smc12/cones-full.json
    python research/real_recurrence_sae_v1/smc12_dictionary_controls.py --zip /path/to/smc11c-same-fit-verified-full.zip --out runs/smc12/dictionary-control.json

The [SMC-12 CI workflow](../../.github/workflows/smc12-conic-self-selection.yml) downloads and hashes the original artifact, reruns the mathematical and tampering tests, and archives both exact source-locked experiment receipts.
