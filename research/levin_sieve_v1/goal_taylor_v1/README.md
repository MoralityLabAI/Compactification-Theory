# LSV-17: Signed-adjoint Taylor certificates

Development experiment, 2026-11-04. Exact rational algebra tests pass locally. The checkpoint experiment is run through the pinned GitHub Actions environment; consult its actual outcome before making an empirical claim.

For a composition x_l=f_l(x_(l-1)), let J_l be each Jacobian at the clean state and c_l the clean downstream adjoint of a fixed final affine logit margin. If local Taylor residuals satisfy ||r_l||_2 <= B_l R_(l-1)^2/2 throughout a verified input ball, exact telescoping gives

    m(x+delta)-m(x) = grad(m)(x) dot delta + sum_l c_l dot r_l.

Thus for ||delta||_infinity<=r on a specified support,

    m(x+delta) >= m(x) - r ||grad(m)(x)|support||_1
                         - sum_l ||c_l||_2 B_l R_(l-1)^2/2.

Every competing class is checked. The clean gradient and adjoints use signed Arb reverse differentiation; uniform local curvature constants and transported L2 radii bound the nonlinear residuals. There is no dense Hessian allocation.

For LayerNorm N(x)=gamma*(Px)/sqrt(eps+||Px||^2/d)+beta, the Euclidean Jacobian norm is at most ||gamma||_infinity/s, and its bilinear Hessian norm is at most 2||gamma||_infinity/(sqrt(3d)*s^2). To prove the latter, normalize ||v||=1, put q=||Px||^2/(d*s^2) and y=cos(angle(Px,Pv))^2. The squared scaled directional curvature is q[1+(8-6q)y+(-12q+9q^2)y^2], and its gap below 4/3 is exactly (4-3q)(3qy-1)^2/3. Projection P has norm one. Scalarization by unit output covectors extends the diagonal bound to the symmetric bilinear operator. The constant is attained on the centered subspace when d>=3 and q>=1/3.

For a ball of L2 radius R around a clean LN input, s_min >= max(s_clean-R/sqrt(d),sqrt(eps)). All numerical evaluations of this inequality are outward enclosed. A residual GELU MLP has Jacobian norm at most 1+||W2|| g1 ||W1|| and Hessian norm at most ||W2|| g2 ||W1||^2, where g1=GELU'(sqrt(2)), g2=sqrt(2/pi). Proposed weight spectral bounds are accepted only after an Arb LDL positivity check.

The exact-real model is the frozen 44K external TRM-inspired Sudoku checkpoint used by LSV-05 through LSV-16. The goal is the first missing cell in each of the same 64 puzzles. Initial perturbations cover either all 3,888 stage-one coordinates or that cell's 48 coordinates. Hardware rounding, global contraction, recovery to a correct answer and generalization to other models have separate proof obligations.

The general curvature-certificate idea has prior art: Singla & Feizi, ICML 2020, https://proceedings.mlr.press/v119/singla20a.html ; Entesari, Sharifi & Fazlyab, ICML 2024, https://proceedings.mlr.press/v235/entesari24a.html ; Entesari & Fazlyab, HiTaB (2026), https://arxiv.org/abs/2605.10621 . This is a checkpoint-specific verified implementation and specialization, with no broad novelty claim.
