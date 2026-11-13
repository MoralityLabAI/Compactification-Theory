# SCM-02 mathematics: configuration-space obstruction and finite observation

**Proposed theorem layer; code verification is separate from model experiments.**

## A disk obstruction for the recurrent chart

Let `n >= 2`, and fix a deterministic model, input tokens, a residual injection
site and two linearly independent directions `u,v` at that site. For a radius
`epsilon`, let `p(r,theta) = r * (cos(theta) u + sin(theta) v)` with
`0 <= r <= epsilon`. Fix two observed residual coordinates from a *frozen*
linear chart `P:R^H -> R^2`. Each tracked token `i` defines a continuous map
`z_i:D^2 -> R^2` wherever the model inference and chart are continuous.

Define the pairwise discriminant `D_ij={q in D^2 : z_i(q)=z_j(q)}` and
`D=union_{i<j} D_ij`. If `D` is empty, `F=(z_1,...,z_n)` maps the entire
contractible disk into `Conf_n(R^2)`. Hence its restriction to the boundary
circle is null-homotopic, giving the **identity pure braid**.

Contrapositive: a *certified nontrivial* boundary braid implies an interior
projected collision. This is a geometric obstruction in the chosen chart, not
an intrinsic singularity of neural state space. A nontrivial Artin braid has
more information than its Alexander polynomial or pairwise winding numbers.

### Pairwise integer and index theorem

Fix `i != j` and write `f_ij(q)=z_i(q)-z_j(q)` as a map from the closed
intervention disk to `R^2`. Assume `f_ij` is continuous and nonzero on the
boundary. Its boundary winding (counterclockwise x/y convention) is

    w_ij = (1/(2*pi)) * integral_boundary d arg(f_ij).

It is an integer. If `w_ij != 0`, at least one interior point satisfies
`f_ij(q)=0`. If `f_ij` is `C^1` and transverse to zero, the signed count
of interior zeros equals `w_ij` (Brouwer degree / planar index theorem).
For a pure braid, the algebraic pairwise link `lk_ij` of its closed strands
uses the SCM-01 Artin crossing convention and equals **`-w_ij`**. This
minus sign comes from the convention that positive generators cross with
positive y-height on the left-to-right strand. Tests compare an independent
polygonal winding algorithm to the signed Artin crossing count / 2.

A vanishing vector of pairwise windings leaves room for nonabelian braid
information when `n >= 3`. For `n=2`, the pure braid group is `2Z` inside
`B_2 = Z`; pairwise winding completely characterizes the pure braid class.
The initial measurement deliberately fixes `n=2` for a rigorous scalar
observable with a straightforward interpretation.

## Chart gauge and time sampling

An orientation-preserving *fixed* invertible affine transformation of the
observed plane preserves the winding integers. A reflection changes their
sign. A singular projection, a nonlinear time-dependent readout, or switching
between SAE features during the loop can change the measured braid class.
SCM-02 therefore reads raw residual states with a **fixed linear** decoder-row
chart. The sparse TopK switch is never the map defining the strand points.

The code's `pairwise_windings` computes the exact winding of the observed
piecewise-linear polygon, rejecting an origin crossing in any pair difference
segment. This is a statement about interpolation, not the unknown continuous
neural path between samples. Doubling the angular density is an empirical
robustness control.

A **sufficient continuous-path certificate** can be supplied when an independent
method proves an upper bound `L_ij` on
`|| d(z_i-z_j)/dtheta ||` over every angular interval of length `dtheta`.
If the measured starting difference has norm strictly larger than
`L_ij*dtheta + clearance` on each interval, both the unknown continuous path
and its straight interpolation stay within the origin-avoiding convex disk
centered at the measured starting difference. The linear homotopy between
them avoids every collision. Applied to every pair, this certifies the
observed braid class for the actual continuous boundary path.

A derivative evaluated only at sampled angles, finite differences, and
unvalidated spectral estimates **do not supply such a global upper bound**.
The runner labels all current real-model outputs `polygonal_only` until a
separate verified bound has been supplied and checked.

## Implications for mechanistic interpretability

The earliest valid, nonzero `w_ij` on a downstream recurrence visit marks a
candidate phase boundary where *projected relative token coordinates* become
nontrivial under a fixed cyclic perturbation. It localizes a chart-level
collision within the two-dimensional perturbation disk, assuming the
continuous assumptions and sampling certificate hold. A follow-up can search
for the projected zero using radial sweeps or a root finder and measure its
causal relationship to named intermediate variables and final answers.

The route to Miller's result requires a subsequent, separate construction of
properly embedded 4D surfaces spanning the *same* link and a surface-specific
isotopy obstruction. A braid word and a canonical genus from Seifert's
algorithm do not supply a four-dimensional non-isotopy theorem.

## Sources

- Hayden, Kim, Miller, Park, Sundberg, *Seifert surfaces in the 4-ball*,
  Journal of the European Mathematical Society (2025), DOI: 10.4171/JEMS/1703.
- Kim, Miller, Yoo, *Non-split alternating links bound unique Seifert surfaces
  in the 4-ball*, Compositio Mathematica 161 (2025), 2712-2723.
- Classical identification of `pi_1(Conf_n(R^2))` with the pure braid group,
  and degree of maps from planar disks to the plane. The above proof is
  self-contained modulo those standard identifications.
