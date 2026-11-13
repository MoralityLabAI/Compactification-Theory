# SCM-05: Certified sampled-braid topology and bit-depth under compactification

**Evidence class:** exact-rational finite geometry over recorded samples, a mathematical proof conditional on known interval Lipschitz constants, and executed synthetic fixtures. No trained RMP activation data or certified neural Jacobian bounds are claimed. The observed knot/link belongs to a fixed planar readout chart, not intrinsically to the hidden-state manifold. No Seifert-surface isotopy or Khovanov obstruction is computed.

## 1. The finite-sampling alias problem

A closed cyclic intervention produces a continuous loop `z: S^1 -> Conf_n(R^2)` when every projected pair is collision-free. For a pair `q_ij(theta)=z_i(theta)-z_j(theta)` and a fixed interval `[theta_k,theta_{k+1}]` of width `h`, define `p_ij` to be the straight chord between its sampled endpoints.

Finite samples alone cannot certify the topology of `z`. A loop making `m` whole turns between adjacent observation angles can return to the same sampled positions while its actual winding is nonzero. SCM-05 explicitly rejects a four-turn circle sampled at four uniform angles when supplied its mathematically valid derivative upper bound.

### Exact interpolation lemma

If `q_ij` is `L_ij,k`-Lipschitz on this interval (for the Euclidean norm), write `theta=(1-s)theta_k+s theta_{k+1}`. Then

    q(theta) - p(theta)
      = (1-s) (q(theta)-q(theta_k)) + s (q(theta)-q(theta_{k+1})).

By the triangle inequality and Lipschitz bounds,

    ||q(theta)-p(theta)|| <= 2 L_ij,k h s(1-s) <= L_ij,k h/2.

The constant `1/2` is sharp for the class of L-Lipschitz maps; a triangular excursion at the interval midpoint saturates it. The bound is applicable to *relative differences*, so it may be much sharper than summing separately estimated per-strand constants.

Let `m_ij,k` be the minimum Euclidean distance of the PL chord `p_ij([theta_k,theta_{k+1}])` from zero. If

    m_ij,k > L_ij,k h/2 + epsilon              (SCM-05.A)

for every interval and pair, the straight homotopy between `q` and `p` remains collision-free. The full labeled continuous configuration is homotopic to the sampled PL configuration through `Conf_n(R^2)`. For exact closed individual strands, the oriented, ordered closed braid and every associated link invariant (including Milnor's triple number when defined) are preserved. The based braid is only directly comparable when its base configuration is fixed.

**Proof boundary:** the program does not derive or verify `L_ij,k` from a trained model. An isolated autodiff Jacobian, finite-difference estimate, or agreeing 32/64-sample braid does not supply a supremum on the interval. A positive program result reads **conditional on an independently validated global/interval bound**. The program also compares recorded numeric endpoints *exactly*; exact closure of the underlying continuous loop must follow from the deterministic intervention/model contract.

## 2. Two continuous histories, one topology

Let `z` be the original continuous loop and `zhat` a reconstructed continuous loop, with exact closed endpoints. For each pair and interval, suppose trusted derivative bounds `L_ij,k` for `z_i-z_j` and `Lhat_ij,k` for `zhat_i-zhat_j`. Let

- `m`: minimum original PL pair clearance on the interval;
- `E`: maximum relative *sampled* reconstruction error at its two endpoints;
- `B = L h/2`, `C = Lhat h/2`.

The PL reconstructions have pairwise difference at most `E` throughout each interval. Hence the original continuous `z` and reconstructed continuous `zhat` are both homotopic to the original PL loop through collision-free maps if

    m > max(B, E+C) + epsilon                  (SCM-05.B)

for each pair and interval. The two homotopies concatenate. This preserves the conjugacy/free-homotopy class, hence the oriented link type, and preserves the based braid whenever both start at the same ordered configuration. It does not identify any particular Seifert surface spanning that link.

## 3. Exact rational measurement layer

For a chord from rational vectors `a` to `b` in `R^2`, set `v=b-a` and

    tau = clamp(-<a,v>/<v,v>, 0, 1),
    m^2 = ||a+tau v||^2.

Everything is evaluated in Python `Fraction`: decimal strings are exact decimals, binary floats are treated as their exact stored rational values. Square-root comparisons use mathematically outward rational rounding to a fixed `10^-12` grid. A pass therefore remains conservative relative to the supplied rational *recorded* coordinates and Lipschitz claims, with no floating-point margin mistakes in the geometric predicate. The continuous neural trajectory and its derivative bounds require separate mathematical justification.

## 4. Explicit quantization bound

Quantize each measured chart coordinate in the fixed interval `[-R,R]` with `b` bits and `2^b` uniform bins. Reconstruct bin centers and interpolate linearly between sampled quantized configurations. This decoder is a continuous polygonal curve by construction, even though the coordinatewise quantizer itself is discrete.

The error per coordinate is bounded by `delta=R/2^b`, provided *all sampled values are in range* (the program rejects saturation). A strand's Euclidean sample error is at most `sqrt(2)delta`, and a pair's relative error is at most `2 sqrt(2)delta`.

If the original sampled PL pair clearance `m` satisfies both

    m > L h/2 + epsilon,
    m > 2 sqrt(2) R / 2^b + epsilon             (SCM-05.C)

for every pair and interval, the true original continuous loop and decoded quantized PL loop have the same closed-braid/link topology. The program reports both an actual-data-dependent error certificate and the stronger **uniform worst-case** certificate (SCM-05.C). A sufficient bit knee is the smallest tested `b` satisfying the worst-case check. It is an upper bound on bits needed by this specific chart quantizer, not an information-theoretic lower bound or a neural-model compression achievement.

A useful subtlety: low-bit quantization can create exact ties in the **x coordinate** at a sampled slice. The SCM-01 extractor intentionally rejects such frames when recovering Artin crossing words. In SCM-05 the independently verified straight-homotopy witness can still establish isotopy when complete **2D** pair separation is preserved. The synthetic `C=[sigma_1^2,sigma_2^2]` fixture has Milnor `mu_123=+1`; its PL curve survives a certificate at 3 bits per coordinate with `R=3` even though the strict Artin extractor refuses the quantized x ties. The link invariant follows from the certified homotopy, not from an attempted extraction of the rounded path.

## 5. What must be externally proven for real RMP activation data

A global derivative bound can in principle be derived from an interval/Jacobian verifier, operator-norm and smooth-activation bounds on a bounded input domain, or verified interval arithmetic through the fixed checkpoint. A numeric gradient sampled at a finite set of angles is an empirical diagnostic only. Claims must pin checkpoint hash, dictionary hash, chosen feature axes, readout chart, token positions, injection site, angular intervals and exact model variant. SCM-05 accepts externally supplied claims as assumptions, records their provenance, and labels every resulting pass `conditional_pass` until the bound derivation has been separately verified.

RMP's sealed G0, data splits, training code and checkpoints stay untouched. A future source-locked record should compare the sufficient bit knee, 32/64 angular stability, prior paired winding/Alexander/Milnor readouts and causal counterfactual answer changes on calibration data. Held-out neural results require a new registration and evidence.

## 6. Mathematical literature

The use of configuration space and pure-braid abelianization is classical; see a review in *Braids, Homotopy and Homology*, CRM Barcelona (2022), https://www.crm.cat/wp-content/uploads/2022/06/bookQ-copy.pdf. For the distinction between ordinary pair linking and Milnor triple linking, see Mellor and Melvin, *The Milnor triple linking number of string links by cut-and-paste topology*, Algebraic & Geometric Topology 14 (2014), https://msp.org/agt/2014/14-2/agt-v14-n2-p20-s.pdf . The interpolation and quantization lemmas above are elementary derivations specialized here to measured activation-chart paths, not claims of an existing theorem attributed to either work.
