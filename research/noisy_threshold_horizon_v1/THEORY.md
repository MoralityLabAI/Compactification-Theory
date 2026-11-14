# NHB-01: noise horizon, threshold complexity and decision memory

Date: 2026-11-14. This campaign pairs OpenAI Mathematics #119 and #127. The general source results are imported research claims. The deductions below have supplied proofs and finite computational checks; their novelty relative to the wider literature has not been established.

## Source statements and proof status

**#119: Sharp binary-information contraction on the discrete cube**, September 24, 2026. Scope: https://github.com/openai/math/blob/main/lean/docs/119.md . The paper's main theorem, read in build/main.tex, considers a uniform sign cube X and any randomized binary variable U generated from X. Independent coordinate noise produces Y with correlation rho. Define psi_2(a)=1-h_2((1-a)/2), for 0<=a<=1, in bits. The claimed sharp result is

    I(U;Y) <= psi_2(|rho| * psi_2^{-1}(I(U;X))).

A noisy coordinate U attains equality at each initial information level. This includes the deterministic Courtade-Kumar bound I(f(X);Y)<=1-h_2(p), p=(1-rho)/2. The supplied SoftChannel204 comparator contains an admitted fullMain interface; that comparator alone is not an independent proof certificate. We have inspected the source statement and its scope, rather than rechecked the entire original proof.

**#127: Average sensitivity of polynomial threshold functions**, September 25, 2026. Scope: https://github.com/openai/math/blob/main/lean/docs/127.md . For n>=1 and 1<=d<=n, the reported formalized theorem bounds the average sensitivity of any degree-at-most-d real multilinear polynomial threshold function by 8*d*sqrt(n), with sign(0)=+1. This is an upper bound using a threshold representation's degree. It is distinct from the real Fourier degree of the resulting Boolean truth table. For example, seven-bit majority has threshold degree one and real Fourier degree seven.

Source snapshot identified in the repository search: openai/math@4eaee87548250b2682affda7d05d947c05554bae. Scope-file Git blob SHAs: #119 e8de61a6f182c52430a35f4792c84eba2b14e08d; #127 7ed32b1d857ef786dde308593b470c9d53b78427. The #119 main TeX blob is 805e65be20856ee37f0987d026b612d80efeb00a. No source file is modified by this campaign.

## Interface and random variables

X_0 is uniform on {-1,+1}^n. At every step, each state coordinate flips independently with probability epsilon in [0,1/2]; all flips are independent across coordinates, time and the original source. A fixed sign function f defines the target U=f(X_0)*Z, where Z is independent and Pr(Z=-1)=eta in [0,1/2]. The parameter eta represents initial target uncertainty and is not a trained model's observed error rate.

The terminal observer sees X_T, or a fixed encoded representation M=E(X_T), and independent randomness. It receives no stored copy of U or X_0 and no earlier trajectory. Static encoders, posterior tables and arithmetic must be charged separately from mutable output-state bits. A monitor that observed and retained U at time zero can recover it perfectly in one bit; that interface lies outside this terminal-only setup.

## NHB-1: exact noisy recurrence

Let rho=1-2*epsilon. The product of the independent coordinate noise signs has mean rho^T, so the entire T-step transition is a binary symmetric channel with crossover

    p_T = (1-rho^T)/2.

Two crossovers compose as p*q=p+q-2pq. These statements hold exactly at the full-vector level because coordinates remain independent. The reference checks the factorized transition against explicit enumeration and repeated kernel application.

For sign f with Fourier coefficients fhat(S), orthogonality gives

    NS_p(f) = Pr(f(X)!=f(Y))
            = (1/2) sum_S [1-(1-2p)^|S|] fhat(S)^2.

Its total influence is AS(f)=sum_S |S| fhat(S)^2. Bernoulli's inequality gives NS_p(f)<=p AS(f). Nonnegative Fourier noise weights also give NS_p(f)<=1/2. This derivation is elementary and independent of #119/#127.

## NHB-2: the information and error envelope

Put I0=I(U;X_0). Source #119 and data processing imply, for any terminal encoding M,

    I(U;M) <= psi_2(rho^T * psi_2^{-1}(I0)).

This is the conditional contribution from #119. It allows randomized U and arbitrary output bias.

When f is balanced, U is balanced and I0=1-h_2(eta). With a=1-2eta and q_T=eta*p_T=eta+a p_T, the bound becomes

    I(U;M) <= 1-h_2(q_T).

Binary Fano then gives optimal terminal error e*(M)>=q_T. There is also an independent elementary proof of this error bound: E[U|X_T]=a*T_(rho^T)f(X_T), and balanced f has no constant Fourier coefficient. Thus the squared L2 norm of that conditional mean is at most a^2 rho^(2T). Cauchy-Schwarz bounds its mean absolute value by a rho^T, yielding e*(X_T)>=(1-a rho^T)/2=q_T. Any further terminal encoding has at least this error. The sharp information ceiling is stronger than that elementary error statement.

Coordinate f attains both bounds when its noisy coordinate is observed. Initial bias requires the general information envelope; the balanced error floor cannot be applied unchanged to a rare-event target.

## NHB-3: a two-paper guarantee for an approximate low-degree monitor

Let h=sign(P) be a degree-at-most-d polynomial threshold approximation to a balanced target f. Its uniform same-state error is alpha=Pr(h(X_0)!=f(X_0)). Suppose h(X_T) can be recovered from the terminal code M. Set

    A = min(n, 8*d*sqrt(n)),
    D_T = min(1/2, eta + (1-2eta) * [alpha + min(1/2,p_T*A)]).

Conditional on #119 and #127,

    q_T <= e*(M) <= D_T,
    1-h_2(D_T) <= I(U;M) <= 1-h_2(q_T).

Proof of the upper error bound: by a union bound,

    Pr(f(X_0)!=h(X_T))
      <= alpha + Pr(h(X_0)!=h(X_T))
      <= alpha + min(1/2,p_T AS(h)).

Independent flipping of the target multiplies this error by 1-2eta and adds eta. Source #127 supplies AS(h)<=8d sqrt(n), while AS(h)<=n always. The optimal decoder can also guess a constant at error 1/2, which explains the outer minimum. Fano gives the information lower bound. Source #119 gives the information upper bound, as above.

Whenever AS(h) is known exactly, substitute that value for A without relying on #127. The parameter-only source bound is often loose on small cubes. At n=129,d=1, its upper bound is approximately 90.8625, below the trivial 129. Exact majority sensitivity there is approximately 9.07981. This finite example checks the source statement at one point and does not establish the general theorem.

A second exact approximation certificate compares the direct and approximate readouts:

    |Pr(U!=h(X_T))-Pr(U!=f(X_T))| <= (1-2eta)*alpha.

To prove it, condition on X_T. Its distribution is uniform, and the target conditional mean has absolute value at most 1-2eta. The two prediction signs differ only on the alpha fraction of inputs where f and h disagree.

## NHB-4: sharp terminal-observation horizon

For 0<epsilon<1/2, 0<=eta<tau<1/2, balanced target recovery with error at most tau requires

    (1-2epsilon)^T >= (1-2tau)/(1-2eta).

Therefore

    T <= log((1-2tau)/(1-2eta)) / log(1-2epsilon).

For integer horizons take the floor. The necessary bound is attained by a coordinate target. It follows from the elementary error argument, and its information counterpart is supplied by #119.

At epsilon=1/20, eta=1/10, tau=1/4:

    q_4 = 5939/25000 = 0.23756,
    q_5 = 65951/250000 = 0.263804.

Thus four steps still permit 75% accuracy for an optimal coordinate target, while five steps exclude it for every balanced target and every terminal-only decoder. This conclusion does not constrain monitors with a retained target bit or a trajectory-dependent side channel.

## NHB-5: degree-sensitive Fourier refinement

For balanced f, let beta be its Fourier mass on singleton supports. All other nonzero supports have size at least two. For r=1-2p,

    E[(E[U|Y])^2] <= (1-2eta)^2 [beta*r^2+(1-beta)*r^4].

Consequently

    e*(Y) >= (1-(1-2eta)*sqrt(beta*r^2+(1-beta)*r^4))/2.

A homogeneous k-bit parity has the sharper exact result

    e*(Y) = [1-(1-2eta)*r^k]/2.

These are elementary spectral deductions. PTF representation degree and Fourier degree must remain distinct when using them. For the parity control the two degrees happen to coincide; for majority they do not.

## NHB-6: the decision interface determines the compressed memory

For any finite observation Y and binary target U, let p(y)=Pr(U=+1|Y=y). The single output bit B(y)=1{p(y)>=1/2} preserves the best possible zero-one classification accuracy. Within each B-cell, the sign of p(y)-1/2 is constant, so replacing y with B(y) leaves the integrated Bayes error unchanged. Computing this bit can still require the entire state and a large posterior table.

Now require a deterministic code to answer every posterior-threshold query 1{p(y)>=t}, t in [0,1], exactly. If K distinct posterior values occur on positive-probability observations, that code requires at least K states, hence ceil(log2 K) fixed-length bits. Two different values are separated by a threshold strictly between them, so they cannot share a code. Encoding the posterior-value index attains the bound and also preserves all information about U.

At the registered four-step noise condition, seven-bit majority has eight distinct posterior values. One bit retains its optimum binary classification error. Three bits are necessary and sufficient for the full threshold-query interface. The exact information in the full state is about 0.151222 bits; its best-decision bit retains about 0.119789 bits. The code's output width, semantic query interface and computational cost are therefore measured separately.

## NHB-7: high agreement without information

Let f be the seven-input conjunction represented by sign(sum_i x_i-6). The positive event has probability 1/128. At crossover p=1/2 and eta=0, X and Y are independent, so I(f(X);Y)=0 exactly. Nevertheless

    Pr(f(X)=f(Y)) = 1-2*(1/128)*(127/128)
                  = 8065/8192, approximately 98.4497%.

The optimal prediction is a constant, with error 1/128. This is an exact control for apparent monitor stability caused by a biased target. The benchmark records target priors and information alongside raw agreement.

## Experimental scope and current AI work

The executed source models are complete finite Boolean cubes with specified polynomial threshold representations. Their transitions and Bayes errors use exact rational arithmetic. Shannon entropies and inverse-entropy envelopes are computed numerically with explicit tolerances. The source theorems' asymptotic proofs are not reproduced by those finite checks.

The application to recurrent model monitoring is a registered test design: identify a binary feature interface, measure its initial distribution, verify independent symmetric state corruption, and account for all history or persistent memory before invoking the horizon result. For an SAE or low-degree probe, measure alpha on an independent population and account for sampling uncertainty before substituting it into NHB-3. Neural trajectories that violate those assumptions require a different channel law, as the preceding KSR native-state audit illustrated.

A proposed finite readout stores one output bit, but its truth table, polynomial coefficients, feature computation and observation bandwidth are separate costs. This round evaluates no trained transformer or GRU checkpoint. The native model tests from earlier campaigns remain unchanged.
