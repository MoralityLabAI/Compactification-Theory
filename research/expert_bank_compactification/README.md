# EBC-01: resource-relative expert-bank compactification

Status: new finite mathematics and executed synthetic oracle, 2026-11-27. This extends [RC-01](../recurrent_circuits_v1/THEORY.md), [IRC-02](../../experiments/interface-relative-compactification/README.md), [CGC-01](../causal_gauge_compactification/THEORY.md) and [Inkling depth/precision](../inkling_depth_precision/README.md). [InkyNano INK-CHANT-01](https://github.com/MoralityLabAI/LoopedTransformers/pull/417) owns native integration. No weights loaded and no new native task-quality or latency claim. Nothing changes earlier NOR or other preregistered verdicts.

## 1. Interface and exact static support

For each full native state h, public visit coordinate l (physical layer), and fixed arithmetic/tie convention, let A_l(h) be the original top-k expert IDs. Inkling has k=8, 128 original IDs per layer, 24 physical layers, and 48 applications per token. The native state includes every required hidden, attention, loop, and persistent cache value. A support is a set of eligible IDs; residency is a different set. The full native router sees h. We ask only that the outer support controller admit every demanded ID; it need not compute the native router.

Fix a finite declared execution domain and take the union of native demands per physical layer. This union is the unique inclusion-minimal static support preserving every original selected ID along those executions. Necessity: any omitted union member is demanded on some visit. Sufficiency: restricting the ranking to a superset of the top-k leaves the selected IDs and normalized unbiased mixture scores unchanged (fixed total tie order), and with identical weights and arithmetic it leaves the next state unchanged. Induct on the complete execution. This is conditional on exact source identity and runtime parity, and it is necessary for exact route preservation rather than merely final-answer equality. No empirical finite trace establishes universal support coverage.

## 2. Capacity/state trade-off

Take a finite successor-closed deterministic state domain X with public action alphabet U. The one-layer demand map is A:X -> subsets of [n]. The outer controller must have an autonomous update q(F_a(x))=G_a(q(x)), with the native router still seeing x. For any partition P of X, a support attached to a class C must contain the union U(C) of A(x) for x in C. Hence the partition is feasible under logical capacity K exactly when:

1. each class union has cardinality at most K;
2. each action maps all states of a class to one successor class.

Necessity follows from the support and deterministic update contracts. For sufficiency, assign the union to every class and define the successor using any representative; the congruence condition guarantees well-definedness. Exact routes then follow from section 1 at every visit. If N_K is the minimum feasible class count, the minimum fixed-length controller state is ceil(log2 N_K) bits: distinct classes need different codes and the class index attains the bound. This is a live-state cardinality result with fixed side information. Charge the encoder, support table, transition program, full native state, and action/clock disclosure separately. A host-supplied class ID is charged input, not free memory. A feasible partition exists whenever K is at least the largest individual demand, using singleton classes. N_K is nonincreasing in K. With all demands of size k and K=k, it equals the ordinary future-demand equivalence class count; with K at least the global union size it is one. Public layer or pass coordinates may index separate supports and transition tables; the executable oracle handles a single public mode/layer.

The optimum need not be a unique coarsest partition. Pairwise merge compatibility does not imply joint feasibility: three disjoint singleton demands with K=2 can be merged pairwise, but their triple union is too large. Successor congruence imposes further delayed constraints. The restricted-growth enumeration in reference.py searches every partition of at most ten states, pruning monotonically on capacity and incumbent class count; it is an exact small oracle, not a neural encoder or scalable clustering method.

Fixture: demands {a},{a},{b},{c}, successors 2,3,2,3 under one action. Capacity 1 needs four classes (two live bits), capacity 2 needs two (one bit), capacity 3 needs one (zero bits). The first two states have identical current demand but different future demand. The model's full native state still exists in all three arms.

## 3. Complete-route coverage is a hypergraph objective

On weighted teacher visits i with demand edges A_i and nonnegative weights w_i, the exact static budget objective is maximize sum_i w_i * 1[A_i subset S], subject to |S|<=K. A visit weight may include measured transfer costs or task-specific weighting fixed before evaluation. This is weighted induced hyperedge coverage, a monotone supermodular set function, generally not submodular. Demand mass or marginal expert frequency optimizes a different objective.

Counterexample: routes AB four times, AC three times, DE five times. With capacity three, individual frequency selects A,D,E and preserves five complete visits. A,B,C preserves seven and is optimal by exhaustive enumeration. General k=2 contains the densest-K-subgraph objective, so the exact small search is unsuitable for full-scale training. MILP with binary expert variables z_e, binary edge variables y_i, constraints y_i<=z_e for e in A_i, and sum z_e<=K is an offline oracle if time and size permit. The objective is measured on teacher traces; after a miss, downstream actual routing can diverge, so coverage alone is not an answer-quality guarantee.

## 4. Exact local mixture deviation and conditional depth envelope

Let the unrestricted and masked top-k sets share total unbiased sigmoid mass C; let A denote removed mass and B added mass. At the same hidden state, scores for shared experts are identical. With positive total mass, the probability-normalized mixtures p and q have total variation

    TV(p,q) = 1 - C/max(C+A,C+B)
            = max(A/(C+A), B/(C+B)).

Proof: half the L1 difference equals one minus the sum of coordinatewise minima. On common IDs the smaller weight is each common score divided by the larger normalizer, and all other minima vanish. Bias enters ranking only. Thus dropped native mass alone cannot certify mixture error: added mass may dominate.

For expert output vectors v_e at the same input and output diameter D=max_{e,f}||v_e-v_f||, the local difference between the two mixtures with common native scaling g satisfies ||g sum_e(p_e-q_e)v_e|| <= g D TV(p,q). Couple common probability mass identically; the remaining TV mass transports between vectors a distance at most D. This is stronger than the trivial 2g max||v|| TV bound when diameter is small. Exact Fraction tests verify the TV identity on 205 valid integer-mass triples.

At a reference state, if the relevant top-k ranking gap is gamma and ranking perturbation is bounded by eta in sup norm, gamma>2 eta guarantees membership unchanged among allowed IDs containing the native top-k. For logits w_e^T h and sigmoid, eta <= (max_e||w_e||_*) ||delta h||/4 for fixed bias. Floating-point and score-calibration errors require their own slack. Membership preservation alone does not preserve mixture values at perturbed h. The full complete-state transition must have a valid local/tube Lipschitz bound L_j, including residuals, attention, recurrence gates and router discontinuity boundaries. For local error epsilon_j bounded on the reference path and controlled candidate neighborhoods, e_{j+1} <= L_j e_j + epsilon_j. Iteration yields the product/sum precision envelope already defined in the Inkling depth-precision work. Without valid tube bounds use the TV identity as a local diagnostic, not a global numerical certificate.

## 5. Residency compactification and traffic

With per-layer resident capacity C_r>=k and mandatory co-resident native top-k, successive demands A,B at the same physical layer imply a lower bound max(0,|A union B|-C_r) additional admissions between visits, even for a clairvoyant cache (assuming no other requirements). At C_r=k, the post-demand cache is exactly A; the next visit admits |B minus A|. More logical experts may permit a smaller outer controller while increasing miss traffic; the cache is not the controller partition. A fetch preserves native choice whereas a residency-driven replacement changes the model. Charge prefetch, cold starts, disk/host/device transfers, CPU bypass and staging.

## 6. Published-bank support accounting

Source: [INK-CHANT-01](https://github.com/MoralityLabAI/LoopedTransformers/blob/f9e0186a8360d1dca2fdc696875337d795451a42/InkyNano/INK_CHANT_01.md) and [GGUF conversion](https://github.com/MoralityLabAI/LoopedTransformers/blob/76a0fd8930a567b97b26d43d119f3b7b7cdb9ef1/InkyNano/GGUF_CONVERSION.md). Chantless mask revision b532d51f81b031fced1023676656ad263c22b4df, source checkpoint ed40e7da625f594fc9c4990f5f66be24873c22d7. The per-layer intersection counts (Math 96, classifier 48) are 46,44,42,46,45,46,45,41,37,43,42,41,46,43,43,41,43,46,47,44,41,45,44,43. Assuming 4,810,752 packed bytes per early-layer expert and 7,174,656 per late-layer expert (physical layers 20–23), the shared 1,044 experts cost 5,431,380,480 payload bytes. Math extension: 1,260 experts / 6,560,331,264 bytes. Classifier extension: 108 experts / 564,475,392 bytes. Union: 2,412 experts / 12,556,187,136 bytes (11.694 GiB). Union counts vary from 97 to 107 across layers. These are weight-payload estimates before backbone, overhead, KV, runtime and transfer; masks need a shared bank or uniform padded layout, not a claim of standalone native compatibility. Static intersection does not identify sufficient task or recurrent support. The classification mask also changes width and vocabulary; this study imports only support.

## 7. Prospective native gate

Use independent train/dev roots from native full-support quantized traces with pinned bank/teacher/runtime provenance. First check source and expert index parity. Then compare static published supports, random capacity-matched supports, marginal-frequency supports, complete-edge coverage optimized supports, phase-conditioned support with public pass, and learned causal outer controller. Cross each with exact demand paging, independently varying logical capacity and physical residency. For the RTX 3050 start with 8/12/16 resident slots, context 512 smoke before 12k-default/16k-cap stress; do not allocate full logical support on device. For each arm report native route-set matches, mixture TV, task verifier score, cache misses, measured bytes/timing, model description and mutable state bytes, and root-clustered uncertainty. Include exact full support and matched random/shuffled controls. Counterfactual teacher demand replay measures coverage, not off-policy live execution. Apply FSI-01 before promotion; do not open unrelated sealed cohorts.

The optimization target is a Pareto frontier over (bank storage, support description, controller description, controller live bits, native live state, peak host/device memory, transfer bytes, actual latency, verified task error). A mask table can dominate its controller's live bits, so charge it.

## 8. Executed finite evidence

Run: python -m unittest discover -s research/expert_bank_compactification -p test_reference.py -v

Five tests passed locally. Exhaustive independently labelled-map oracle matched the partition solver on 2,187 (3-state unary transition system, singleton demands, capacity) instances; 205 rational mixture-TV cases passed. The capacity frontier and frequency counterexample passed. Tested Git blob hashes: reference.py cdc68e7d2abd0943ba92e9c29d1475e8a1cbfffe; test_reference.py 66ee42bd69a09af7a8ed5260e82b5121bddba254. No native trace, large gym, GPU, cloud spend, formal proof kernel, or full repo test.
