# Communication rank frontier v1

Canonical implementation and finite receipts: https://github.com/MoralityLabAI/LoopedTransformers/tree/6b72a5cf78135c746cefcf444921098d087d691f/experiments/logrank_bpd_01. Cogentic's claimed exponent improvement motivates this note; it is not used as an empirical bound.

For a distributed Boolean reasoning task M(s,a), distinguish local representation rank r=rank_R M, one-way worst-case bits B1=ceil(log2 distinct rows), interactive worst-case bits B*, local recurrent depth T, local parameter counts P, local precision p, local peak working memory H and compute F. For Bob-only output (Bob holds a), each successful deterministic leaf is rank at most one on its rectangle, hence r<=2^B*. Thus ceil(log2 r)<=B*<=B1 for nonzero M. Distinct rows must receive distinct one-way messages. Both bounds concern exact worst-case correctness, not average accuracy.

## An unbounded separation

Index_k(x,j)=x_j for x in {0,1}^k, j in {1,...,k}. Its rank is k: its rows contain the k unit vectors and it has k columns. Its distinct row count is 2^k, so B1=k. Interactive protocol: Bob sends j in ceil(log2 k) bits and Alice sends x_j, giving B*<=ceil(log2 k)+1. Thus rank alone cannot imply a polylogarithmic one-way deterministic bound even though interaction can attain logarithmic cost on this family. The LoopedTransformers receipt verifies Index4, 4 bits one-way vs 3-bit explicit interactive protocol.

Rank, local dimensionality, storage and communication are separate resources. Introduce feasible set C_epsilon(M)={(P,p,T,H,F,B): protocol accuracy >=1-epsilon under named distribution}. For exact results use epsilon=0 and require all-pair replay; Pareto-order tuples with matched output semantics and compute scope. A rank witness does not prove a local state dimension lower bound. A finite neural training result does not bound general communication complexity.

Proposed benchmark: matched fixed-message-cap one-way vs recurrent alternating protocol, fresh generator roots, exact classical solver/upper bounds where available, all training and inference resource accounting. Distinguish interactive schedule from more local computation by matching local recurrence steps and FLOPs. Keep public constants fixed and charge any input-dependent memory transfer, selector or early stop. Report bit counts separately from local RAM; increased RAM may support caching without increasing communication. No native model training or frontier claim is made here.
