# PSS-01: protected-subspace weight surgery

Status: implemented CPU reference with synthetic tests. Pretrained-model behavioral evaluation is pending. No rented GPU was used for this package.

This adds a self-contained operator alongside `research/algernon/mono01/`. MONO-01 handles minimum-energy keyed corrections under a reference covariance; PSS-01 handles removing writes into chosen output directions while protecting specified output and input subspaces. Existing research files remain unchanged.

## Run

Requires Python 3.10+ and NumPy. From the repository root:

```sh
python -m unittest discover -s research/weight_surgery_pss01 -p 'test_*.py' -v
python research/weight_surgery_pss01/demo.py > pss01_cpu_receipt.json
```

`cpu_receipt.json` summarizes the initial synthetic run and includes the full rank-4 case. The demo regenerates every case with effective rank and realized BF16 storage errors. The test suite includes a separate least-squares solution of the vectorized constraints, a 25-seed sweep, and degenerate-subspace checks.

## Exact operator

Let W have shape m by n. Let S and H have orthonormal columns representing protected output directions and protected receiver-input directions. Define R = I - H H^T. Let Q have orthonormal columns spanning the retained part of (I - S S^T) D, where D contains candidate failure directions. Thus S^T Q = 0. All spaces refer to the same pinned linear receiver.

For 0 <= alpha <= 1:

```
Delta* = -alpha Q Q^T W R
W'     = W + Delta*
B      = -alpha Q
A      = Q^T W R
Delta* = B A
```

The implementation applies both projectors through thin factors, without constructing square identity/projector matrices. A rank-r failure space gives rank(Delta*) <= r and a LoRA-shaped update with r(m+n) stored factor entries. The merged matrix can have changes in every entry. BF16 merging can increase the numerical rank of the realized difference.

### Minimum-change theorem

Among real matrices Delta satisfying

```
S^T Delta = 0
Delta H   = 0
Q^T Delta R = -alpha Q^T W R,
```

Delta* uniquely minimizes ||Delta||_F.

Proof. Delta H = 0 implies Delta = Delta R. Every feasible Delta therefore has Q^T Delta = -alpha Q^T W R. Write Delta = Delta* + Z. It follows that Q^T Z = 0, while every column of Delta* belongs to span(Q). Hence their Frobenius inner product is zero and

```
||Delta||_F^2 = ||Delta*||_F^2 + ||Z||_F^2.
```

Equality holds exactly when Z = 0. Delta* satisfies the output-protection constraint because S^T Q = 0, and the input constraint because R H = 0. This proves feasibility and uniqueness.

### Scope and feasibility floor

Output protection means preserving selected linear output coordinates for every fixed input. Input protection additionally preserves the full linear output on span(H). A composed network changes its downstream inputs after an edit; whole-model utility requires held-out measurement.

At full strength:

```
Q^T W' = Q^T W H H^T
||Q^T W'||_F = ||Q^T W H||_F.
```

Any edit satisfying Delta H = 0 retains that component. Global removal of Q and exact preservation on H are compatible precisely when Q^T W H = 0. The receipt exposes this floor. If D lies entirely in span(S), the constructor returns a rank-zero no-op. A cutoff relative to the ORIGINAL D scale prevents floating-point projection debris becoming a spurious normalized failure direction.

For a unit direction d and unit protected direction s, the retained energy is 1 - (s^T d)^2. At cosine -0.77 this is 0.4071. The three-dimensional identity example in `demo.py` uses this geometry; its numbers describe a constructed matrix.

### Rounding and factor execution

For a realized merged matrix W_hat = W' + E, the protected-output error is S^T E and the protected-input error is E H. Each Frobenius norm is at most ||E||_F. The target-constraint error is Q^T E R, also bounded by ||E||_F. `audit` evaluates these errors on the supplied tensor and records hashes of source and candidate.

`bf16_round` models nearest-even storage conversion from FP32. It excludes hardware accumulation and device subnormal policies. The initial implementation was also compared against PyTorch 2.10.0+cpu BF16 conversion on 32,768 finite random FP32 values with exact equality. Regenerate this check on the execution stack before reporting a native-kernel result.

Biases remain unchanged. A one-sided branch hook `y <- (I-alpha Q Q^T)y` matches a weight-only edit only when the receiver has zero bias; otherwise subtract/re-add the unchanged bias or use `y <- y + B(Ax)`. A whole-residual-stream projection also changes the skip path and constitutes a separate intervention. Two-sided surgery requires access to the receiver input x.

## Control-Harness interface

The core module has no repository-specific imports. Import `surgery.py` by an explicit pinned path, or vendor it with its commit and SHA-256 recorded. Plan construction runs on CPU in FP64. Export B and A to the runtime device/dtype only after checking the target tensor's name, orientation, shape and source hash.

The runtime operation is `y = linear(x) + (x @ A.T) @ B.T` for batch-major activations. Use effective LoRA scale 1; alpha is already contained in B. Receiver inputs are the post-gating/post-normalization values consumed by that exact linear map. Hook at that location, including in hybrid recurrent blocks.

Record the source revision, exact module key and hashes with each patch. Preserve original weights for rollback. Re-audit the actual merged tensors and compare teacher-forced outputs of hook, unfused factors and merged weights before free generation. Export via NumPy arrays with pickle disabled or an existing safe tensor format. The integration contract is supplied here; a live Control-Harness model adapter remains to be connected.

## Evidence and sources

Muninn's November 3, 2026 Monad report motivates the protected output space. It reports a sink-orthogonal projection at attention output matrices, selected on a separate design set. The source is a reported model experiment; the present package supplies algebra and CPU diagnostics.

- https://muninn.austegard.com/blog/editing-monad-weights-to-stop-loops.html
- Existing MONO-01 source at base commit ff1579130b936363af35c8481c076e423a6143b7: `research/algernon/mono01/monoshot.py`
- NumPy SVD API: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
- Planned larger-model evaluation: https://huggingface.co/Qwen/Qwen3.5-9B

`SILICO_WORKORDER.md` defines the proposed GPU experiment. Posting this package has not started that experiment.
