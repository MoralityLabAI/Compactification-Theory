# ROC-QWEN-01: RTX 3050 4GB

Frozen Qwen3.5-0.8B-Base text-only NF4/FP16 residual observer experiment.
Read PREREGISTRATION.md before interpreting results. The allocator cap is decimal
3.2 GB, below the decimal 3.4 GB requested ceiling. Device UUID is audited before
CUDA_VISIBLE_DEVICES is assigned. Only the RTX 3050 Laptop 4096 MiB is allowed.

Run from repository root in WSL2 (CUDA 12.1 PyTorch wheel, Python 3.10):

```sh
python3 -m pip install -r experiments/roc_qwen01_3050/environment.lock.txt
python3 -m unittest discover -s experiments/roc_qwen01_3050/tests -v
python3 experiments/roc_qwen01_3050/smoke.py
python3 experiments/roc_qwen01_3050/campaign.py
```

Downloads use only the named immutable checkpoint, range-resume on disk, and
verify its LFS SHA256. No vision encoder is instantiated. Original safetensors
may contain vision tensors on disk. The installed Transformers text-only
checkpoint conversion is checked for missing or mismatched parameters.

Forward cache receipts and selected-token NumPy memmaps prevent matching passes
from being repeated. Cached execution requires the exact source fingerprint and
settings. Delete nothing to rerun CPU fitting: `python3 experiments/roc_qwen01_3050/fit.py`.
The published compressed receipt database can be restored with
`python3 experiments/roc_qwen01_3050/restore_forward_cache.py` before replay.
Audit saved results without a model using
`python3 experiments/roc_qwen01_3050/verify_results.py`.

`results/REPORT.md`, `summary.json`, row CSVs, manifests, and memory logs carry
the evidence. Sensitivity sketches have only 16 measured directions; unmeasured
orthogonal sensitivity remains unknown. Residual donor routing is a declared
isometry and is not a measurement of recurrent A/B/C. There is no full ROC card
or T-B neural closure claim. The pure readout-ablation gate remains unsupported;
centered residual erasure is a separate exploratory arm.
