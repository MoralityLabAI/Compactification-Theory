# OIC-31: Exact mixed-fault prime-cycle sensing

**Executed 11 November 2026.** The OIC-30 seven-phase prime-block construction and its conditional sensor-omission lower bound now extend to a mixed channel allowing `e` substituted report bits and `r` marked erasures.

The exact theorem is `Q* = p(2e+r+1)` for fixed policies over prime cycles `p` using fewer than `p` distinct sensor labels, provided `2<=2e+r+1<p`. The lower bound applies for all cycle sizes `n>=2` under marker omission. These qualifications are part of the theorem's scope.

The example `p=7,e=1,r=1` requires **28 queries across four marker labels**, giving exact minimum cross-shift distance four. An independent finite check covers every different-shift phase pair; the Hamming distance condition ensures correction for every placement of the allowed faults.

The new `oic31_mixed.py` implements the construction, lower-bound witness and explicit ambiguous corrupted receipt. Six unit tests pass both with ordinary Python and `python -O`. The pinned JSON reference contains eight prime/fault configurations. No GPU, training or learned hidden-state model is involved.

The earlier unrestricted seven-marker Q20 question remains unresolved by a proof certificate. The Q20 SMT attempt returned no verdict before its execution cap. The registered OIC-31 fixed-schedule proof-certificate and adaptive Q19 tasks retain their open status.

Reproduce:

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic31_mixed.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic31_mixed.py' -v
python research/observational_inwardness_v1/oic31_run.py --check
~~~
