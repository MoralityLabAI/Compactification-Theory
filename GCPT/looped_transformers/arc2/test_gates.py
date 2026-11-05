import numpy as np
import torch
from gates import LoopedGate, GateBank, rows

def test_rows():
    x, y = rows()
    assert x.shape == (64,3) and y.shape == (64,)

def test_parameters():
    assert sum(p.numel() for p in LoopedGate().parameters()) == 9378

def test_exact_control_comparator():
    table = np.array([[(r >> (2*a+b)) & 1 for a in range(2) for b in range(2)]
                      for r in range(16)]).reshape(16,2,2)
    bank=GateBank(table)
    a,b=np.meshgrid(np.arange(64),np.arange(64),indexing="ij")
    pred=bank.leq(a,b,bits=6)
    assert np.array_equal(pred,a<=b)
    result, trace=bank.leq(5,6,bits=6,trace=True)
    assert bool(result)
    assert len(trace)==6

def test_guard():
    table=np.zeros((16,2,2),dtype=np.int64)
    bank=GateBank(table)
    import pytest
    with pytest.raises(ValueError):
        bank.gate(0,np.array([2]),0)
