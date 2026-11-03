"""CPU-only toy interface checks for Algernon. Not Qwenputer or Qwen graft evidence."""
from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import numpy as np
import torch
from torch import nn


def u32(x):
    return int(x) & 0xffffffff


def i32(x):
    return ctypes.c_int32(u32(x)).value


class ToyRV32:
    """Independent software-only SUBSET of an RV32 instruction interface."""
    def __init__(self, bytes_count=128):
        self.mem = bytearray(bytes_count)
        self.reg = [0] * 8
        self.pc = 0
        self.instructions = 0

    def run(self, program, limit=100):
        while self.pc < len(program):
            if self.instructions >= limit:
                raise RuntimeError('instruction limit')
            op, *arg = program[self.pc]
            jumped = False
            if op == 'li':
                rd, imm = arg
                self.reg[rd] = u32(imm)
            elif op in {'add', 'sub', 'mul', 'xor'}:
                rd, a, b = arg
                if op == 'add': value = self.reg[a] + self.reg[b]
                elif op == 'sub': value = self.reg[a] - self.reg[b]
                elif op == 'mul': value = self.reg[a] * self.reg[b]
                else: value = self.reg[a] ^ self.reg[b]
                self.reg[rd] = u32(value)
            elif op in {'sw', 'lw'}:
                a, off = arg
                if not 0 <= off <= len(self.mem) - 4 or off % 4:
                    raise ValueError('memory bounds/alignment')
                if op == 'sw':
                    self.mem[off:off+4] = struct.pack('<I', self.reg[a])
                else:
                    self.reg[a] = struct.unpack('<I', self.mem[off:off+4])[0]
            elif op == 'beq':
                a, b, target = arg
                if not 0 <= target < len(program):
                    raise ValueError('invalid branch target')
                if self.reg[a] == self.reg[b]:
                    self.pc = target
                    jumped = True
            elif op == 'halt':
                self.instructions += 1
                break
            else:
                raise ValueError(f'unknown instruction {op}')
            self.reg[0] = 0
            self.instructions += 1
            if not jumped:
                self.pc += 1
        return self

    def state_receipt(self):
        state = b''.join(struct.pack('<I', x) for x in self.reg) + bytes(self.mem) + struct.pack('<I',self.pc)
        return {'instruction_count': self.instructions, 'state_after_sha256': hashlib.sha256(state).hexdigest(), 'registers': self.reg.copy()}


class FrozenToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(6, 12), nn.Tanh())
        self.tail = nn.Linear(12, 4)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.tail(self.pre(x))


class ToyGraft(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.P = nn.Linear(12, 8)
        self.F = nn.Linear(20, 8)
        self.Q = nn.Linear(8, 12)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, K):
        assert K in (0, 1, 2, 4, 8)
        h = self.base.pre(x)
        z = self.P(h)
        for _ in range(K):
            z = torch.tanh(self.F(torch.cat((z,h), -1)))
        h2 = h + torch.clamp(self.gate, -0.2, 0.2) * self.Q(z)
        return self.base.tail(h2)


def run_smokes(seed=20261103, trials=1000):
    rng = np.random.default_rng(seed)
    failures = 0
    last = None
    for _ in range(trials):
        a, b = [int(x) for x in rng.integers(-(2**31), 2**31, size=2, dtype=np.int64)]
        program = [('li', 1, a), ('li', 2, b), ('add', 3, 1, 2), ('sub', 4, 1, 2), ('mul', 5, 1, 2), ('sw', 3, 0), ('lw', 6, 0), ('halt',)]
        vm = ToyRV32().run(program)
        expected = [u32(ctypes.c_int32(x).value) for x in (a+b, a-b, a*b)]
        if [vm.reg[i] for i in (3,4,5,6)] != [*expected, expected[0]]:
            failures += 1
        last=vm
    torch.manual_seed(seed)
    base = FrozenToyBase()
    graft = ToyGraft(base)
    inputs = torch.randn(64,6)
    no_graft = base(inputs)
    zero_identity = all(torch.equal(no_graft, graft(inputs,K)) for K in (0,1,2,4,8))
    with torch.no_grad(): graft.gate.copy_(torch.tensor(0.1))
    k_changes_output = not torch.equal(graft(inputs,1), graft(inputs,8))
    target = torch.randint(0,4,(64,))
    optimizer = torch.optim.Adam([p for p in graft.parameters() if p.requires_grad],lr=.001)
    objective = nn.CrossEntropyLoss()(graft(inputs,4),target)
    objective.backward()
    gradients_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in graft.parameters() if p.requires_grad)
    optimizer.step()
    return {'origin': 'toy_cpu_contracts_only', 'arithmetic_trials': trials, 'reference_mismatches': failures, 'last_state_receipt': last.state_receipt(), 'reversible_bridge_zero_gate_exact': zero_identity, 'recurrence_output_changes_with_K': k_changes_output, 'gradient_finite': gradients_finite, 'training_loss_before_first_step':float(objective.item()), 'trainable_parameters': sum(p.numel() for p in graft.parameters() if p.requires_grad)}


if __name__ == '__main__':
    print(json.dumps(run_smokes(),indent=2))
