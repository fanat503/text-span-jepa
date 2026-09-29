"""Decisive: does FlopCounterMode count all backward matmuls, or fewer?"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

M, K, N = 64, 32, 128
lin = nn.Linear(K, N)
x = torch.randn(M, K, requires_grad=True)

with FlopCounterMode(display=False) as f:
    y = lin(x)
fwd = f.get_total_flops()

# bwd ONLY inside the mode
with FlopCounterMode(display=False) as f2:
    y.sum().backward()
bwd_only = f2.get_total_flops()

print("fwd            =", fwd)
print("bwd (alone)    =", bwd_only, " ratio bwd/fwd =", bwd_only / fwd)

lin.zero_grad(set_to_none=True)
y = lin(x)
with FlopCounterMode(display=False) as f3:
    y.sum().backward()
print("again          =", f3.get_total_flops())

# both inside
lin.zero_grad(set_to_none=True)
with FlopCounterMode(display=False) as f4:
    lin(x).sum().backward()
print("both inside    =", f4.get_total_flops(), " ratio =", f4.get_total_flops() / fwd)

# stack of linears, both inside
stack = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
z = torch.randn(16, 64, requires_grad=True)
with FlopCounterMode(display=False) as fa:
    stack(z)
ff = fa.get_total_flops()
with FlopCounterMode(display=False) as fb:
    stack(z).sum().backward()
print("\nstack fwd      =", ff, " stack bwd alone =", fb.get_total_flops(),
      " ratio =", fb.get_total_flops() / ff)
with FlopCounterMode(display=False) as fc2:
    stack(z).sum().backward()
print("stack both     =", fc2.get_total_flops(), " ratio =", fc2.get_total_flops() / ff)
