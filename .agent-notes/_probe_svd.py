"""Does torch.linalg.svd dispatch counted matmuls? That would explain the residual."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__


def cnt(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


N = 32
a = torch.randn(N, N)
print("svd (32,32)          = {:.6e}".format(cnt(lambda: torch.linalg.svd(a, full_matrices=False))))
print("svdvals (32,32)      = {:.6e}".format(cnt(lambda: torch.linalg.svdvals(a))))
b = torch.randn(N, 22)
print("svd (32,22)          = {:.6e}".format(cnt(lambda: torch.linalg.svd(b, full_matrices=False))))
print("svd (32,23)          = {:.6e}".format(cnt(lambda: torch.linalg.svd(torch.randn(32, 23), full_matrices=False))))
print("var(dim=0)           = {:.6e}".format(cnt(lambda: a.var(dim=0))))
print("T@T                  = {:.6e}".format(cnt(lambda: a.T @ a)))
print("diag                 = {:.6e}".format(cnt(lambda: torch.diag(torch.diag(a)))))
print("1x32 @ 32x1          = {:.6e}".format(
    cnt(lambda: torch.randn(1, 32) @ torch.randn(32, 1))))
