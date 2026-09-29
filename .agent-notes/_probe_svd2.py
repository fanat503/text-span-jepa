"""Find the unmodelled (32,32)@(32,32) x17 -- suspect a batched SVD/QR path."""

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


a2 = torch.randn(32, 32)
b2 = torch.randn(2, 32, 32)
print("svd 2-D (32,32)          = {:.6e}".format(cnt(lambda: torch.linalg.svd(a2, full_matrices=False))))
print("svd batched (2,32,32)    = {:.6e}".format(
    cnt(lambda: torch.linalg.svd(b2, full_matrices=False))))
print("svdvals batched (2,32,32)= {:.6e}".format(cnt(lambda: torch.linalg.svdvals(b2))))
print("eigh (32,32)             = {:.6e}".format(cnt(lambda: torch.linalg.eigh(a2))))
print("qr (32,32)               = {:.6e}".format(cnt(lambda: torch.linalg.qr(a2))))
print("pinv (32,32)             = {:.6e}".format(cnt(lambda: torch.linalg.pinv(a2))))
print("lstsq (32,32)            = {:.6e}".format(cnt(lambda: torch.linalg.lstsq(a2, a2))))
print("solve (32,32)            = {:.6e}".format(cnt(lambda: torch.linalg.solve(a2, a2))))
print("cholesky (32,32)         = {:.6e}".format(cnt(lambda: torch.linalg.cholesky(a2 @ a2.T + 32 * torch.eye(32)))))
print("matrix_exp               = {:.6e}".format(cnt(lambda: torch.linalg.matrix_exp(a2))))
print()
print("cross_entropy            = {:.6e}".format(
    cnt(lambda: torch.nn.functional.cross_entropy(torch.randn(6, 97), torch.randint(0, 97, (6,))))))
print("smooth_l1                = {:.6e}".format(
    cnt(lambda: torch.nn.functional.smooth_l1_loss(a2, a2))))
print("log_softmax              = {:.6e}".format(
    cnt(lambda: torch.log_softmax(torch.randn(6, 97), dim=-1))))
print("argmax                   = {:.6e}".format(cnt(lambda: torch.randn(6, 97).argmax(dim=-1))))
