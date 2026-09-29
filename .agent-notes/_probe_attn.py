"""Is attention counted in FORWARD but not BACKWARD (or vice versa) by FlopCounterMode?

This decides whether ANY analytic estimator can be pinned to FlopCounterMode
on this torch build.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__
print("src ->", src.__file__, "| torch", torch.__version__)
print()


def both(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def split(forward_fn):
    """(fwd, fwd+bwd) measured in two separate processes-equivalent runs."""
    f = forward_fn()
    fwd = forward_fn()  # rebuild
    with FlopCounterMode(display=False) as fc:
        out = forward_fn()
        out.backward()
    del fwd
    return f, fc.get_total_flops()


B, H, L, C = 4, 12, 64, 768
attn_fwd = 4 * B * L * L * C          # QK^T + AV, full precision
print("reference attention flops for (B=%d,H=%d,L=%d,C=%d) = %.6e" % (B, H, L, C, attn_fwd))

print("\n--- encoder-style Attention (custom, SDPA), train mode ---")
a = nn.Module()
attn = None
from src.models.encoder import Attention as EncAttn

attn = EncAttn(dim=C, num_heads=H)
attn.train()
x = torch.randn(B, L, C, requires_grad=True)
f_only = both(lambda: attn(x))
with FlopCounterMode(display=False) as fc:
    y = attn(x)
    y.sum().backward()
fwd_bwd = fc.get_total_flops()
lin = 2 * B * L * (4 * C * C)
print("fwd        = %.6e  (linears only would be %.6e, ratio %.4f)" % (f_only, lin, f_only / lin))
print("fwd+bwd    = %.6e" % fwd_bwd)
print("implied bwd= %.6e  = %.4f x fwd" % (fwd_bwd - f_only, (fwd_bwd - f_only) / f_only))
print("bwd vs 2*attn_fwd = %.4f   (2*bwd vs 2*attn_fwd ratio %.4f)" % (
    (fwd_bwd - f_only) / (2 * attn_fwd), (fwd_bwd - f_only) / (2 * attn_fwd)))

print("\n--- same, eval mode (fused fast path) ---")
attn.eval()
x = torch.randn(B, L, C, requires_grad=True)
f_only = both(lambda: attn(x))
with FlopCounterMode(display=False) as fc:
    y = attn(x)
    y.sum().backward()
print("fwd=%.6e fwd+bwd=%.6e implied bwd=%.6e" % (
    f_only, fc.get_total_flops(), fc.get_total_flops() - f_only))

print("\n--- raw SDPA, train-style explicit path ---")
import torch.nn.functional as F

q = torch.randn(B, H, L, C // H, requires_grad=True)
o1 = both(lambda: F.scaled_dot_product_attention(q, q, q))
with FlopCounterMode(display=False) as fc:
    F.scaled_dot_product_attention(q, q, q).sum().backward()
print("sdpa fwd=%.6e fwd+bwd=%.6e implied bwd=%.6e (2*attn=%.6e)" % (
    o1, fc.get_total_flops(), fc.get_total_flops() - o1, 2 * attn_fwd))

print("\n--- raw bmm path (no SDPA) ---")
qa = torch.randn(B * H, L, C // H, requires_grad=True)
ka = torch.randn(B * H, L, C // H, requires_grad=True)
o2 = both(lambda: qa @ ka.transpose(-2, -1))
with FlopCounterMode(display=False) as fc:
    (qa @ ka.transpose(-2, -1)).sum().backward()
print("bmm fwd=%.6e fwd+bwd=%.6e implied bwd=%.6e" % (
    o2, fc.get_total_flops(), fc.get_total_flops() - o2))
