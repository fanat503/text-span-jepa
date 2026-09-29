"""Micro-diagnostic 2: attention accounting + encoder fwd/bwd multiplier."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import src
from src.models.encoder import TextSpanJEPAEncoder

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__
print("src ->", src.__file__, "torch", torch.__version__)


def dump(name, fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    tot = fc.get_total_flops()
    detail = "  ".join(
        "{}={:.4e}".format(op, sum(d.values())) for op, d in fc.get_flop_counts().items()
    )
    print("{:>34}  total={:>14.6e}   {}".format(name, tot, detail))
    return tot


B, H, T, D = 4, 8, 128, 64
q = torch.randn(B, H, T, D, requires_grad=True)
dump("sdpa fwd", lambda: F.scaled_dot_product_attention(q, q, q))
dump("sdpa fwd+bwd", lambda: F.scaled_dot_product_attention(q, q, q).sum().backward())
qb = q.transpose(1, 2)
dump("bmm attn fwd", lambda: q @ q.transpose(-2, -1))
dump("bmm attn fwd+bwd", lambda: (q @ q.transpose(-2, -1)).sum().backward())

mha = nn.MultiheadAttention(D, H, batch_first=True)
z = torch.randn(B, T, D, requires_grad=True)
dump("mha fwd", lambda: mha(z, z, z, need_weights=False))
dump("mha fwd+bwd", lambda: mha(z, z, z, need_weights=False)[0].sum().backward())
mha.eval()
dump("mha(eval) fwd", lambda: mha(z, z, z, need_weights=False))
dump("mha(eval) fwd+bwd", lambda: mha(z, z, z, need_weights=False)[0].sum().backward())
mha.train()

print()
enc = TextSpanJEPAEncoder(vocab_size=256, embed_dim=64, depth=2, num_heads=8, max_seq_len=64)
enc.train()
ids = torch.randint(0, 256, (4, 32))
f = dump("encoder fwd", lambda: enc(ids))
with FlopCounterMode(display=False) as fc:
    h, _ = enc(ids)
    h.sum().backward()
fb = fc.get_total_flops()
print("encoder fwd+bwd =", fb, " ratio =", fb / f)
h2, _ = enc(ids)
with FlopCounterMode(display=False) as fc:
    enc.encoder_frozen_probe = None
    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
print("analytic linear-only 2 blocks, d=64, B*T=128:",
      24 * 128 * 64 * 64 * 2)
print("analytic + attention 2 layers: ", 4 * 4 * 32 * 32 * 64 * 2)
