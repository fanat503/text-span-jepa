"""Exact per-module fwd+bwd attribution via FlopCounterMode(depth=...)."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

B, D, NE, NP, DP, V, HEADS, R = 4, 768, 12, 6, 384, 4096, 12, 4.0
OFFSETS, REFINE, MASK = (1, 4, 16), 3, 0.35


def make(T):
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=512, embed_dim=D, encoder_depth=NE, num_heads=HEADS,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=OFFSETS, num_refine_steps=REFINE,
    )
    torch.manual_seed(0)
    m = TextSpanJEPA(cfg)
    m.train()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, V, (B, T), generator=g)
    n = max(1, int(T * MASK))
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        s = (b * 3) % max(1, T - n)
        mask[b, s : s + n] = 1
    masked = ids.clone()
    masked[mask.bool()] = 0
    return m, masked, ids, mask


T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
m, masked, ids, mask = make(T)
with FlopCounterMode(mods=[m], depth=4, display=False) as fc:
    loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
    loss.backward()
total = fc.get_total_flops()
print("T={} total={:.6e}".format(T, total))
agg = {}
for op, mods in fc.get_flop_counts().items():
    for mod, f in mods.items():
        agg[mod] = agg.get(mod, 0.0) + f
s = 0.0
for mod, f in sorted(agg.items(), key=lambda kv: -kv[1]):
    if f <= 0:
        continue
    s += f
    print("  {:>50} {:>14.6e}  {:>6.3%}".format(str(mod)[:50], f, f / total))
print("  sum={:.6e}".format(s))
