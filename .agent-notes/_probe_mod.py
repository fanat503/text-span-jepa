"""Per-module FLOP attribution on a SMALL model, forward only (no_grad)."""

import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402

D, NE, DP, NP, H, R = 32, 2, 16, 2, 4, 2.0
OFF, REF, T, B, V, MR = (1, 4), 1, 16, 2, 97, 0.2
n_mask = int(round(T * MR))
mask = torch.zeros(B, T, dtype=torch.long)
for b in range(B):
    mask[b, 1 : 1 + n_mask] = 1

cfg = TextSpanJEPAConfig(
    vocab_size=V, max_seq_len=T, embed_dim=D, encoder_depth=NE, num_heads=H,
    mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
    future_offsets=OFF, num_refine_steps=REF,
)
torch.manual_seed(0)
m = TextSpanJEPA(cfg)
m.train()
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, V, (B, T), generator=g)
masked = ids.clone()
masked[mask.bool()] = 0

print("predictor block mlp:", m.predictor.predictor_blocks[0].mlp)
print()

with FlopCounterMode(depth=4, display=False) as fc:
    with torch.no_grad():
        m.compute_loss_with_targets(masked, ids, mask)
total = fc.get_total_flops()

agg = defaultdict(float)
for op, mods in fc.get_flop_counts().items():
    for mod, f in mods.items():
        if f:
            agg[mod] += f
print("total fwd = {}".format(total))
for mod, f in sorted(agg.items(), key=lambda kv: -kv[1]):
    if f > 0:
        print("  {:>58} {:>12.0f}".format(str(mod)[:58], f))
