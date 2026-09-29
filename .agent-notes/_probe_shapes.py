"""Diff the EXPECTED matmul shape multiset against the RECORDED one.

Run only the forward (no backward) so the shapes are easy to enumerate.
"""

import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402


class Rec(TorchDispatchMode):
    def __init__(self):
        self.c = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        n = str(func)
        op = n.split(".")[1]
        ts = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        if op in ("mm", "bmm") and len(ts) >= 2:
            self.c[(op, ts[0], ts[1])] += 1
        elif op in ("addmm", "baddbmm") and len(ts) >= 3:
            # (bias, mat1, mat2, beta, alpha)
            self.c[(op, ts[1], ts[2])] += 1
        return func(*args, **kwargs)


D, NE, DP, NP, H, R = 32, 2, 16, 2, 4, 2.0
OFF, REF, T, B, V, MR = (1, 4), 1, 16, 2, 97, 0.2
offs = [d for d in OFF if d < T]
n_mask = int(round(T * MR))
NM = B * n_mask
BT = B * T

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
mask = torch.zeros(B, T, dtype=torch.long)
for b in range(B):
    mask[b, 1 : 1 + n_mask] = 1
masked = ids.clone()
masked[mask.bool()] = 0

rec = Rec()
with rec:
    with torch.no_grad():
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)

print("FORWARD ONLY (no_grad) -- recorded shapes, flops descending")
rows = []
for (op, s0, s1), c in rec.c.items():
    f = 2 * s0[0] * s0[1] * s1[1] * c
    rows.append((f, op, s0, s1, c))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
for f, op, s0, s1, c in rows:
    print("  {:>14.0f}  {:>6} x{}  {} @ {}".format(f, op, c, s0, s1))
print("  TOTAL = {:.0f}".format(tot))
print("\n  B*T={}  D={}  n_masked={}  3D={}  4D={}  DP={}  3DP={}  4DP={}".format(
    BT, D, NM, 3 * D, 4 * D, DP, 3 * DP, 4 * DP))
