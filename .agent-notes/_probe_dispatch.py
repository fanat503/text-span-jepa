"""Exact attribution: record every mm/addmm/bmm shape during one real step."""

import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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


class Recorder(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name in ("aten.mm.default", "aten.addmm.default", "aten.bmm.default",
                    "aten.baddbmm.default"):
            ts = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
            if len(ts) >= 2:
                self.calls.append((name, ts))
        return func(*args, **kwargs)


T = 256
m, masked, ids, mask = make(T)
rec = Recorder()
with rec:
    loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
    loss.backward()

# Collapse: (op, m, k, n) -> count
agg = defaultdict(int)
for name, ts in rec.calls:
    if len(ts) < 2:
        continue
    try:
        if name.startswith("aten.b"):
            (b_, m_, k_), (b2, n_, k2) = ts[0], ts[1]
            agg[("bmm", m_, k_, n_)] += 1
        else:
            (m_, k_), (m2, n_) = ts[0], ts[1]
            agg[(name.split(".")[1], m_, k_, n_)] += 1
    except ValueError:
        pass

total = 0
rows = []
items = sorted(agg.items(), key=lambda kv: -(kv[0][1] * kv[0][2] * kv[0][3] * kv[1]))
for key, c in items:
    op, mm, kk, nn = key
    f = 2 * mm * kk * nn * c
    total += f
    rows.append((op, mm, kk, nn, c, f))

print("T={}  total matmul flops = {:.6e}  ({} distinct shapes)".format(T, total, len(rows)))
print("{:>8} {:>7} {:>7} {:>7} {:>5} {:>15} {:>7}".format(
    "op", "m", "k", "n", "cnt", "flops", "share"))
for op, mm, kk, nn, c, f in rows[:25]:
    print("{:>8} {:>7} {:>7} {:>7} {:>5} {:>15.6e} {:>6.2%}".format(
        op, mm, kk, nn, c, f, f / total))
