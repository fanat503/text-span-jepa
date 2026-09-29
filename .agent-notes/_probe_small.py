"""Dispatch-level exact accounting on a SMALL config, compared to the analytic.

Small dims mean the residual is a large fraction, so it is identifiable.
"""

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

B, D, NE, NP, DP, V, HEADS, R = 2, 32, 2, 2, 16, 97, 4, 2.0
OFFSETS, REFINE = (1, 4), 1


def build(T):
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=T, embed_dim=D, encoder_depth=NE, num_heads=HEADS,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=OFFSETS, num_refine_steps=REFINE,
    )
    torch.manual_seed(0)
    m = TextSpanJEPA(cfg)
    m.train()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, V, (B, T), generator=g)
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, 1:4] = 1
    masked = ids.clone()
    masked[mask.bool()] = 0
    return m, masked, ids, mask


class Rec(TorchDispatchMode):
    def __init__(self):
        self.rec = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        n = str(func)
        if n in ("aten.mm.default", "aten.addmm.default", "aten.bmm.default",
                 "aten.baddbmm.default", "aten.linear.default",
                 "aten._scaled_dot_product_attention.default",
                 "aten._scaled_dot_product_flash_attention.default"):
            self.rec.append((n, [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]))
        return func(*args, **kwargs)


for T in (16,):
    m, mk, ids, mask = build(T)
    r = Rec()
    with r:
        loss, _i, _d = m.compute_loss_with_targets(mk, ids, mask)
        loss.backward()
    agg = defaultdict(int)
    for n, ts in r.rec:
        agg[(n, tuple(ts))] += 1
    total = 0
    for (n, ts), c in sorted(agg.items(), key=lambda kv: -len(str(kv[0])) * kv[1]):
        print("  {:>44} x{}".format(str(ts) + " " + n.split(".")[1], c))
    print("\nT={} recorded {} distinct call shapes".format(T, len(agg)))
