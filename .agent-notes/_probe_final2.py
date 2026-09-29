"""Ratio with diagnostics OFF, at several small configs, plus timing."""

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402
from src.utils.flops import estimate_jepa_step_flops  # noqa: E402


class Off(nn.Module):
    def compute(self, *a, **k):
        return {}


def one(D, NE, DP, NP, H, R, off, ref, T, B, V, mr, nodiag):
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=T, embed_dim=D, encoder_depth=NE, num_heads=H,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=off, num_refine_steps=ref,
    )
    torch.manual_seed(0)
    m = TextSpanJEPA(cfg)
    m.train()
    if nodiag:
        m.diagnostics = Off()
        m.jspace_metrics = Off()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, V, (B, T), generator=g)
    n = max(1, int(round(T * mr)))
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        mask[b, 1 : 1 + n] = 1
    masked = ids.clone()
    masked[mask.bool()] = 0
    t0 = time.monotonic()
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    dt = time.monotonic() - t0
    est = estimate_jepa_step_flops(
        embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
        seq_len=T, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
        num_refine_steps=ref, future_offsets=off, mask_ratio=n / T,
    )
    return fc.get_total_flops(), est["total_flops"], dt


CASES = [
    (32, 2, 16, 2, 4, 2.0, (1, 4), 1, 16, 2, 97, 3 / 16),
    (32, 2, 16, 2, 4, 2.0, (1, 4), 1, 32, 2, 97, 0.25),
    (64, 3, 32, 2, 4, 4.0, (1, 4), 2, 32, 4, 256, 0.25),
    (48, 2, 24, 3, 4, 2.0, (2, 8), 2, 24, 3, 128, 0.2),
    (32, 2, 16, 2, 4, 2.0, (1, 4), 1, 16, 2, 97, 0.0),
]
print("{:>6} {:>16} {:>10} {:>16} {:>10} {:>7}".format(
    "case", "meas(diag on)", "ratio", "meas(diag off)", "ratio", "secs"))
for i, c in enumerate(CASES):
    a, est, t1 = one(*c, False)
    b, _, t2 = one(*c, True)
    print("{:>6} {:>16.6e} {:>10.5f} {:>16.6e} {:>10.5f} {:>7.2f}".format(
        i, a, est / a, b, est / b, t1 + t2))
