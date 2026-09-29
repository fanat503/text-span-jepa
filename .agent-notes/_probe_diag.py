"""Quantify the diagnostics term: stub CollapseDiagnostics + JSpaceMetrics."""

import os
import sys

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


def run(T, off):
    D, NE, DP, NP, H, R = 768, 12, 384, 6, 12, 4.0
    OFF, REF, B, V, MR = (1, 4, 16), 3, 4, 4096, 0.35
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=512, embed_dim=D, encoder_depth=NE, num_heads=H,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=OFF, num_refine_steps=REF,
    )
    torch.manual_seed(0)
    m = TextSpanJEPA(cfg)
    m.train()
    if off:
        m.diagnostics = Off()
        m.jspace_metrics = Off()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, V, (B, T), generator=g)
    n = max(1, int(T * MR))
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        s = (b * 3) % max(1, T - n)
        mask[b, s : s + n] = 1
    masked = ids.clone()
    masked[mask.bool()] = 0
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    est = estimate_jepa_step_flops(
        embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
        seq_len=T, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
        num_refine_steps=REF, future_offsets=OFF, mask_ratio=MR,
    )
    return fc.get_total_flops(), est["total_flops"], n


print("{:>5} {:>16} {:>16} {:>9} {:>16} {:>9} {:>16}".format(
    "T", "meas_full", "est", "ratio", "meas_nodiag", "ratio", "diag_cost"))
for T in (128, 256, 512):
    full, est, _ = run(T, False)
    nod, _, _ = run(T, True)
    print("{:>5} {:>16.6e} {:>16.6e} {:>9.5f} {:>16.6e} {:>9.5f} {:>16.6e}".format(
        T, full, est, est / full, nod, est / nod, full - nod))

print()
print("diagnostics as a multiple of 2*B*T*D^2 (forward) and 3x (fwd+bwd):")
D, B = 768, 4
for T in (128, 256, 512):
    full, est, _ = run(T, False)
    nod, _, _ = run(T, True)
    d = full - nod
    print("  T={:<4} diag={:.6e}   diag/(2*B*T*D^2)={:.4f}   diag/(6*B*T*D^2)={:.4f}".format(
        T, d, d / (2 * B * T * D * D), d / (6 * B * T * D * D)))
