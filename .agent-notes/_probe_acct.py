"""Assemble the analytic accounting and compare to FlopCounterMode step-by-step."""

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
OFFSETS = (1, 4, 16)
REFINE = 3
MASK = 0.35


def analytic(T, mask_ratio=MASK):
    n_tok = B * T
    enc = 24 * n_tok * D * D * NE                      # qkv(3D^2)+proj(D^2)+mlp(8D^2)
    enc_step = 3 * enc                                  # fwd + 2x bwd
    tenc = 1 * enc                                      # target encoder: no_grad forward only
    passes = REFINE * T + sum(T - d for d in OFFSETS if d < T)
    pred = 24 * B * passes * DP * DP * NP               # predictor blocks
    pred += 2 * B * T * D * DP                          # predictor_embed
    pred += 2 * B * D * DP * (T + sum(T - d for d in OFFSETS if d < T))  # predictor_proj
    pred_step = 3 * pred
    n_masked = int(n_tok * mask_ratio)
    dec = 2 * n_masked * (4 * D * D + D * V)            # D->2D, 2D->D, D->V
    dec_step = 3 * dec
    return enc_step, tenc, pred_step, dec_step


def measure(T, mask_ratio=MASK):
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
    n = max(1, int(T * mask_ratio))
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        s = (b * 3) % max(1, T - n)
        mask[b, s : s + n] = 1
    masked = ids.clone()
    masked[mask.bool()] = 0
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    return fc.get_total_flops(), float(masked[mask.bool()].numel() and mask.sum())


print("{:>5} {:>16} {:>16} {:>8}".format("T", "measured", "analytic_sum", "ratio"))
for T in (128, 256, 512):
    meas, nmasked = measure(T)
    enc, tenc, pred, dec = analytic(T)
    tot = enc + tenc + pred + dec
    print("{:>5} {:>16.6e} {:>16.6e} {:>8.4f}".format(T, meas, tot, tot / meas))
    print("      enc={:.4e} tenc={:.4e} pred={:.4e} dec={:.4e}  n_masked={}".format(
        enc, tenc, pred, dec, nmasked))

print("\n--- sensitivity: is the residual the regularizers (variance/cov)? ---")
for T in (128, 512):
    meas, _ = measure(T)
    base = sum(analytic(T))
    print("T={} measured={:.6e} base={:.6e} residual={:.6e} ({:.2%})".format(
        T, meas, base, meas - base, (meas - base) / meas))
    print("   candidate: 2*B*T*D*D (variance) =", 2 * B * T * D * D,
          " D^3-ish cov:", 2 * (B * T) * D * D)
