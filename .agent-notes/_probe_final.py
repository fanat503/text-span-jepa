"""Decisive: per-component forward vs fwd+bwd, and the assembled step accounting."""

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


for T in (128, 256, 512):
    m, masked, ids, mask = make(T)
    offs = [d for d in OFFSETS if d < T]
    S = REFINE * T + sum(T - d for d in offs)      # total predictor token-passes

    e_enc = 24 * B * T * D * D * NE
    e_pred = (
        24 * B * S * DP * DP * NP
        + 2 * B * T * D * DP * (1 + len(offs))                       # predictor_embed x (1+n)
        + 2 * B * DP * D * (T + sum(T - d for d in offs))            # predictor_proj
    )
    nm = B * int(T * MASK)
    e_dec = 2 * nm * (4 * D * D + D * V)
    e_cov = 2 * B * T * D * D
    pred_total = 3 * (e_enc + e_pred + e_dec + e_cov) + e_enc

    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    meas = fc.get_total_flops()
    print("T={:<4} measured={:.6e}  model={:.6e}  ratio={:.5f}  delta={:.4e}".format(
        T, meas, pred_total, pred_total / meas, meas - pred_total))

    # per-component fwd and fwd+bwd in isolation
    m2, mk2, i2, ms2 = make(T)
    with FlopCounterMode(display=False) as f1:
        ho, te = m2.encoder(mk2)
    with FlopCounterMode(display=False) as f2:
        ho.sum().backward()
    m2.zero_grad(set_to_none=True)
    print("      encoder fwd={:.6e} bwd={:.6e} ratio={:.4f}  analytic_fwd={:.6e}".format(
        f1.get_total_flops(), f2.get_total_flops(),
        f1.get_total_flops() / f2.get_total_flops(), e_enc))
    with torch.no_grad():
        ht, _ = m2.target_encoder(i2)
    ho2, te2 = m2.encoder(mk2)
    with FlopCounterMode(display=False) as f3:
        out = m2.predictor(ho2, ms2, te2, ht)
    p = out[0]
    with FlopCounterMode(display=False) as f4:
        sum(x.sum() for x in out[3].values()) + p.sum()
    with FlopCounterMode(display=False) as f5:
        (sum(x.sum() for x in out[3].values()) + p.sum()).backward()
    print("      predictor fwd={:.6e} bwd={:.6e} ratio={:.4f}  analytic_fwd={:.6e}".format(
        f3.get_total_flops(), f5.get_total_flops(),
        f3.get_total_flops() / f5.get_total_flops(), e_pred))
